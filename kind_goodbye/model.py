"""The reference model wrapper (transformers): loading, chat-template encoding, batched sampling, and forward-pass scoring with prefix reuse.

Modes: "bf16" (plain), "int8" (bitsandbytes 8-bit weights) and "cpuemb" (bf16 with the tied embedding table on the CPU, which lets Gemma 3
12B's bf16 forward passes fit a 24 GiB card; scoring only). The rating digits and the yes/no answers are read where the chat template
starts a reply that is just the answer: for Gemma 3, Gemma 4 and Qwen3.8 that is right after the generation prompt.

Prefix reuse (extend, snapshot/restore, fan): with a chunk size, a prefill from 0 runs in chunks ending at multiples of it, so a cache
rolled back to such a boundary and run on from there gives every later forward the shape and contents it has in that prefill: bit-identical,
where a split anywhere else would not be.
"""

import copy
import gc
import logging
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList



GEN_KW = dict(do_sample=True, temperature=1.0, top_k=64, top_p=0.95)  # Gemma 3's generation_config defaults
# 8-bit matmuls log a bf16-to-fp16 input cast on every forward pass (the normal int8 path): gigabytes of log per run
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)

TURN_END = ["<end_of_turn>", "<|eot_id|>"]          # the token that ends a turn (Gemma 3 first)
TURN_OPEN = ["<start_of_turn>", "<|start_header_id|>"]
# A vocabulary with neither of TURN_END (ChatML models such as Qwen): its turn end -> the tokens that open a turn. Checked only then,
# since Gemma's vocabulary also holds a "</s>".
MORE_TURNS = {"<|im_end|>": ["<|im_start|>"], "</s>": ["[INST]", "[/INST]"]}


def text_model(m):
    """The decoder stack (has .layers and .embed_tokens): inside the multimodal wrapper or not."""
    return m.model.language_model if hasattr(m.model, "language_model") else m.model


def drop_vision(m):
    """Delete a multimodal Gemma 3's image encoder and projector: they run only on pixel_values, which never come."""
    if getattr(m.model, "vision_tower", None) is not None:
        del m.model.vision_tower, m.model.multi_modal_projector
        gc.collect()
        torch.cuda.empty_cache()


class _Steps(StoppingCriteria):
    """Counts decode steps and stops nothing: an out-of-memory error at step 0 came in the prefill."""

    def __init__(self):
        self.n = 0

    def __call__(self, input_ids, scores, **kw):
        self.n += 1
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


def _leaves(tree):
    for v in tree.values():
        yield from _leaves(v) if isinstance(v, dict) else [v]


class LM:
    def __init__(self, mid, mode="bf16", chunk=None):
        assert mode in ("bf16", "int8", "cpuemb"), mode
        tok = AutoTokenizer.from_pretrained(mid)
        if mode == "cpuemb":
            m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cpu")
            assert m.config.get_text_config().tie_word_embeddings, "cpuemb needs tied embeddings"
            for name, child in text_model(m).named_children():
                if name != "embed_tokens":
                    child.to("cuda")
        else:
            q = BitsAndBytesConfig(load_in_8bit=True) if mode == "int8" else None
            m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda", quantization_config=q)
        drop_vision(m)
        self._setup(mid, m, tok, mode, chunk, "cuda")

    @classmethod
    def from_parts(cls, model, tok, mode="bf16", chunk=None, dev="cpu", mid="test"):
        """Wrap an already-built model (used by the CPU tests with a tiny random Gemma 3)."""
        self = cls.__new__(cls)
        self._setup(mid, model, tok, mode, chunk, dev)
        return self

    def _setup(self, mid, m, tok, mode, chunk, dev):
        self.mid, self.mode, self.chunk, self.dev = mid, mode, chunk, dev
        self.model, self.tok = m.eval(), tok
        self.gen_kw = GEN_KW  # generate()'s sampling; loading.gen_lm puts a config's "gen_kw" here
        self.tok.padding_side = "left"
        self.lm = text_model(m)
        self.E = self.lm.embed_tokens
        self.layers = self.lm.layers
        self.n_layers = len(self.layers)
        self.d = m.config.get_text_config().hidden_size
        vocab = tok.get_vocab()
        ends, opens = TURN_END, TURN_OPEN
        if not any(t in vocab for t in TURN_END):
            end = next(t for t in MORE_TURNS if t in vocab)
            ends, opens = [end], MORE_TURNS[end]
        self.eot = next(vocab[t] for t in ends if t in vocab)
        self.turn_open = [vocab[t] for t in opens if t in vocab]
        gen_eos = m.generation_config.eos_token_id if m.generation_config is not None else None
        gen_eos = gen_eos if isinstance(gen_eos, list) else [gen_eos] if gen_eos is not None else []
        self.stop_ids = sorted({self.eot, tok.eos_token_id, *gen_eos} - {None})
        if tok.pad_token_id is None:  # left padding needs a pad token
            tok.pad_token = tok.convert_ids_to_tokens(self.eot)
        self.pad_id = tok.pad_token_id
        # Control tokens spelled out in plain text would become real control tokens when the
        # transcript is re-tokenized from text; generated text containing one is flagged.
        self.special_strings = [t for t in tok.all_special_tokens if t] + ends + opens
        self.digit_lead, self.digit_ids = self._answer_tokens([str(k) for k in range(1, 8)])

    def _answer_tokens(self, answers):
        """(lead, ids): the chat template renders a reply that is just answers[i] after the generation prompt as the tokens
        lead, the same for every answer, then the single token ids[i] and the end of the turn. For the models here the lead
        is empty and each answer is one token."""
        q = [{"role": "user", "content": "?"}]
        head = self.ids(self.text(q, True))
        tails = []
        for a in answers:
            full = self.ids(self.text(q + [{"role": "assistant", "content": a}], False))
            assert full[:len(head)] == head, (a, "the template's reply does not extend its generation prompt")
            tails.append(full[len(head):])
        n = next(i for i, col in enumerate(zip(*tails)) if len(set(col)) > 1)
        ids = [t[n] for t in tails]
        assert all(len(t) > n + 1 and t[n + 1] == self.eot for t in tails) and len(set(ids)) == len(ids), (answers, tails)
        return tails[0][:n], ids

    # ---------- encoding ----------

    def text(self, messages, gen_prompt):
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=gen_prompt)

    def ids(self, text):
        """The template text already starts with the BOS token; adding special tokens again would double it. A tokenizer
        without a BOS token (Qwen's) has none to check."""
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        bos = self.tok.bos_token_id
        assert bos is None or ids[0] == bos, "BOS"
        return ids

    # ---------- generation ----------

    @torch.no_grad()
    def generate(self, messages_list, max_new_tokens, seed, gen_kw=None):
        """Sampled replies, with gen_kw or else self.gen_kw. On out-of-memory the batch is split in half and retried, so each result
        records the batch size and seed it was actually generated with. Each result also carries the
        whole call's wall seconds (call_s) and its out-of-memory splits (ooms: batch size, phase prefill
        or decode, decode steps done, seconds lost), which are also printed as they happen."""
        assert self.mode != "cpuemb", "cpuemb is for measurement only"
        gen_kw = self.gen_kw if gen_kw is None else gen_kw
        t0, ooms = time.time(), []
        res = self._split(messages_list, max_new_tokens, seed, gen_kw, ooms)
        return [{**o, "call_s": round(time.time() - t0, 2), "ooms": list(ooms)} for o in res]

    def _split(self, messages_list, max_new_tokens, seed, gen_kw, ooms):
        """The retry happens after the except block, so the failed attempt's tensors can be freed first."""
        t0, self._steps = time.time(), None
        try:
            return self._generate(messages_list, max_new_tokens, seed, gen_kw)
        except torch.OutOfMemoryError:
            if len(messages_list) == 1:
                raise
        n = None if self._steps is None else self._steps.n
        ooms.append({"batch": len(messages_list), "phase": None if n is None else "decode" if n else "prefill", "step": n,
                     "lost_s": round(time.time() - t0, 2)})
        print(f"generate: out of memory {ooms[-1]}, splitting the batch", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        h = len(messages_list) // 2
        return (self._split(messages_list[:h], max_new_tokens, seed * 2 + 1, gen_kw, ooms)
                + self._split(messages_list[h:], max_new_tokens, seed * 2 + 2, gen_kw, ooms))

    def _generate(self, messages_list, max_new_tokens, seed, gen_kw):
        texts = [self.text(m, True) for m in messages_list]
        enc = self.tok(texts, add_special_tokens=False, return_tensors="pt", padding=True).to(self.dev)
        torch.manual_seed(seed)
        chunked = {"prefill_chunk_size": self.chunk} if self.chunk else {}  # one-shot prefill of a long prompt runs out of memory
        self._steps = _Steps()
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, eos_token_id=self.stop_ids,
                                  pad_token_id=self.pad_id, stopping_criteria=StoppingCriteriaList([self._steps]), **chunked, **gen_kw)
        new = out[:, enc["input_ids"].shape[1]:].tolist()
        res = []
        for row, n_in in zip(new, enc["attention_mask"].sum(1).tolist()):
            stop = next((i for i, t in enumerate(row) if t in self.stop_ids), None)
            body = row[:stop] if stop is not None else row
            cut = next((i for i, t in enumerate(body) if t in self.turn_open), None)
            if cut is not None:  # a sampled turn opener would break the transcript; cut there and flag
                body = body[:cut]
            text = self.tok.decode(body, skip_special_tokens=True).strip()
            res.append({"text": text, "n_tokens": len(body),
                        "truncated": stop is None and cut is None, "stray_turn_token": cut is not None,
                        "special_string": any(x in text for x in self.special_strings),
                        "prompt_tokens": n_in, "gen_batch": len(texts), "gen_seed": seed})
        return res

    # ---------- forward passes ----------

    def _fwd(self, ids, past, all_logits=False):
        """One forward over ids (a 1 x n LongTensor on CPU), continuing past. Returns float32 CPU
        logits for the last position (or every position) and the cache."""
        if self.mode == "cpuemb":
            out = self.lm(inputs_embeds=self.E(ids).to(self.dev), past_key_values=past, use_cache=True)
            h = out.last_hidden_state[0] if all_logits else out.last_hidden_state[0, -1:]
            logits = (h.cpu() @ self.E.weight.T).float()
        else:
            out = self.model(input_ids=ids.to(self.dev), past_key_values=past, use_cache=True, logits_to_keep=0 if all_logits else 1)
            logits = out.logits[0].float().cpu()
        return (logits if all_logits else logits[-1]), out.past_key_values

    @torch.no_grad()
    def prefill(self, ids, past=None):
        ids = torch.tensor([ids])
        step = self.chunk or ids.shape[1]
        logits = None
        for s in range(0, ids.shape[1], step):
            logits, past = self._fwd(ids[:, s:s + step], past)
        return logits, past

    @torch.no_grad()
    def fork(self, prefix_ids, prefix_cache, full_ids):
        """Logits after full_ids, reusing the cache of prefix_ids. The cache is copied, so the
        prefix state is never modified by a fork."""
        assert full_ids[:len(prefix_ids)] == prefix_ids, "fork does not extend the prefix"
        return self.prefill(full_ids[len(prefix_ids):], copy.deepcopy(prefix_cache))

    # ---------- prefix reuse ----------

    @torch.no_grad()
    def extend(self, ids, start, past, stop=None, cap=None):
        """Run ids[start:stop] on from a cache holding ids[:start] (start a multiple of the chunk size) in
        chunks ending at multiples of it, as a prefill from 0 runs them. Returns the last logits (None if
        nothing ran) and the cache; cap, a Capture, gets each chunk's offset."""
        logits, stop = None, len(ids) if stop is None else stop
        for s in range(start, stop, self.chunk):
            if cap is not None:
                cap.offset = s
            logits, past = self._fwd(torch.tensor([ids[s:min(s + self.chunk, stop)]]), past)
        return logits, past

    @staticmethod
    def snapshot(past):
        """What restore needs to roll past back to this point: the length of each full-attention layer
        (they only grow, by concatenation) and a copy of each sliding window (replaced, never written in
        place). None stands for the empty cache."""
        if past is None:
            return None
        return [(x.keys.clone(), x.values.clone(), x.cumulative_length) if x.is_sliding else x.keys.shape[-2] for x in past.layers]

    @staticmethod
    def restore(past, snap):
        """past rolled back to snap. Snapshots nest: restoring one drops those taken after it."""
        if snap is None:
            return None
        for x, s in zip(past.layers, snap):
            if x.is_sliding:
                x.keys, x.values, x.cumulative_length = s
            else:
                x.keys, x.values = x.keys[..., :s, :], x.values[..., :s, :]
        return past

    @torch.no_grad()
    def fan(self, tree, leaf, past=None, pos=0):
        """leaf(key, logits, past) -> (result, past) for each token-id list in tree (key -> list, or key
        -> subtree), with the logits after the whole list. Each subtree's common prefix, cut back to a
        chunk boundary, is prefilled once. past holds the first pos tokens of every list (pos a multiple
        of the chunk size) and is consumed. Returns {leaf key: result} in tree order."""
        seqs = list(_leaves(tree))
        lo, hi = min(seqs), max(seqs)  # the common prefix of the lexicographic extremes is everyone's
        n = next((i for i, (x, y) in enumerate(zip(lo, hi)) if x != y), min(len(lo), len(hi)))
        B = min(n, min(map(len, seqs)) - 1) // self.chunk * self.chunk
        assert B >= pos, (B, pos)
        past = self.extend(seqs[0], pos, past, stop=B)[1]
        snap = self.snapshot(past) if len(tree) > 1 else None
        out = {}
        for i, (k, v) in enumerate(tree.items()):
            if i:
                past = self.restore(past, snap)
            if isinstance(v, dict):
                out.update(self.fan(v, leaf, past, B))
            else:
                logits, past = self.extend(v, B, past)
                out[k], past = leaf(k, logits, past)
        return out

    # ---------- scoring ----------

    @staticmethod
    def _dist(logits, ids):
        return torch.log_softmax(logits, -1)[ids].exp()

    def digit_score(self, logits):
        """probs and logp are renormalised over the digits 1-7; logp comes straight from the digit
        logits, so it stays finite where probs underflow."""
        p = self._dist(logits, self.digit_ids)
        q = p / p.sum()
        return {"expected": float((q * torch.arange(1, 8)).sum()), "probs": [float(x) for x in q],
                "logp": [float(x) for x in torch.log_softmax(logits[self.digit_ids], -1)],
                "mass": round(float(p.sum()), 5), "argmax": int(q.argmax()) + 1}
