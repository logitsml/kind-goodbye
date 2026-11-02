"""A tiny random Gemma 3 text model and a byte-level tokenizer with Gemma 3's chat format, built in memory (no download), for the CPU
tests. The tokenizer has one token per byte, Gemma 3's control tokens, and "yes" and "no" as single tokens (the yes/no reads need them);
its chat template is Gemma 3's for text messages. The model's weights are random (seeded), so its replies are noise, which is all the
pipeline tests need."""
import json

GEMMA3_TEMPLATE = (
    "{{ bos_token }}"
    "{%- if messages[0]['role'] == 'system' -%}"
    "{%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}"
    "{%- set loop_messages = messages[1:] -%}"
    "{%- else -%}"
    "{%- set first_user_prefix = '' -%}"
    "{%- set loop_messages = messages -%}"
    "{%- endif -%}"
    "{%- for message in loop_messages -%}"
    "{%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{%- endif -%}"
    "{%- if message['role'] == 'assistant' -%}{%- set role = 'model' -%}{%- else -%}{%- set role = message['role'] -%}{%- endif -%}"
    "{{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else '') }}"
    "{{ message['content'] | trim }}"
    "{{ '<end_of_turn>\n' }}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{ '<start_of_turn>model\n' }}{%- endif -%}"
)
SPECIAL = ["<pad>", "<eos>", "<bos>", "<start_of_turn>", "<end_of_turn>"]
WORDS = ["yes", "no"]


def tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {t: i for i, t in enumerate(SPECIAL)}
    for ch in sorted(alphabet):
        vocab[ch] = len(vocab)
    merges = [("y", "e"), ("ye", "s"), ("n", "o")]
    for a, b in merges:
        vocab[a + b] = len(vocab)
    tk = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tk.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>",
                                  additional_special_tokens=["<start_of_turn>", "<end_of_turn>"])
    tok.chat_template = GEMMA3_TEMPLATE
    return tok


def model(tok, seed=0):
    import torch
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig
    cfg = Gemma3TextConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=2,
                           num_key_value_heads=1, head_dim=32, sliding_window=32, query_pre_attn_scalar=32, max_position_embeddings=32768,
                           pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id, eos_token_id=[tok.eos_token_id,
                                                                                                       tok.convert_tokens_to_ids("<end_of_turn>")])
    torch.manual_seed(seed)
    return Gemma3ForCausalLM(cfg).eval()


def lm(chunk=64, seed=0):
    """model.LM around the tiny model (float32 weights on the CPU)."""
    from kind_goodbye.model import LM
    tok = tokenizer()
    return LM.from_parts(model(tok, seed), tok, mode="bf16", chunk=chunk, dev="cpu")


if __name__ == "__main__":
    t = tokenizer()
    print(json.dumps(t.apply_chat_template([{"role": "user", "content": "hi yes"}, {"role": "assistant", "content": "no"}], tokenize=False)))
