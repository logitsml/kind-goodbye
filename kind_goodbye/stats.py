"""Bootstrap helpers shared by the analyses: resampling indices and the Hodges-Lehmann estimate."""
import numpy as np

B_CHECK = 2000


def boot_idx(n, B=B_CHECK, seed=0):
    return np.random.default_rng(seed).integers(0, n, (B, n))


def hl(x):
    x = np.asarray(x, float)
    i, j = np.triu_indices(len(x))
    return float(np.median((x[i] + x[j]) / 2))


def hl_rows(M):
    i, j = np.triu_indices(M.shape[1])
    return np.median((M[:, i] + M[:, j]) / 2, axis=1)

