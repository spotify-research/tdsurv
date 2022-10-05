# Copyright 2022 Spotify AB
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
import numpy as np

from jax.scipy.special import entr
from lifelines.utils import concordance_index as _concordance_index


@jax.jit
def bce_logits(targets, logits):
    """Compute the binary cross-entropy with logits.

    Letting `p = targets` and `q = sigmoid(logits)`, this function returns the
    binary cross-entropy `H(p, q) = -p * log(q) - (1 - p) * log(1 - q)`.
    """
    return -targets * logits + jax.nn.softplus(logits)


@jax.jit
def cce_logits(targets, logits):
    """Compute the categorical cross-entropy with logits.

    Letting `p = targets` and `q = softmax(logits)`, this function returns the
    categorical cross-entropy `H(p, q) = -sum(p * log(q))`.
    """
    return -jnp.sum(targets * logits, axis=-1) + jax.nn.logsumexp(logits, axis=-1)


@jax.jit
def kldiv_logits(targets, logits):
    """Compute the KL-divergence with logits.

    Letting `p = targets` and `q = softmax(logits)`, this function returns the
    KL-divergence `H(p, q) = sum(p * log(p / q))`.
    """
    return cce_logits(targets, logits) - jnp.sum(entr(targets), axis=-1)


def kaplan_meier(ts, cs):
    """Kaplan-Meier estimator of survival curve."""
    cs = cs.astype(bool)
    steps = np.arange(0, np.max(ts) + 1)
    # Number of individuals known to have survived up to step k = 0, 1, ...
    ns = np.sum(ts[:, np.newaxis] >= steps, axis=0)
    # Number of events that happened at step k = 0, 1, ...
    ds = np.sum(ts[~cs, np.newaxis] == steps, axis=0)
    # Product over k of (1 - empirical hazard at k).
    return np.cumprod(1 - ds / ns)


def unroll(seqs, ts, cs, compress=False):
    """Unroll sequences.

    This function transforms each sequence `(x1, x2, ..., xt)` into
    subsequences `((x1, x2, ..., xt), (x2, ..., xt), ..., (xt,))`.

    Note: the smallest subsequence always contains two observed states, whether
    implicitly or explicitly.

    - For uncensored sequences, the smallest subsequence is `(xt,)` and
      implicitly accounts for the terminal state that follows.
    - For censored sequences, the smallest subsequence is `(x{t-1}, xt)`.
    """
    cs = cs.astype(bool)
    seqs_ = np.copy(seqs)
    ts_ = np.copy(ts)
    cs_ = np.copy(cs)
    for i in range(1, np.max(ts)):
        idx = ts > i  # Indices of seqs whose successor state is observed.
        new = np.zeros((np.sum(idx),) + seqs.shape[1:], dtype=seqs.dtype)
        new[:, :-i] = seqs[idx, i:]
        seqs_ = np.concatenate((seqs_, new))
        ts_ = np.concatenate((ts_, ts[idx] - i))
        cs_ = np.concatenate((cs_, cs[idx]))
    if compress:
        return (seqs_[:,:2], ts_, cs_)
    return (seqs_, ts_, cs_)


def concordance_index(scores, ts, cs):
    """Compute concordance-index for given scores."""
    # Thin wrapper around the `lifelines` implementation.
    cs = cs.astype(bool)
    return _concordance_index(ts + cs, scores, ~cs)
