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

import abc
import jax
import jax.numpy as jnp
import numpy as np

from functools import partial
from scipy.linalg import solve
from scipy.optimize import minimize

from .utils import bce_logits, kaplan_meier


# Enable double-precision floating-point numbers.
jax.config.update("jax_enable_x64", True)


class Model(metaclass=abc.ABCMeta):
    def __init__(self, horizon, params):
        self.horizon = horizon
        self.params = params

    @abc.abstractmethod
    def hazard_logits(self, params, xs):
        """Compute logits of hazard functions.

        Letting `h(k | x)` be the hazard at step `k` from state `x`, this
        function returns

            [ logit(h(1 | x))  ...  logit(h(K | x)) ]

        where `K` is the horizon.
        """

    def survival_curve(self, xs):
        """Compute the fixed-horizon survival CCDF, a.k.a. survival curve.

        Letting `S(k | x)` be the survival at step `k` from state `x`, this
        function returns

            [ 1  S(1 | x)  ...  S(K | x) ]

        where `K` is the horizon.
        """
        logits = self.hazard_logits(self.params, xs)
        log_hs = jax.nn.log_sigmoid(logits)
        surv = jnp.exp(jnp.cumsum(log_hs - logits, axis=1))
        return jnp.insert(surv, 0, 1.0, axis=1)

    def survival_dist(self, xs):
        """Compute the fixed-horizon survival distribution.

        Letting `f(k | x)` and `S(k | x)` be the probability mass and survival
        (respectively) at step `k` from state `x`, this function returns

            [ f(1 | x)  ...  f(K | x)  S(K | x) ]

        where `K` is the horizon.
        """
        return -jnp.diff(self.survival_curve(xs), append=0.0)

    @partial(jax.jit, static_argnums=(0,))
    def obj(self, params, xs, ys, ws, l2):
        """Weighted binary cross-entropy relative to `ys`."""
        return jnp.sum(
            ws * bce_logits(ys, self.hazard_logits(params, xs))
        ) + l2 * jnp.sum(jnp.square(params))

    grad = jax.jit(jax.grad(obj, argnums=1), static_argnums=(0,))
    hess = jax.jit(jax.hessian(obj, argnums=1), static_argnums=(0,))

    @partial(jax.jit, static_argnums=(0,))
    def hvp(self, params, vec, *args):
        """Efficiently compute a Hessian-vector product."""
        # Forward-over-reverse, inspired from the Autodiff Cookbook.
        grad = lambda params: self.grad(params, *args)
        return jax.jvp(grad, (params,), (vec,))[1]

    def _inner_loop(self, xs, ys, ws, tol=1e-4, l2=0.0, verbose=False):
        """Minimize weighted binary cross-entropy relative to `ys`."""
        res = minimize(
            fun=self.obj,
            x0=self.params,
            args=(xs, ys, ws, l2),
            method="Newton-CG",
            jac=self.grad,
            # hess=self.hess,
            hessp=self.hvp,
            options={"disp": verbose},
            tol=tol,
        )
        return res.x

    def fit(
        self,
        seqs,
        ts,
        cs,
        *,
        lambda_,
        n_iters=5,
        tol=1e-4,
        l2=0.0,
        verbose=False,
    ):
        """Fit the model.

        This method iteratively estimates the parameters of the model from
        samples by minimizing the cross-entropy relative to TD pseudo-targets.
        `N` is the number of sequences, `K` the maximum length of a sequence
        and `D` the dimensionality of the state space.

        Parameters
        ----------
        seqs : ndarray, shape (N, K, ...)
            Sequences of states.
        ts : ndarray(int), shape (N,)
            Length of each sequence.
        cs : ndarray(bool), shape (N,)
            Right-censoring indicator for each sequence.
        lambda_ : float
            Trace decay parameter, between 0 and 1. If equal to one, inference
            is identical to standard MLE.
        n_iters : int, optional
            Number of outer-loop iterations, only relevant if `lambda_ < 1`.
            Default: 5.
        tol : float, optional
            Threshold to determine convergence of inner-loop optimization.
        l2 : float, optional
            Regularization strength. Default: 0.0.
        verbose : bool, optional
            If true, print diagnostics for inner-loop optimization calls.
        """
        cs = cs.astype(bool)
        n = len(seqs)
        if lambda_ == 1.0:
            n_iters = 1
        elif lambda_ < 0.0 or lambda_ > 1.0:
            raise ValueError("lambda_ outside of range [0, 1]")
        # Exponentially decreasing multipliers.
        multipliers = lambda_ ** np.arange(self.horizon)
        multipliers[:-1] *= 1 - lambda_
        # Outer loop.
        for _ in range(n_iters):
            xs = seqs[:, 0]
            ys = np.zeros((len(seqs), self.horizon))
            ws = np.zeros((len(seqs), self.horizon))
            # Compute backup targets and weights at all steps.
            for m, mult in enumerate(multipliers, start=1):
                if mult == 0.0:
                    # Multiplier is zero, we can ignore this step. This leads to
                    # a significant speedup for `lambda_ = 0` and `lambda_ = 1`
                    continue
                ys_m, ws_m = self._targets(m, seqs, ts, cs)
                ys += mult * ys_m
                ws += mult * ws_m
            # Inner loop: find parameters given pseudo-targets.
            self.params = self._inner_loop(xs, ys, ws, tol, l2, verbose)

    def _targets(self, m, seqs, ts, cs):
        """Compute pseudo-targets and weights for the m-step backup."""
        ys = np.zeros((len(seqs), self.horizon))
        ws = np.zeros((len(seqs), self.horizon))
        # Observed outcomes within first m steps.
        idx = (ts <= m) & ~cs  # Seqs that reached terminal state within window.
        ys[idx, ts[idx] - 1] = 1.0
        ws[:, :m] = (np.arange(m) < ts[:, np.newaxis]).astype(float)
        # Predicted outcomes after first m steps.
        if m < self.horizon:
            # Seqs that are still active after the window.
            idx = (ts > m) | ((ts == m) & cs)
            nxt = seqs[idx, m]
            logits = self.hazard_logits(self.params, nxt)
            log_hs = jax.nn.log_sigmoid(logits)
            ys[idx, m:] = jnp.exp(log_hs[:, :-m])
            ws[idx, m] = 1.0
            ws[idx, (m + 1) :] = jnp.exp(
                jnp.cumsum(
                    log_hs[:, : -(m + 1)] - logits[:, : -(m + 1)],
                    axis=1,
                )
            )
        return (ys, ws)

    def loglike(self, xs, ts, cs):
        """Compute the log-likelihood of the model under the data."""
        cs = cs.astype(bool)
        seqs = np.expand_dims(xs, axis=1)
        ys, ws = self._targets(self.horizon, seqs, ts, cs)
        return -self.obj(self.params, xs, ys, ws, l2=0.0)

    def integrated_brier_score(self, xs, ts, cs):
        """Compute the integrated Brier score."""
        cs = cs.astype(bool)
        surv = self.survival_curve(xs)
        ws = kaplan_meier(ts - ~cs, ~cs)
        t_max = np.max(ts)
        tot = 0.0
        for h in range(1, t_max + 1):
            # Sequences that terminated.
            idx = (ts <= h) & ~cs
            tot += np.sum((1 / ws[ts[idx] - 1]) * (0.0 - surv[idx, h - 1]) ** 2)
            # Sequences that are still active.
            idx = (ts > h) | ((ts == h) & cs)
            tot += np.sum((1 / ws[h - 1]) * (1.0 - surv[idx, h - 1]) ** 2)
        return tot / (t_max * len(ts))

    def params_with_stderr(self, seqs, ts, cs, cutoff=None):
        """Return parameters with approximate standard error."""
        # Uses a Laplace approximation.
        if cutoff is None:
            cutoff = len(self.params)
        ys, ws = self._targets(self.horizon, seqs, ts, cs)
        hess = self.hess(self.params, seqs[:, 0], ys, ws, l2=0.0)
        inv = solve(hess[:cutoff, :cutoff], np.eye(cutoff), assume_a="pos")
        return (self.params[:cutoff], np.sqrt(np.diag(inv)))


class SeparableModel(Model):

    """
    This subclass provides specialized inference functions that take advantage
    of the fact that the parameters of the hazard function separate across
    time.
    """

    @abc.abstractmethod
    def hazard_logit(self, params_at_step, xs):
        """Compute logit of hazard function at given step."""

    @abc.abstractmethod
    def get_params_at(self, k):
        """Get parameters for hazard at step `k` (zero-indexed)."""

    @abc.abstractmethod
    def set_params_at(self, s, val):
        """Set parameters for hazard at step `k` (zero-indexed)."""

    @partial(jax.jit, static_argnums=(0,))
    def obj_step(self, params, xs, ys, ws, l2):
        return jnp.sum(
            ws * bce_logits(ys, self.hazard_logit(params, xs))
        ) + l2 * jnp.sum(jnp.square(params))

    grad_step = jax.jit(jax.grad(obj_step, argnums=1), static_argnums=(0,))

    @partial(jax.jit, static_argnums=(0,))
    def hvp_step(self, params, vec, *args):
        """Efficiently compute a Hessian-vector product."""
        # Forward-over-reverse, inspired from the Autodiff Cookbook.
        grad = lambda params: self.grad_step(params, *args)
        return jax.jvp(grad, (params,), (vec,))[1]

    def _fit_step(self, xs, ys, ws, k, tol=1e-4, l2=0.0, verbose=False):
        res = minimize(
            fun=self.obj_step,
            x0=self.get_params_at(k),
            args=(xs, ys, ws, l2),
            method="Newton-CG",
            jac=self.grad_step,
            hessp=self.hvp_step,
            options={"disp": verbose},
            tol=tol,
        )
        return res.x

    def fit(
        self,
        seqs,
        ts,
        cs,
        *,
        lambda_,
        tol=1e-4,
        l2=0.0,
        verbose=False,
    ):
        """Fit the model."""
        cs = cs.astype(bool)
        n = len(seqs)
        if lambda_ < 0.0 or lambda_ > 1.0:
            raise ValueError("lambda_ outside of range [0, 1]")
        # Exponentially decreasing multipliers.
        multipliers = lambda_ ** np.arange(self.horizon)
        multipliers[:-1] *= 1 - lambda_
        # Outer loop.
        for k in range(self.horizon):
            xs = seqs[:, 0]
            ys = np.zeros(len(seqs))
            ws = np.zeros(len(seqs))
            # Compute backup targets and weights at all steps.
            for m, mult in enumerate(multipliers, start=1):
                if mult == 0.0:
                    # Multiplier is zero, we can ignore this step. This leads to
                    # a significant speedup for `lambda_ = 0` and `lambda_ = 1`
                    continue
                ys_m, ws_m = self._targets_step(m, k, seqs, ts, cs)
                ys += mult * ys_m
                ws += mult * ws_m
            # Inner loop: find parameters given pseudo-targets.
            res = self._fit_step(xs, ys, ws, k, tol, l2, verbose)
            self.set_params_at(k, res)

    def _targets_step(self, m, k, seqs, ts, cs):
        """Compute pseudo-targets and weights at step k for the m-step backup."""
        ys = np.zeros(len(seqs))
        ws = np.zeros(len(seqs))
        if k < m:
            # Observed outcome.
            ys = np.array(~cs & (ts - 1 == k), dtype=float)
            ws = np.array(ts > k, dtype=float)
        else:
            # Predicted outcome.
            # Seqs that are still active after the window.
            idx = (ts > m) | ((ts == m) & cs)
            nxt = seqs[idx, m]
            logits = self.hazard_logits(self.params, nxt)
            log_hs = jax.nn.log_sigmoid(logits)
            ys[idx] = jnp.exp(log_hs[:, k - m])
            ws[idx] = jnp.exp(jnp.sum(log_hs[:, : k - m] - logits[:, : k - m], axis=1))
        return (ys, ws)
