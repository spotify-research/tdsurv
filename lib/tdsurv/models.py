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

from functools import partial

from .base import Model, SeparableModel


class Linear(SeparableModel):

    """Separable model with linear conditional log-odds."""

    def __init__(self, horizon, n_feats):
        params = jnp.zeros(n_feats * horizon)
        super().__init__(horizon, params)

    def get_params_at(self, k):
        idx = jnp.arange(k, len(self.params), step=self.horizon)
        return self.params[idx]

    def set_params_at(self, k, val):
        idx = jnp.arange(k, len(self.params), step=self.horizon)
        self.params = self.params.at[idx].set(val)

    @partial(jax.jit, static_argnums=(0,))
    def hazard_logits(self, params, xs):
        params = params.reshape(-1, self.horizon)
        return jnp.dot(xs, params)

    @partial(jax.jit, static_argnums=(0,))
    def hazard_logit(self, params_at_step, xs):
        return jnp.dot(xs, params_at_step)


class Tabular(SeparableModel):

    """Tabular & nonparametric model."""

    def __init__(self, horizon, n_states):
        params = jnp.zeros(n_states * horizon)
        super().__init__(horizon, params)

    def get_params_at(self, k):
        idx = jnp.arange(k, len(self.params), step=self.horizon)
        return self.params[idx]

    def set_params_at(self, k, val):
        idx = jnp.arange(k, len(self.params), step=self.horizon)
        self.params = self.params.at[idx].set(val)

    @partial(jax.jit, static_argnums=(0,))
    def hazard_logits(self, params, xs):
        params = params.reshape(-1, self.horizon)
        return params[xs, :]

    @partial(jax.jit, static_argnums=(0,))
    def hazard_logit(self, params_at_step, xs):
        return params_at_step[xs]


class CoxPH(Model):

    """Proportional-hazards model with nonparametric baseline."""

    def __init__(self, horizon, n_feats):
        params = jnp.zeros(n_feats + horizon)
        super().__init__(horizon, params)

    @partial(jax.jit, static_argnums=(0,))
    def hazard_logits(self, params, xs):
        return (
            jnp.expand_dims(jnp.dot(xs, params[: -self.horizon]), axis=1)
            + params[-self.horizon :]
        )

    def score(self, xs):
        return -jnp.dot(xs, self.params[: -self.horizon])


class BetaGeom(Model):

    """Beta-geometric model with log-linear parametrization."""

    def __init__(self, horizon, mask_a, mask_b, offset_idx=None):
        assert len(mask_a) == len(mask_b)
        self.mask_a = mask_a
        self.mask_b = mask_b
        self.offset_idx = offset_idx
        self.cutoff = jnp.count_nonzero(mask_a)
        params = jnp.zeros(jnp.count_nonzero(mask_a) + jnp.count_nonzero(mask_b))
        super().__init__(horizon, params)

    @partial(jax.jit, static_argnums=(0,))
    def hazard_logits(self, params, xs):
        if self.offset_idx is not None:
            offset = xs[:, self.offset_idx, None]
        else:
            offset = 0
        return jnp.expand_dims(
            jnp.dot(xs[..., self.mask_a], params[: self.cutoff]),
            axis=1,
        ) - jnp.logaddexp(
            jnp.expand_dims(
                jnp.dot(xs[..., self.mask_b], params[self.cutoff :]),
                axis=1,
            ),
            jnp.log(offset + jnp.arange(self.horizon)),
        )
