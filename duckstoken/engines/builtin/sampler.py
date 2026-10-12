"""Token sampling for the built-in engine, in the order llama.cpp applies it: penalties, top-k, top-p, min-p, temperature."""

from __future__ import annotations

import math

try:
    import numpy as np
except ImportError:  # the engine package reports itself unavailable instead
    np = None


def _finite(name: str, value, low: float | None = None) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number") from None
    if not math.isfinite(v) or (low is not None and v < low):
        raise ValueError(f"{name} is out of range: {value!r}")
    return v


class Sampler:
    """Picks the next token from logits. Deterministic for a given `seed`; `temperature <= 0` means greedy (argmax).

    top_k <= 0 and top_p >= 1 and min_p <= 0 switch those filters off; `repeat_last_n` is how many recent tokens the penalties
    look at (0 = none, -1 = the whole history). The penalties follow llama.cpp: a repeated token's logit is divided by
    `repeat_penalty` (multiplied if negative), then lowered by `frequency_penalty` per occurrence and `presence_penalty` once.
    """

    def __init__(self, temperature=0.8, top_k=40, top_p=0.95, min_p=0.0, repeat_penalty=1.0, presence_penalty=0.0,
                 frequency_penalty=0.0, seed=None, repeat_last_n=64):
        self.temperature = _finite("temperature", temperature)
        self.top_k = int(_finite("top_k", top_k))
        self.top_p = _finite("top_p", top_p)
        self.min_p = _finite("min_p", min_p)
        self.repeat_penalty = _finite("repeat_penalty", repeat_penalty)
        if self.repeat_penalty <= 0:
            raise ValueError("repeat_penalty must be positive")
        self.presence_penalty = _finite("presence_penalty", presence_penalty)
        self.frequency_penalty = _finite("frequency_penalty", frequency_penalty)
        self.repeat_last_n = int(_finite("repeat_last_n", repeat_last_n))
        self.seed = seed if isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0 else None
        self._rng = np.random.default_rng(self.seed)

    def sample(self, logits, history: list[int]) -> int:
        logits = np.array(logits, dtype=np.float32)  # a private copy: the caller's logits are never modified
        if logits.ndim != 1 or logits.size == 0:
            raise ValueError("logits must be a non-empty 1-D array")
        if np.isnan(logits).any() or logits.max() == -np.inf:
            raise ValueError("the logits contain NaN or no usable token (the model computation overflowed)")
        self._penalize(logits, history)
        if self.temperature <= 0:
            return int(np.argmax(logits))
        order = self._candidates(logits)
        scores = logits[order].astype(np.float64)
        probs = self._softmax(scores)
        if 0 < self.top_p < 1:
            keep = int(np.searchsorted(np.cumsum(probs), self.top_p)) + 1
            order, scores, probs = order[:keep], scores[:keep], probs[:keep]
        if self.min_p > 0:
            keep = int(np.count_nonzero(probs >= self.min_p * probs[0]))  # sorted descending, so these are a prefix
            order, scores = order[:keep], scores[:keep]
        cumulative = np.cumsum(self._softmax(scores / self.temperature))
        pick = int(np.searchsorted(cumulative, self._rng.random() * cumulative[-1], side="right"))
        return int(order[min(pick, len(order) - 1)])

    def _penalize(self, logits, history) -> None:
        n = self.repeat_last_n
        active = self.repeat_penalty != 1.0 or self.presence_penalty or self.frequency_penalty
        if not active or n == 0 or not len(history):
            return
        window = np.asarray(history if n < 0 else history[-n:], dtype=np.int64)
        window = window[(window >= 0) & (window < logits.size)]
        ids, counts = np.unique(window, return_counts=True)
        if not ids.size:
            return
        values = logits[ids]
        if self.repeat_penalty != 1.0:
            values = np.where(values <= 0, values * self.repeat_penalty, values / self.repeat_penalty)
        logits[ids] = values - counts * self.frequency_penalty - self.presence_penalty

    def _candidates(self, logits) -> "np.ndarray":
        """Token ids in descending logit order, cut to the top_k best."""
        if 0 < self.top_k < logits.size:
            best = np.argpartition(logits, -self.top_k)[-self.top_k:]
            return best[np.argsort(-logits[best], kind="stable")]
        return np.argsort(-logits, kind="stable")

    @staticmethod
    def _softmax(scores):
        e = np.exp(scores - scores.max())
        return e / e.sum()
