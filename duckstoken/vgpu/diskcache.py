"""Two bounded caches.

`BoundedCache`   an in-memory LRU with an exact byte budget (and optional item limit), hit/miss/eviction counters and
                 thrash detection.  It is a building block; it does not touch the disk.

`StagingCache`   the optional disk tier: copy a model from a slow drive (HDD) onto a fast one (SSD/NVMe) so that
                 llama.cpp's mmap page faults are served by the fast drive.  It never starts on its own (the caller
                 must pass ``confirm=True``), copies through ``<file>.part`` + atomic rename, removes partial files
                 when it fails or is cancelled, and only ever deletes files it created itself: everything it owns is
                 listed in ``manifest.json`` and deletion is limited to those paths inside its own directory.
                 Originals are never modified or removed.

Honest limits: staging does not make an HDD fast, it makes a *copy* on an SSD that is fast.  The copy takes as long as
reading the original once, needs free space for a full second copy of the model, and the staged file must be kept in
sync by re-staging when the original changes (`lookup` refuses a copy whose original changed).
"""

from __future__ import annotations

import collections
import threading
import time
from typing import Callable

from .staging import GIB, MIB, StagingCache  # noqa: F401 - the staging tier lives in staging.py; import it from here


# ---------------------------------------------------------------------------------------------------- BoundedCache
class BoundedCache:
    """Thread-safe LRU keyed by str with a hard byte budget.

    ``put`` returns False (and changes nothing) when the item alone is larger than the budget.  Sizes default to the
    length of a ``bytes`` value or the UTF-8 length of a ``str``; pass ``size`` for anything else.
    """

    def __init__(self, max_bytes: int, max_items: int | None = None, clock: Callable[[], float] = time.monotonic):
        if max_bytes < 0:
            raise ValueError("max_bytes must be >= 0")
        self.max_bytes = int(max_bytes)
        self.max_items = None if max_items is None else max(0, int(max_items))
        self._clock = clock
        self._lock = threading.RLock()
        self._items: collections.OrderedDict[str, tuple[object, int]] = collections.OrderedDict()
        self._bytes = 0
        self.hits = self.misses = self.evictions = 0
        self._events: collections.deque[tuple[float, str]] = collections.deque(maxlen=4096)  # (time, "hit"|"evict")

    @staticmethod
    def _size_of(value, size: int | None) -> int:
        if size is not None:
            return max(0, int(size))
        if isinstance(value, (bytes, bytearray, memoryview)):
            return len(value)
        if isinstance(value, str):
            return len(value.encode("utf-8"))
        raise TypeError("size is required for values that are not bytes or str")

    def put(self, key: str, value, size: int | None = None) -> bool:
        n = self._size_of(value, size)
        with self._lock:
            if n > self.max_bytes or self.max_items == 0:
                return False
            old = self._items.pop(key, None)
            if old is not None:
                self._bytes -= old[1]
            while self._items and (self._bytes + n > self.max_bytes
                                   or (self.max_items is not None and len(self._items) + 1 > self.max_items)):
                _, (_, freed) = self._items.popitem(last=False)
                self._bytes -= freed
                self.evictions += 1
                self._events.append((self._clock(), "evict"))
            self._items[key] = (value, n)
            self._bytes += n
            return True

    def get(self, key: str, default=None):
        with self._lock:
            hit = self._items.get(key)
            if hit is None:
                self.misses += 1
                return default
            self._items.move_to_end(key)
            self.hits += 1
            self._events.append((self._clock(), "hit"))
            return hit[0]

    def pop(self, key: str, default=None):
        with self._lock:
            hit = self._items.pop(key, None)
            if hit is None:
                return default
            self._bytes -= hit[1]
            return hit[0]

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0

    def __contains__(self, key) -> bool:
        with self._lock:
            return key in self._items

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def keys(self) -> list[str]:
        """Oldest (next to be evicted) first."""
        with self._lock:
            return list(self._items)

    def thrashing(self, window_s: float = 60.0) -> bool:
        """True when, within the last `window_s`, more items were evicted than there were hits: the working set does
        not fit and the cache is churning instead of helping."""
        with self._lock:
            cutoff = self._clock() - window_s
            hits = evicts = 0
            for t, kind in reversed(self._events):
                if t < cutoff:
                    break
                if kind == "hit":
                    hits += 1
                else:
                    evicts += 1
            return evicts > hits

    def stats(self) -> dict:
        with self._lock:
            lookups = self.hits + self.misses
            return {"hits": self.hits, "misses": self.misses, "evictions": self.evictions, "bytes": self._bytes,
                    "items": len(self._items), "max_bytes": self.max_bytes, "max_items": self.max_items,
                    "hit_rate": (self.hits / lookups) if lookups else None}
