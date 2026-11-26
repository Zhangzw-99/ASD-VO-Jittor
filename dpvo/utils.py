from contextlib import ContextDecorator
import time

import jittor as jt
import jittor.nn as nn


all_times = []


class Timer(ContextDecorator):
    def __init__(self, name, enabled=True):
        self.name = name
        self.enabled = enabled
        self._start = 0.0

    def __enter__(self):
        if self.enabled:
            jt.sync_all(True)
            self._start = time.perf_counter()

    def __exit__(self, type, value, traceback):
        global all_times
        if self.enabled:
            jt.sync_all(True)
            elapsed = (time.perf_counter() - self._start) * 1000.0
            all_times.append(elapsed)
            print(f"{self.name} {elapsed:.03f}")


def coords_grid(b, n, h, w, **kwargs):
    """Coordinate grid of shape (b, n, 2, h, w)."""
    dtype = kwargs.get("dtype", "float32")
    if dtype in (float, "float"):
        dtype = "float32"
    x = jt.arange(0, w).cast(dtype)
    y = jt.arange(0, h).cast(dtype)
    yy, xx = jt.meshgrid(y, x)
    coords = jt.stack([xx, yy], dim=0)
    return coords.reshape(1, 1, 2, h, w).repeat(b, n, 1, 1, 1)


def coords_grid_with_index(d, **kwargs):
    """Coordinate grid with frame index."""
    b, n, h, w = d.shape
    dtype = kwargs.get("dtype", "float32")
    if dtype in (float, "float"):
        dtype = "float32"

    x = jt.arange(0, w).cast(dtype)
    y = jt.arange(0, h).cast(dtype)

    yy, xx = jt.meshgrid(y, x)
    yy = yy.reshape(1, 1, h, w).repeat(b, n, 1, 1)
    xx = xx.reshape(1, 1, h, w).repeat(b, n, 1, 1)

    coords = jt.stack([xx, yy, d.cast(dtype)], dim=2)
    index = jt.arange(0, n).cast(dtype)
    index = index.reshape(1, n, 1, 1, 1).repeat(b, 1, 1, h, w)

    return coords, index


def patchify(x, patch_size=3):
    """Extract patches from video."""
    b, n, c, h, w = x.shape
    x = x.reshape(b * n, c, h, w)
    y = nn.unfold(x, patch_size)
    y = y.transpose(1, 2)
    return y.reshape(b, -1, c, patch_size, patch_size)


def pyramidify(fmap, lvls=[1]):
    """Turn fmap into a pyramid."""
    b, n, c, h, w = fmap.shape

    pyramid = []
    for lvl in lvls:
        gmap = nn.avg_pool2d(fmap.reshape(b * n, c, h, w), lvl, stride=lvl)
        pyramid += [gmap.reshape(b, n, c, h // lvl, w // lvl)]

    return pyramid


def all_pairs_exclusive(n, **kwargs):
    dtype = kwargs.get("dtype", "int64")
    idx_1d = jt.arange(n).cast(dtype)
    ii, jj = jt.meshgrid(idx_1d, idx_1d)
    ii_flat = ii.reshape(-1)
    jj_flat = jj.reshape(-1)
    k = jt.where(ii_flat != jj_flat)[0]
    return ii_flat[k], jj_flat[k]


def set_depth(patches, depth):
    patches[..., 2, :, :] = depth[..., None, None]
    return patches


def flatmeshgrid(*args, **kwargs):
    indexing = kwargs.pop("indexing", None)
    if indexing is not None and indexing != "ij":
        raise NotImplementedError(f"Unsupported meshgrid indexing: {indexing}")
    grid = jt.meshgrid(*args)
    return (x.reshape(-1) for x in grid)


def quantile_05(x):
    """Median (q=0.5) with linear interpolation matching numpy.quantile(x, 0.5)."""
    flat = x.reshape(-1).float32()
    n = flat.shape[0]
    if n == 0:
        return jt.array(0.0, dtype="float32")
    if n == 1:
        return flat[0]
    _, sorted_vals = jt.argsort(flat, dim=0)
    lo = (n - 1) // 2
    hi = n // 2
    if lo == hi:
        return sorted_vals[lo]
    return 0.5 * (sorted_vals[lo] + sorted_vals[hi])
