import jittor as jt

try:
    from . import altcorr_cuda
except Exception:
    altcorr_cuda = None

_BACKEND = "AUTO"
_SUPPORTED_BACKENDS = ("AUTO", "CUDA_INLINE", "JITTOR")


def _cuda_inline_available():
    return bool(getattr(jt.flags, "use_cuda", 0)) and altcorr_cuda is not None


def set_backend(name):
    """Configure the active altcorr backend."""
    global _BACKEND
    norm = str(name).upper()
    if norm not in _SUPPORTED_BACKENDS:
        raise RuntimeError(f"Unknown or unsupported altcorr backend: {name!r}")
    if norm == "CUDA_INLINE" and altcorr_cuda is None:
        raise RuntimeError("CUDA_INLINE backend requested, but altcorr_cuda is unavailable")
    _BACKEND = norm


def get_backend():
    """Return the resolved active altcorr backend name."""
    if _BACKEND == "AUTO":
        return "CUDA_INLINE" if _cuda_inline_available() else "JITTOR"
    return _BACKEND


def _resolve_backend(backend=None):
    norm = _BACKEND if backend is None else str(backend).upper()
    if norm not in _SUPPORTED_BACKENDS:
        raise RuntimeError(f"Unknown or unsupported altcorr backend: {(backend or _BACKEND)!r}")
    if norm == "AUTO":
        return "CUDA_INLINE" if _cuda_inline_available() else "JITTOR"
    if norm == "CUDA_INLINE" and not _cuda_inline_available():
        raise RuntimeError("CUDA_INLINE backend requested, but CUDA is disabled or altcorr_cuda is unavailable")
    return norm


def _patchify_pure_jittor(net, coords, radius):
    B, C, H, W = net.shape
    M = coords.shape[1]
    D = 2 * radius + 2
    if M == 0:
        return jt.zeros((B, 0, C, D, D), dtype=net.dtype)

    coords_f = coords.float32()
    x0 = jt.floor(coords_f[:, :, 0]).int32()
    y0 = jt.floor(coords_f[:, :, 1]).int32()

    y_expr = f"@e1(i0,i1)+i3-{radius}"
    x_expr = f"@e0(i0,i1)+i4-{radius}"
    return net.reindex(
        [B, M, C, D, D],
        ["i0", "i2", y_expr, x_expr],
        overflow_conditions=[
            f"{y_expr}<0",
            f"{y_expr}>={H}",
            f"{x_expr}<0",
            f"{x_expr}>={W}",
        ],
        overflow_value=0,
        extras=[x0, y0],
    )


def patchify(net, coords, radius, mode="bilinear", backend=None):
    """Extract patches at floating-point coordinates using floor sampling and optional bilinear interpolation."""
    resolved = _resolve_backend(backend)
    B, C, _, _ = net.shape
    M = coords.shape[1]

    if M == 0:
        out_d = 2 * radius + 1 if mode == "bilinear" else 2 * radius + 2
        return jt.zeros((B, 0, C, out_d, out_d), dtype=net.dtype)

    if resolved == "CUDA_INLINE":
        patches = altcorr_cuda.patchify_raw_cuda(net, coords, radius)
    else:
        patches = _patchify_pure_jittor(net, coords, radius)

    if mode == "bilinear":
        coords_f = coords.float32()
        offset = coords_f - jt.floor(coords_f)
        dx = offset[:, :, 0].reshape(B, M, 1, 1, 1)
        dy = offset[:, :, 1].reshape(B, M, 1, 1, 1)

        d = 2 * radius + 1
        x00 = (1.0 - dy) * (1.0 - dx) * patches[:, :, :, :d, :d]
        x01 = (1.0 - dy) * dx * patches[:, :, :, :d, 1:]
        x10 = dy * (1.0 - dx) * patches[:, :, :, 1:, :d]
        x11 = dy * dx * patches[:, :, :, 1:, 1:]
        return x00 + x01 + x10 + x11

    return patches


def _corr_pure_jittor(fmap1, fmap2, coords, ii, jj, radius=1, dropout=1.0):
    """Compute local correlation volumes using pure Jittor reindex operations."""
    del dropout  # Unused during inference forward pass.

    B, M, _, Hp, Wp = coords.shape
    d = 2 * radius + 1
    if M == 0:
        return jt.zeros((B, 0, d, d, Hp, Wp), dtype=fmap1.dtype)

    C = fmap1.shape[2]
    H2, W2 = fmap2.shape[3], fmap2.shape[4]
    D = 2 * radius + 2
    out_dtype = fmap1.dtype

    ii_i = ii.reshape(-1).int32()
    jj_i = jj.reshape(-1).int32()

    elem_per_edge = max(1, B * C * D * D * Hp * Wp)
    chunk_size = max(1, min(256, (32 * 1024 * 1024) // elem_per_edge))

    y_expr = f"@e2(i0,i1,i5,i6)+i3-{radius}"
    x_expr = f"@e1(i0,i1,i5,i6)+i4-{radius}"
    overflow_conds = [
        f"{y_expr}<0",
        f"{y_expr}>={H2}",
        f"{x_expr}<0",
        f"{x_expr}>={W2}",
    ]

    out_chunks = []
    for start in range(0, M, chunk_size):
        end = min(M, start + chunk_size)
        Mc = end - start
        ii_c = ii_i[start:end]
        jj_c = jj_i[start:end]
        coords_c = coords[:, start:end].float32()

        x_c = coords_c[:, :, 0]
        y_c = coords_c[:, :, 1]
        x0 = jt.floor(x_c).int32()
        y0 = jt.floor(y_c).int32()

        f1 = fmap1.reindex(
            [B, Mc, C, Hp, Wp],
            ["i0", "@e0(i1)", "i2", "i3", "i4"],
            extras=[ii_c],
        ).reshape(B, Mc, C, 1, 1, Hp, Wp)

        f2 = fmap2.reindex(
            [B, Mc, C, D, D, Hp, Wp],
            ["i0", "@e0(i1)", "i2", y_expr, x_expr],
            overflow_conditions=overflow_conds,
            overflow_value=0,
            extras=[jj_c, x0, y0],
        )

        corr_c = (f1.float32() * f2.float32()).sum(dim=2).cast(out_dtype)

        dx = (x_c - jt.floor(x_c)).cast(out_dtype).reshape(B, Mc, 1, 1, Hp, Wp)
        dy = (y_c - jt.floor(y_c)).cast(out_dtype).reshape(B, Mc, 1, 1, Hp, Wp)
        one = jt.ones((), dtype=out_dtype)

        out_c = (
            (one - dx) * (one - dy) * corr_c[:, :, :d, :d]
            + dx * (one - dy) * corr_c[:, :, :d, 1:]
            + (one - dx) * dy * corr_c[:, :, 1:, :d]
            + dx * dy * corr_c[:, :, 1:, 1:]
        )
        out_chunks.append(out_c.cast(out_dtype).permute(0, 1, 3, 2, 4, 5))

    if len(out_chunks) == 1:
        return out_chunks[0]
    return jt.concat(out_chunks, dim=1)


def corr(fmap1, fmap2, coords, ii, jj, radius=1, dropout=1.0, backend=None):
    """Compute local correlation volumes between fmap1 patches and fmap2 pyramids."""
    resolved = _resolve_backend(backend)
    if resolved == "CUDA_INLINE":
        return altcorr_cuda.corr_cuda(
            fmap1, fmap2, coords, ii, jj, radius=radius, dropout=dropout
        )
    return _corr_pure_jittor(
        fmap1, fmap2, coords, ii, jj, radius=radius, dropout=dropout
    )
