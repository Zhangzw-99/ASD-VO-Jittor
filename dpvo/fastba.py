import jittor as jt
import numpy as np
import scipy.linalg

try:
    from . import fastba_cuda as _fastba_cuda
except Exception:
    _fastba_cuda = None

_BACKEND = "AUTO"
_SUPPORTED_BACKENDS = ("AUTO", "CUDA_INLINE", "JITTOR")


def _cuda_inline_available():
    return bool(jt.flags.use_cuda) and (_fastba_cuda is not None)


def set_backend(name):
    """Configure the active fastba backend."""
    global _BACKEND
    norm = str(name).upper()
    if norm not in _SUPPORTED_BACKENDS:
        raise RuntimeError(f"Unknown or unsupported fastba backend: {name!r}")
    if norm == "CUDA_INLINE" and not _cuda_inline_available():
        raise RuntimeError("CUDA_INLINE fastba backend requires use_cuda=1 and fastba_cuda")
    _BACKEND = norm


def get_backend():
    """Return the active fastba backend name."""
    return _resolve_backend(_BACKEND)


def _resolve_backend(backend=None):
    norm = _BACKEND if backend is None else str(backend).upper()
    if norm not in _SUPPORTED_BACKENDS:
        raise RuntimeError(f"Unknown or unsupported fastba backend: {backend!r}")
    if norm == "AUTO":
        return "CUDA_INLINE" if _cuda_inline_available() else "JITTOR"
    if norm == "CUDA_INLINE" and not _cuda_inline_available():
        raise RuntimeError("CUDA_INLINE fastba backend requires use_cuda=1 and fastba_cuda")
    return norm


def _extract_var(obj):
    return obj if isinstance(obj, jt.Var) else obj.data


def _act_so3(q, X):
    """Apply unit quaternion q (..., 4) [qx,qy,qz,qw] to 3D vectors X (..., 3)."""
    q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x0, x1, x2 = X[..., 0], X[..., 1], X[..., 2]

    uv0 = 2.0 * (q1 * x2 - q2 * x1)
    uv1 = 2.0 * (q2 * x0 - q0 * x2)
    uv2 = 2.0 * (q0 * x1 - q1 * x0)

    y0 = x0 + q3 * uv0 + (q1 * uv2 - q2 * uv1)
    y1 = x1 + q3 * uv1 + (q2 * uv0 - q0 * uv2)
    y2 = x2 + q3 * uv2 + (q0 * uv1 - q1 * uv0)
    return jt.stack([y0, y1, y2], dim=-1)


def _act_se3(t, q, X):
    """Apply SE3 (t, q) to homogeneous 4D points X (..., 4)."""
    xyz = _act_so3(q, X[..., :3]) + X[..., 3:4] * t
    return jt.concat([xyz, X[..., 3:4]], dim=-1)


def _rel_se3(ti, qi, tj, qj):
    """Compute relative transform Tj * Ti^{-1} returning (tij, qij)."""
    qi0, qi1, qi2, qi3 = qi[..., 0], qi[..., 1], qi[..., 2], qi[..., 3]
    qj0, qj1, qj2, qj3 = qj[..., 0], qj[..., 1], qj[..., 2], qj[..., 3]

    qij0 = -qj3 * qi0 + qj0 * qi3 - qj1 * qi2 + qj2 * qi1
    qij1 = -qj3 * qi1 + qj1 * qi3 - qj2 * qi0 + qj0 * qi2
    qij2 = -qj3 * qi2 + qj2 * qi3 - qj0 * qi1 + qj1 * qi0
    qij3 = qj3 * qi3 + qj0 * qi0 + qj1 * qi1 + qj2 * qi2
    qij = jt.stack([qij0, qij1, qij2, qij3], dim=-1)

    tij = tj - _act_so3(qij, ti)
    return tij, qij


def _adj_se3(t, q, X):
    """Apply coadjoint transformation matching ba_cuda.cu adjSE3."""
    qinv = jt.stack([-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]], dim=-1)
    y_lin = _act_so3(qinv, X[..., :3])
    y_ang = _act_so3(qinv, X[..., 3:6])

    u0 = t[..., 2] * X[..., 1] - t[..., 1] * X[..., 2]
    u1 = t[..., 0] * X[..., 2] - t[..., 2] * X[..., 0]
    u2 = t[..., 1] * X[..., 0] - t[..., 0] * X[..., 1]
    u = jt.stack([u0, u1, u2], dim=-1)

    v = _act_so3(qinv, u)
    return jt.concat([y_lin, y_ang + v], dim=-1)


def _exp_so3(phi):
    """SO3 exponential map matching ba_cuda.cu expSO3."""
    theta_sq = (phi * phi).sum(dim=-1)
    theta_p4 = theta_sq * theta_sq
    theta = jt.sqrt(jt.maximum(theta_sq, 1e-30))

    small = theta_sq < 1e-8
    imag_small = 0.5 - (1.0 / 48.0) * theta_sq + (1.0 / 3840.0) * theta_p4
    real_small = 1.0 - (1.0 / 8.0) * theta_sq + (1.0 / 384.0) * theta_p4

    imag_large = jt.sin(0.5 * theta) / theta
    real_large = jt.cos(0.5 * theta)

    imag = jt.where(small, imag_small, imag_large)
    real = jt.where(small, real_small, real_large)

    return jt.stack(
        [imag * phi[..., 0], imag * phi[..., 1], imag * phi[..., 2], real],
        dim=-1,
    )


def _cross(a, b):
    return jt.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        dim=-1,
    )


def _exp_se3(xi):
    """SE3 exponential map matching ba_cuda.cu expSE3."""
    tau = xi[..., :3]
    phi = xi[..., 3:6]
    dq = _exp_so3(phi)

    theta_sq = (phi * phi).sum(dim=-1)
    theta = jt.sqrt(jt.maximum(theta_sq, 1e-30))
    large = theta > 1e-4

    safe_tsq = jt.where(large, theta_sq, jt.ones_like(theta_sq))
    safe_t = jt.where(large, theta, jt.ones_like(theta))
    a = (1.0 - jt.cos(safe_t)) / safe_tsq
    b = (safe_t - jt.sin(safe_t)) / (safe_t * safe_tsq)

    c1 = _cross(phi, tau)
    c2 = _cross(phi, c1)
    dt_large = tau + a[..., None] * c1 + b[..., None] * c2
    dt = jt.where(large[..., None], dt_large, tau)
    return dt, dq


def _retr_se3(xi, t, q):
    """Left-multiplicative SE3 retraction matching ba_cuda.cu retrSE3."""
    dt, dq = _exp_se3(xi)
    dq0, dq1, dq2, dq3 = dq[..., 0], dq[..., 1], dq[..., 2], dq[..., 3]
    q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    q_new = jt.stack(
        [
            dq3 * q0 + dq0 * q3 + dq1 * q2 - dq2 * q1,
            dq3 * q1 + dq1 * q3 + dq2 * q0 - dq0 * q2,
            dq3 * q2 + dq2 * q3 + dq0 * q1 - dq1 * q0,
            dq3 * q3 - dq0 * q0 - dq1 * q1 - dq2 * q2,
        ],
        dim=-1,
    )
    t_new = _act_so3(dq, t) + dt
    return t_new, q_new


def _neighbors_pure(kk, jj):
    """Find previous/next temporal neighbor edge indices grouped by patch kk and sorted by jj."""
    E = int(kk.numel())
    if E == 0:
        return jt.zeros((0,), dtype=jt.int64), jt.zeros((0,), dtype=jt.int64)

    kk_np = kk.reshape(-1).numpy().astype(np.int64, copy=False)
    jj_np = jj.reshape(-1).numpy().astype(np.int64, copy=False)

    order = np.lexsort((np.arange(E, dtype=np.int64), jj_np, kk_np))
    kk_sorted = kk_np[order]

    ix_np = np.full(E, -1, dtype=np.int64)
    jx_np = np.full(E, -1, dtype=np.int64)
    if E > 1:
        same_group = kk_sorted[1:] == kk_sorted[:-1]
        ix_np[order[1:]] = np.where(same_group, order[:-1], -1)
        jx_np[order[:-1]] = np.where(same_group, order[1:], -1)

    return jt.array(ix_np).int64(), jt.array(jx_np).int64()


def neighbors(kk, jj, backend=None):
    """Find previous/next temporal neighbor edge indices grouped by patch kk and sorted by jj."""
    if _resolve_backend(backend) == "CUDA_INLINE":
        return _fastba_cuda.neighbors_cuda(kk, jj)
    return _neighbors_pure(kk, jj)


def _reproject_pure(poses, patches, intrinsics, ii, jj, kk):
    """Reproject patches from frame ii to frame jj matching ba_cuda.cu reproject."""
    poses_var = _extract_var(poses).reshape(-1, 7).float32()
    patches_var = _extract_var(patches)
    P = patches_var.shape[-1]
    patches_var = patches_var.reshape(-1, 3, P, P).float32()
    intr_var = intrinsics.reshape(-1, 4).float32()

    E = int(ii.numel())
    if E == 0:
        return jt.zeros((1, 0, 2, P, P), dtype=jt.float32)

    ii_i = ii.reshape(-1).int64()
    jj_i = jj.reshape(-1).int64()
    kk_i = kk.reshape(-1).int64()

    fx, fy, cx, cy = intr_var[0, 0], intr_var[0, 1], intr_var[0, 2], intr_var[0, 3]

    pi = poses_var[ii_i]
    pj = poses_var[jj_i]
    tij, qij = _rel_se3(pi[:, :3], pi[:, 3:7], pj[:, :3], pj[:, 3:7])

    pk = patches_var[kk_i]
    Xi_x = (pk[:, 0] - cx) / fx
    Xi_y = (pk[:, 1] - cy) / fy
    Xi_z = jt.ones_like(Xi_x)
    Xi_w = pk[:, 2]
    Xi = jt.stack([Xi_x, Xi_y, Xi_z, Xi_w], dim=-1)  # (E, P, P, 4)

    Xj = _act_se3(tij[:, None, None, :], qij[:, None, None, :], Xi)
    x = fx * (Xj[..., 0] / Xj[..., 2]) + cx
    y = fy * (Xj[..., 1] / Xj[..., 2]) + cy
    coords = jt.stack([x, y], dim=1)  # (E, 2, P, P)
    return coords.reshape(1, E, 2, P, P).float32()


def reproject(poses, patches, intrinsics, ii, jj, kk, backend=None):
    """Reproject patches from frame ii to frame jj matching ba_cuda.cu reproject."""
    if _resolve_backend(backend) == "CUDA_INLINE":
        return _fastba_cuda.reproject_cuda(poses, patches, intrinsics, ii, jj, kk)
    return _reproject_pure(poses, patches, intrinsics, ii, jj, kk)


def _ba_pure(
    poses,
    patches,
    intrinsics,
    target,
    weight,
    lmbda,
    ii,
    jj,
    kk,
    t0,
    t1,
    M,
    iterations,
    eff_impl=False,
    huber_delta=0.0,
    relative_depth_beta=0.0,
):
    """Gauss-Newton bundle adjustment with Schur complement, Huber loss, and relative depth damping."""
    del M, eff_impl  # Pure Jittor solves exact Schur complement uniformly.

    relative_depth_beta = float(relative_depth_beta)
    if not np.isfinite(relative_depth_beta) or relative_depth_beta < 0.0:
        raise RuntimeError("relative_depth_beta must be finite and nonnegative")

    huber_delta = float(huber_delta)

    poses_var = _extract_var(poses)
    patches_var = _extract_var(patches)

    if int(ii.numel()) == 0:
        return poses_var, patches_var

    P = patches_var.shape[-1]
    poses_w = poses_var.reshape(-1, 7).float32()
    patches_w = patches_var.reshape(-1, 3, P, P).float32()
    intr_w = intrinsics.reshape(-1, 4).float32()
    target_w = target.reshape(-1, 2).float32()
    weight_w = weight.reshape(-1, 2).float32()
    lmbda_w = (
        lmbda.reshape(-1).float32()
        if isinstance(lmbda, jt.Var)
        else jt.array(lmbda).reshape(-1).float32()
    )
    ii_w = ii.reshape(-1).int64()
    jj_w = jj.reshape(-1).int64()
    kk_w = kk.reshape(-1).int64()

    # Sorted unique kx and inverse indices ku matching the CUDA reference.
    # Note: Jittor 1.3.11 jt.unique corrupts int64 tensors, so compute via NumPy.
    kk_np = kk_w.numpy().astype(np.int64, copy=False)
    kx_np, ku_np = np.unique(kk_np, return_inverse=True)
    kx = jt.array(kx_np).int64()
    ku = jt.array(ku_np).int64()

    N = int(t1 - t0)
    M_uniq = int(kx.shape[0])
    fx, fy, cx, cy = intr_w[0, 0], intr_w[0, 1], intr_w[0, 2], intr_w[0, 3]
    I_mat = jt.init.eye(6 * N, dtype=jt.float32) if N > 0 else None

    for itr in range(iterations):
        pi = poses_w[ii_w]
        pj = poses_w[jj_w]
        tij, qij = _rel_se3(pi[:, :3], pi[:, 3:7], pj[:, :3], pj[:, 3:7])

        pk_c = patches_w[kk_w, :, 1, 1]
        Xi = jt.stack(
            [
                (pk_c[:, 0] - cx) / fx,
                (pk_c[:, 1] - cy) / fy,
                jt.ones_like(pk_c[:, 0]),
                pk_c[:, 2],
            ],
            dim=-1,
        )
        Xj = _act_se3(tij, qij, Xi)
        X, Y, Z, W = Xj[:, 0], Xj[:, 1], Xj[:, 2], Xj[:, 3]

        valid = (
            jt.isfinite(X)
            & jt.isfinite(Y)
            & jt.isfinite(Z)
            & jt.isfinite(W)
            & (Z > 0.2)
            & jt.isfinite(target_w[:, 0])
            & jt.isfinite(target_w[:, 1])
            & jt.isfinite(weight_w[:, 0])
            & jt.isfinite(weight_w[:, 1])
            & (weight_w[:, 0] >= 0.0)
            & (weight_w[:, 1] >= 0.0)
        )

        Z_safe = jt.where(valid, Z, jt.ones_like(Z))
        d = jt.where(valid, 1.0 / Z_safe, jt.zeros_like(Z))
        d2 = d * d

        x1 = fx * (X * d) + cx
        y1 = fy * (Y * d) + cy
        rx = target_w[:, 0] - x1
        ry = target_w[:, 1] - y1

        # Zero non-finite residuals before sqrt/bounds check.
        rx_safe = jt.where(valid, rx, jt.zeros_like(rx))
        ry_safe = jt.where(valid, ry, jt.zeros_like(ry))

        in_bounds = (
            valid
            & (jt.sqrt(rx_safe * rx_safe + ry_safe * ry_safe) < 128.0)
            & (Z > 0.2)
            & (x1 > -64.0)
            & (y1 > -64.0)
            & (x1 < 2.0 * cx + 64.0)
            & (y1 < 2.0 * cy + 64.0)
        )

        if huber_delta > 0.0:
            mean_w = 0.5 * (weight_w[:, 0] + weight_w[:, 1])
            safe_mw = jt.where(mean_w > 0.0, mean_w, jt.ones_like(mean_w))
            scaled_sq = (
                weight_w[:, 0] * rx_safe * rx_safe + weight_w[:, 1] * ry_safe * ry_safe
            ) / safe_mw
            huber_cond = in_bounds & (mean_w > 0.0) & (scaled_sq > huber_delta * huber_delta)
            safe_sq = jt.maximum(scaled_sq, 1e-12)
            robust_w = jt.where(
                huber_cond, huber_delta / jt.sqrt(safe_sq), jt.ones_like(rx_safe)
            )
        else:
            robust_w = jt.ones_like(rx_safe)

        mask0 = in_bounds & (weight_w[:, 0] != 0.0)
        mask1 = in_bounds & (weight_w[:, 1] != 0.0)

        w0 = jt.where(mask0, weight_w[:, 0] * robust_w, jt.zeros_like(rx_safe))
        w1 = jt.where(mask1, weight_w[:, 1] * robust_w, jt.zeros_like(ry_safe))
        r0 = jt.where(mask0, rx_safe, jt.zeros_like(rx_safe))
        r1 = jt.where(mask1, ry_safe, jt.zeros_like(ry_safe))

        zeros_e = jt.zeros_like(d)
        tij_safe = jt.where(valid[:, None], tij, jt.zeros_like(tij))
        qij_safe = jt.where(
            valid[:, None],
            qij,
            jt.stack([zeros_e, zeros_e, zeros_e, jt.ones_like(d)], dim=-1),
        )
        X_s = jt.where(valid, X, zeros_e)
        Y_s = jt.where(valid, Y, zeros_e)
        W_s = jt.where(valid, W, zeros_e)

        Jz0 = jt.where(mask0, fx * (tij_safe[:, 0] * d - tij_safe[:, 2] * (X_s * d2)), zeros_e)
        Jj0 = jt.stack(
            [
                fx * W_s * d,
                zeros_e,
                fx * -X_s * W_s * d2,
                fx * -X_s * Y_s * d2,
                fx * (1.0 + X_s * X_s * d2),
                fx * -Y_s * d,
            ],
            dim=-1,
        )
        Jj0 = jt.where(mask0[:, None], Jj0, jt.zeros_like(Jj0))
        Ji0 = jt.where(mask0[:, None], _adj_se3(tij_safe, qij_safe, Jj0), jt.zeros_like(Jj0))

        Jz1 = jt.where(mask1, fy * (tij_safe[:, 1] * d - tij_safe[:, 2] * (Y_s * d2)), zeros_e)
        Jj1 = jt.stack(
            [
                zeros_e,
                fy * W_s * d,
                fy * -Y_s * W_s * d2,
                fy * (-1.0 - Y_s * Y_s * d2),
                fy * (X_s * Y_s * d2),
                fy * X_s * d,
            ],
            dim=-1,
        )
        Jj1 = jt.where(mask1[:, None], Jj1, jt.zeros_like(Jj1))
        Ji1 = jt.where(mask1[:, None], _adj_se3(tij_safe, qij_safe, Jj1), jt.zeros_like(Jj1))

        c_edge = w0 * Jz0 * Jz0 + w1 * Jz1 * Jz1
        u_edge = w0 * r0 * Jz0 + w1 * r1 * Jz1
        C = c_edge.float64().reindex_reduce("add", [M_uniq], ["@e0(i0)"], extras=[ku]).float32()
        u = (
            u_edge.float64()
            .reindex_reduce("add", [M_uniq], ["@e0(i0)"], extras=[ku])
            .float32()
            .reshape(M_uniq, 1)
        )

        if relative_depth_beta > 0.0:
            depth = patches_w[kx, 2, 1, 1]
            d_valid = jt.isfinite(depth) & (depth > 0.0)
            d_safe = jt.where(d_valid, depth, jt.zeros_like(depth))
            d_count = jt.maximum(d_valid.float32().sum(), 1.0)
            scale = d_safe.sum() / d_count
            rho = jt.maximum(d_safe, 0.1 * scale)
            denom = jt.where(d_valid, rho, jt.ones_like(depth))
            extra = jt.where(
                d_valid, relative_depth_beta / (denom * denom), jt.zeros_like(depth)
            )
            Q = (1.0 / (C + lmbda_w + extra)).reshape(1, M_uniq)
        else:
            Q = (1.0 / (C + lmbda_w)).reshape(1, M_uniq)

        if N == 0:
            dZ = (Q.reshape(M_uniq) * u.reshape(M_uniq))
            dZ_np = dZ.numpy()
            if not np.isfinite(dZ_np).all():
                raise RuntimeError(
                    f"DPVO numerical: non-finite depth update at BA iteration {itr}"
                )
        else:
            ix_rel = ii_w - t0
            jx_rel = jj_w - t0
            act_i = (ix_rel >= 0) & (ix_rel < N)
            act_j = (jx_rel >= 0) & (jx_rel < N)
            safe_ix = jt.where(act_i, ix_rel, jt.zeros_like(ix_rel))
            safe_jx = jt.where(act_j, jx_rel, jt.zeros_like(jx_rel))
            ai = act_i.float32()
            aj = act_j.float32()
            aij = (act_i & act_j).float32()

            B_ii = ai[:, None, None] * (
                w0[:, None, None] * (Ji0[:, :, None] * Ji0[:, None, :])
                + w1[:, None, None] * (Ji1[:, :, None] * Ji1[:, None, :])
            )
            B_jj = aj[:, None, None] * (
                w0[:, None, None] * (Jj0[:, :, None] * Jj0[:, None, :])
                + w1[:, None, None] * (Jj1[:, :, None] * Jj1[:, None, :])
            )
            B_ij = -aij[:, None, None] * (
                w0[:, None, None] * (Ji0[:, :, None] * Jj0[:, None, :])
                + w1[:, None, None] * (Ji1[:, :, None] * Jj1[:, None, :])
            )
            B_ji = B_ij.transpose(0, 2, 1)

            B_blocks = jt.concat([B_ii, B_jj, B_ij, B_ji], dim=0)
            row_idx = jt.concat([safe_ix, safe_jx, safe_ix, safe_jx], dim=0)
            col_idx = jt.concat([safe_ix, safe_jx, safe_jx, safe_ix], dim=0)
            B_4d = B_blocks.float64().reindex_reduce(
                "add",
                [N, N, 6, 6],
                ["@e0(i0)", "@e1(i0)", "i1", "i2"],
                extras=[row_idx, col_idx],
            ).float32()
            B = B_4d.permute(0, 2, 1, 3).reshape(6 * N, 6 * N)

            E_i = -ai[:, None] * (w0[:, None] * Jz0[:, None] * Ji0 + w1[:, None] * Jz1[:, None] * Ji1)
            E_j = aj[:, None] * (w0[:, None] * Jz0[:, None] * Jj0 + w1[:, None] * Jz1[:, None] * Jj1)
            E_blocks = jt.concat([E_i, E_j], dim=0)
            E_pose_idx = jt.concat([safe_ix, safe_jx], dim=0)
            E_patch_idx = jt.concat([ku, ku], dim=0)
            E_3d = E_blocks.float64().reindex_reduce(
                "add",
                [N, 6, M_uniq],
                ["@e0(i0)", "i1", "@e1(i0)"],
                extras=[E_pose_idx, E_patch_idx],
            ).float32()
            E = E_3d.reshape(6 * N, M_uniq)

            v_i = -ai[:, None] * (w0[:, None] * r0[:, None] * Ji0 + w1[:, None] * r1[:, None] * Ji1)
            v_j = aj[:, None] * (w0[:, None] * r0[:, None] * Jj0 + w1[:, None] * r1[:, None] * Jj1)
            v_blocks = jt.concat([v_i, v_j], dim=0)
            v_2d = v_blocks.float64().reindex_reduce(
                "add",
                [N, 6],
                ["@e0(i0)", "i1"],
                extras=[E_pose_idx],
            ).float32()
            v = v_2d.reshape(6 * N, 1)

            EQ = E * Q
            Et = E.transpose(0, 1)
            S = B - jt.matmul(EQ, Et)
            y = v - jt.matmul(EQ, u)
            S = S + I_mat * (1e-4 * S + 1.0)

            S_np = S.numpy()
            y_np = y.numpy()
            if not (np.isfinite(S_np).all() and np.isfinite(y_np).all()):
                raise RuntimeError(
                    f"DPVO numerical: non-finite Schur system at BA iteration {itr}"
                )
            try:
                c_fac, lower = scipy.linalg.cho_factor(S_np, lower=True, check_finite=False)
                dX_np = scipy.linalg.cho_solve((c_fac, lower), y_np, check_finite=False)
            except (np.linalg.LinAlgError, scipy.linalg.LinAlgError) as exc:
                raise RuntimeError(
                    f"DPVO numerical: Cholesky factorization failed at BA iteration {itr}"
                ) from exc

            dX = jt.array(dX_np).float32().reshape(N, 6)
            dZ = (Q.reshape(M_uniq, 1) * (u - jt.matmul(Et, dX.reshape(6 * N, 1)))).reshape(
                M_uniq
            )
            dZ_np = dZ.numpy()
            if not (np.isfinite(dX_np).all() and np.isfinite(dZ_np).all()):
                raise RuntimeError(
                    f"DPVO numerical: non-finite pose/depth update at BA iteration {itr}"
                )

            t_new, q_new = _retr_se3(dX, poses_w[t0:t1, :3], poses_w[t0:t1, 3:7])
            poses_w[t0:t1] = jt.concat([t_new, q_new], dim=-1)

        d_new = patches_w[kx, 2, 0, 0] + dZ
        d_new = jt.where(d_new > 20.0, jt.ones_like(d_new), d_new)
        d_new = jt.maximum(d_new, 1e-4)
        patches_w[kx, 2] = d_new[:, None, None].broadcast([M_uniq, P, P])

    poses_var.assign(poses_w.reshape(poses_var.shape).cast(poses_var.dtype))
    patches_var.assign(patches_w.reshape(patches_var.shape).cast(patches_var.dtype))
    return poses_var, patches_var


def BA(
    poses,
    patches,
    intrinsics,
    target,
    weight,
    lmbda,
    ii,
    jj,
    kk,
    t0,
    t1,
    M,
    iterations,
    eff_impl=False,
    huber_delta=0.0,
    relative_depth_beta=0.0,
    backend=None,
):
    """Gauss-Newton bundle adjustment with Schur complement, Huber loss, and relative depth damping."""
    if _resolve_backend(backend) == "CUDA_INLINE":
        return _fastba_cuda.BA_cuda(
            poses,
            patches,
            intrinsics,
            target,
            weight,
            lmbda,
            ii,
            jj,
            kk,
            t0,
            t1,
            M,
            iterations,
            eff_impl=eff_impl,
            huber_delta=huber_delta,
            relative_depth_beta=relative_depth_beta,
        )
    return _ba_pure(
        poses,
        patches,
        intrinsics,
        target,
        weight,
        lmbda,
        ii,
        jj,
        kk,
        t0,
        t1,
        M,
        iterations,
        eff_impl=eff_impl,
        huber_delta=huber_delta,
        relative_depth_beta=relative_depth_beta,
    )
