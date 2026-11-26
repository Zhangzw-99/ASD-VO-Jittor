import jittor as jt

from .se3 import SE3

MIN_DEPTH = 0.2


def extract_intrinsics(intrinsics):
    return intrinsics[..., None, None, :].unbind(dim=-1)


def coords_grid(ht, wd, **kwargs):
    dtype = kwargs.get("dtype", "float32")
    if dtype in (float, "float"):
        dtype = "float32"
    y, x = jt.meshgrid(
        jt.arange(ht).cast(dtype),
        jt.arange(wd).cast(dtype),
    )
    return jt.stack([x, y], dim=-1)


def iproj(patches, intrinsics):
    """Inverse projection."""
    x, y, d = patches.unbind(dim=2)
    fx, fy, cx, cy = intrinsics[..., None, None].unbind(dim=2)

    i = jt.ones_like(d)
    xn = (x - cx) / fx
    yn = (y - cy) / fy

    X = jt.stack([xn, yn, i, d], dim=-1)
    return X


def proj(X, intrinsics, depth=False):
    """Projection."""
    X, Y, Z, W = X.unbind(dim=-1)
    fx, fy, cx, cy = intrinsics[..., None, None].unbind(dim=2)

    d = 1.0 / jt.maximum(Z, 0.1)
    x = fx * (d * X) + cx
    y = fy * (d * Y) + cy

    if depth:
        return jt.stack([x, y, d], dim=-1)

    return jt.stack([x, y], dim=-1)


def transform(poses, patches, intrinsics, ii, jj, kk, depth=False, valid=False, jacobian=False, tonly=False):
    """Projective transform."""
    if ii.shape[0] == 0:
        b = poses.shape[0]
        p = patches.shape[3]
        c = 3 if depth else 2
        x1 = jt.zeros((b, 0, p, p, c), dtype="float32")
        if jacobian:
            Ji = jt.zeros((b, 0, 2, 6), dtype="float32")
            Jj = jt.zeros((b, 0, 2, 6), dtype="float32")
            Jz = jt.zeros((b, 0, 2, 1), dtype="float32")
            return x1, jt.zeros((b, 0), dtype="float32"), (Ji, Jj, Jz)
        if valid:
            return x1, jt.zeros((b, 0, p, p), dtype="float32")
        return x1

    # backproject
    X0 = iproj(patches[:, kk], intrinsics[:, ii])

    # transform
    Gij = poses[:, jj] * poses[:, ii].inv()

    if tonly:
        id_q = jt.array([0.0, 0.0, 0.0, 1.0], dtype=Gij.dtype).broadcast(Gij.data[..., 3:].shape)
        Gij = SE3(jt.concat([Gij.data[..., :3], id_q], dim=-1))

    X1 = Gij[:, :, None, None] * X0

    # project
    x1 = proj(X1, intrinsics[:, jj], depth)

    if jacobian:
        p = X1.shape[2]
        X, Y, Z, H = X1[..., p // 2, p // 2, :].unbind(dim=-1)
        o = jt.zeros_like(H)

        fx, fy, cx, cy = intrinsics[:, jj].unbind(dim=-1)

        mask_z = jt.abs(Z) > 0.2
        z_safe = jt.ternary(mask_z, Z, jt.ones_like(Z))
        d = jt.ternary(mask_z, 1.0 / z_safe, jt.zeros_like(Z))

        Ja = jt.stack([
            H, o, o, o,  Z, -Y,
            o, H, o, -Z, o,  X,
            o, o, H,  Y, -X, o,
            o, o, o,  o,  o, o,
        ], dim=-1).reshape(1, ii.shape[0], 4, 6)

        Jp = jt.stack([
            fx * d, o, -fx * X * d * d, o,
            o, fy * d, -fy * Y * d * d, o,
        ], dim=-1).reshape(1, ii.shape[0], 2, 4)

        Jj = jt.matmul(Jp, Ja)
        Ji = -Gij[:, :, None].adjT(Jj)

        Jz = jt.matmul(Jp, Gij.matrix()[..., :, 3:])

        return x1, (Z > 0.2).float32(), (Ji, Jj, Jz)

    if valid:
        return x1, (X1[..., 2] > 0.2).float32()

    return x1


def point_cloud(poses, patches, intrinsics, ix):
    """Generate point cloud from patches."""
    if ix.shape[0] == 0:
        b = poses.shape[0]
        p = patches.shape[3]
        return jt.zeros((b, 0, p, p, 4), dtype="float32")
    return poses[:, ix, None, None].inv() * iproj(patches, intrinsics[:, ix])


def flow_mag(poses, patches, intrinsics, ii, jj, kk, beta=0.3):
    """Projective optical flow magnitude."""
    coords0 = transform(poses, patches, intrinsics, ii, ii, kk)
    coords1, val = transform(poses, patches, intrinsics, ii, jj, kk, tonly=False, valid=True)
    coords2 = transform(poses, patches, intrinsics, ii, jj, kk, tonly=True)

    d1 = coords1 - coords0
    d2 = coords2 - coords0
    flow1 = jt.sqrt((d1 * d1).sum(dim=-1))
    flow2 = jt.sqrt((d2 * d2).sum(dim=-1))

    return beta * flow1 + (1 - beta) * flow2, (val > 0.5)
