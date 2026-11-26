"""Pure Jittor SE(3) Lie group implementation matching [tx, ty, tz, qx, qy, qz, qw]."""

import numpy as np
import jittor as jt


_EPS = 1e-6
_PI = 3.14159265358979323846


def _normalize_dtype(dtype):
    if dtype is None:
        return "float32"
    if dtype in (float, "float", "float32", jt.float32):
        return "float32"
    if dtype in ("double", "float64", jt.float64):
        return "float64"
    if dtype in ("half", "float16", jt.float16):
        return "float16"
    return str(dtype)


def _cross(a, b):
    """3D cross product along the last axis."""
    ax, ay, az = a[..., 0:1], a[..., 1:2], a[..., 2:3]
    bx, by, bz = b[..., 0:1], b[..., 1:2], b[..., 2:3]
    return jt.concat([
        ay * bz - az * by,
        az * bx - ax * bz,
        ax * by - ay * bx,
    ], dim=-1)


def _normalize_quat(q):
    """Normalize quaternion [qx, qy, qz, qw] along the last axis."""
    norm = jt.sqrt(jt.maximum((q * q).sum(dim=-1, keepdims=True), 1e-24))
    return q / norm


def _quat_conj(q):
    """Conjugate of unit quaternion [qx, qy, qz, qw]."""
    return jt.concat([-q[..., :3], q[..., 3:4]], dim=-1)


def _quat_mul(q1, q2):
    """Hamilton product of quaternions [qx, qy, qz, qw]."""
    v1, w1 = q1[..., :3], q1[..., 3:4]
    v2, w2 = q2[..., :3], q2[..., 3:4]
    w = w1 * w2 - (v1 * v2).sum(dim=-1, keepdims=True)
    v = w1 * v2 + w2 * v1 + _cross(v1, v2)
    return _normalize_quat(jt.concat([v, w], dim=-1))


def _quat_act(q, p):
    """Rotate 3D point(s) p by unit quaternion q [qx, qy, qz, qw]."""
    q = _normalize_quat(q)
    v, w = q[..., :3], q[..., 3:4]
    uv = 2.0 * _cross(v, p)
    return p + w * uv + _cross(v, uv)


def _so3_hat(phi):
    """Skew-symmetric matrix [phi]_x of shape (..., 3, 3)."""
    px, py, pz = phi[..., 0:1], phi[..., 1:2], phi[..., 2:3]
    o = jt.zeros_like(px)
    row0 = jt.concat([o, -pz, py], dim=-1)
    row1 = jt.concat([pz, o, -px], dim=-1)
    row2 = jt.concat([-py, px, o], dim=-1)
    return jt.stack([row0, row1, row2], dim=-2)


def _so3_matrix(q):
    """Convert quaternion [qx, qy, qz, qw] to 3x3 rotation matrix."""
    q = _normalize_quat(q)
    x, y, z, w = q[..., 0:1], q[..., 1:2], q[..., 2:3], q[..., 3:4]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w

    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - zw)
    r02 = 2.0 * (xz + yw)
    r10 = 2.0 * (xy + zw)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - xw)
    r20 = 2.0 * (xz - yw)
    r21 = 2.0 * (yz + xw)
    r22 = 1.0 - 2.0 * (xx + yy)

    row0 = jt.concat([r00, r01, r02], dim=-1)
    row1 = jt.concat([r10, r11, r12], dim=-1)
    row2 = jt.concat([r20, r21, r22], dim=-1)
    return jt.stack([row0, row1, row2], dim=-2)


def _so3_exp(phi):
    """SO(3) exponential map from rotation vector phi (..., 3) to quaternion (..., 4)."""
    theta2 = (phi * phi).sum(dim=-1, keepdims=True)
    theta = jt.sqrt(theta2)
    theta4 = theta2 * theta2

    imag_small = 0.5 - (1.0 / 48.0) * theta2 + (1.0 / 3840.0) * theta4
    real_small = 1.0 - (1.0 / 8.0) * theta2 + (1.0 / 384.0) * theta4

    theta_safe = jt.maximum(theta, _EPS)
    imag_large = jt.sin(0.5 * theta) / theta_safe
    real_large = jt.cos(0.5 * theta)

    small_mask = theta < _EPS
    imag_factor = jt.ternary(small_mask, imag_small, imag_large)
    real_factor = jt.ternary(small_mask, real_small, real_large)

    return _normalize_quat(jt.concat([imag_factor * phi, real_factor], dim=-1))


def _so3_log(q):
    """SO(3) logarithm map using Hertzberg atan formula."""
    q = _normalize_quat(q)
    v = q[..., :3]
    w = q[..., 3:4]
    squared_n = (v * v).sum(dim=-1, keepdims=True)

    w_safe = jt.ternary(jt.abs(w) < _EPS, jt.ones_like(w), w)
    small_val = 2.0 / w_safe - (2.0 / 3.0) * squared_n / (w_safe * w_safe * w_safe)

    n = jt.sqrt(jt.maximum(squared_n, 1e-24))
    pi_by_n = jt.ternary(w > 0.0, _PI / n, -_PI / n)
    atan_val = 2.0 * jt.atan(n / w_safe) / n
    large_val = jt.ternary(jt.abs(w) < _EPS, pi_by_n, atan_val)

    two_atan_nbyw_by_n = jt.ternary(squared_n < (_EPS * _EPS), small_val, large_val)
    return two_atan_nbyw_by_n * v


def _so3_left_jacobian(phi):
    """SO(3) left Jacobian J(phi) of shape (..., 3, 3)."""
    theta2 = (phi * phi).sum(dim=-1, keepdims=True)
    theta = jt.sqrt(theta2)
    theta_safe = jt.maximum(theta, _EPS)
    theta2_safe = theta_safe * theta_safe

    coef1_small = 0.5 - (1.0 / 24.0) * theta2
    coef1_large = (1.0 - jt.cos(theta)) / theta2_safe
    coef2_small = (1.0 / 6.0) - (1.0 / 120.0) * theta2
    coef2_large = (theta - jt.sin(theta)) / (theta2_safe * theta_safe)

    small_mask = theta < _EPS
    coef1 = jt.ternary(small_mask, coef1_small, coef1_large)[..., None]
    coef2 = jt.ternary(small_mask, coef2_small, coef2_large)[..., None]

    Phi = _so3_hat(phi)
    Phi2 = jt.matmul(Phi, Phi)
    I = jt.array(np.eye(3), dtype=phi.dtype).reshape([1] * (phi.ndim - 1) + [3, 3])
    return I + coef1 * Phi + coef2 * Phi2


def _so3_left_jacobian_inverse(phi):
    """Inverse of SO(3) left Jacobian J^{-1}(phi) of shape (..., 3, 3)."""
    theta2 = (phi * phi).sum(dim=-1, keepdims=True)
    theta = jt.sqrt(theta2)
    half_theta = 0.5 * theta

    theta_safe = jt.maximum(theta, _EPS)
    sin_half_safe = jt.ternary(theta < _EPS, jt.ones_like(theta), jt.sin(half_theta))
    coef2_large = (1.0 - theta * jt.cos(half_theta) / (2.0 * sin_half_safe)) / (theta_safe * theta_safe)
    coef2_small = jt.full_like(theta, 1.0 / 12.0)

    coef2 = jt.ternary(theta < _EPS, coef2_small, coef2_large)[..., None]

    Phi = _so3_hat(phi)
    Phi2 = jt.matmul(Phi, Phi)
    I = jt.array(np.eye(3), dtype=phi.dtype).reshape([1] * (phi.ndim - 1) + [3, 3])
    return I - 0.5 * Phi + coef2 * Phi2


class SE3:
    """SE(3) Lie group backed by Jittor Var with layout [tx, ty, tz, qx, qy, qz, qw]."""

    group_name = "SE3"
    manifold_dim = 6
    embedded_dim = 7

    def __init__(self, data):
        if isinstance(data, SE3):
            data = data.data
        self.data = data

    def __repr__(self):
        return f"SE3: size={self.shape}, dtype={self.dtype}"

    @property
    def shape(self):
        return tuple(self.data.shape[:-1])

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def tangent_shape(self):
        return tuple(self.data.shape[:-1]) + (self.manifold_dim,)

    def vec(self):
        return self.data

    @classmethod
    def Identity(cls, *batch_shape, **kwargs):
        """Construct identity SE(3) element(s) with the given batch shape."""
        if len(batch_shape) == 1 and isinstance(batch_shape[0], (tuple, list, jt.NanoVector)):
            batch_shape = tuple(batch_shape[0])
        dtype = _normalize_dtype(kwargs.get("dtype", "float32"))
        numel = int(np.prod(batch_shape)) if len(batch_shape) > 0 else 1
        elem = jt.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).cast(dtype).reshape(1, 7)
        data = elem.repeat(numel, 1).reshape(tuple(batch_shape) + (7,))
        return cls(data)

    @classmethod
    def IdentityLike(cls, G):
        """Construct identity SE(3) element(s) matching batch shape and dtype of G."""
        return cls.Identity(G.shape, dtype=G.data.dtype)

    @classmethod
    def Random(cls, *batch_shape, sigma=1.0, **kwargs):
        """Construct random SE(3) element(s) by sampling in the tangent space."""
        if len(batch_shape) == 1 and isinstance(batch_shape[0], (tuple, list, jt.NanoVector)):
            batch_shape = tuple(batch_shape[0])
        dtype = _normalize_dtype(kwargs.get("dtype", "float32"))
        xi = jt.randn(tuple(batch_shape) + (cls.manifold_dim,), dtype=dtype) * sigma
        return cls.exp(xi)

    @classmethod
    def exp(cls, xi):
        """Exponential map from se(3) [tau, phi] (..., 6) to SE(3) (..., 7)."""
        tau = xi[..., :3]
        phi = xi[..., 3:6]
        q = _so3_exp(phi)
        V = _so3_left_jacobian(phi)
        t = jt.matmul(V, tau[..., None]).squeeze(-1)
        return cls(jt.concat([t, q], dim=-1))

    def log(self):
        """Logarithm map from SE(3) (..., 7) to se(3) [tau, phi] (..., 6)."""
        t = self.data[..., :3]
        q = self.data[..., 3:7]
        phi = _so3_log(q)
        Vinv = _so3_left_jacobian_inverse(phi)
        tau = jt.matmul(Vinv, t[..., None]).squeeze(-1)
        return jt.concat([tau, phi], dim=-1)

    def inv(self):
        """Group inverse."""
        t = self.data[..., :3]
        q = _normalize_quat(self.data[..., 3:7])
        q_inv = _quat_conj(q)
        t_inv = -_quat_act(q_inv, t)
        return SE3(jt.concat([t_inv, q_inv], dim=-1))

    def mul(self, other):
        """Group multiplication."""
        t1, q1 = self.data[..., :3], _normalize_quat(self.data[..., 3:7])
        t2, q2 = other.data[..., :3], _normalize_quat(other.data[..., 3:7])
        q = _quat_mul(q1, q2)
        t = t1 + _quat_act(q1, t2)
        return SE3(jt.concat([t, q], dim=-1))

    def retr(self, a):
        """Left retraction: Exp(a) * self."""
        return SE3.exp(a) * self

    def adj(self, a):
        """Adjoint operator: b = Adj(X) * a."""
        t = self.data[..., :3]
        q = _normalize_quat(self.data[..., 3:7])
        a_v, a_w = a[..., :3], a[..., 3:6]
        b_w = _quat_act(q, a_w)
        b_v = _quat_act(q, a_v) + _cross(t, b_w)
        return jt.concat([b_v, b_w], dim=-1)

    def adjT(self, a):
        """Transposed adjoint operator: b = Adj(X)^T * a."""
        t = self.data[..., :3]
        q_inv = _quat_conj(_normalize_quat(self.data[..., 3:7]))
        a_v, a_w = a[..., :3], a[..., 3:6]
        b_v = _quat_act(q_inv, a_v)
        b_w = _quat_act(q_inv, a_w + _cross(-t, a_v))
        return jt.concat([b_v, b_w], dim=-1)

    def act(self, p):
        """Action on 3D or homogeneous 4D points."""
        t = self.data[..., :3]
        q = self.data[..., 3:7]
        if p.shape[-1] == 3:
            return _quat_act(q, p) + t
        if p.shape[-1] == 4:
            xyz = _quat_act(q, p[..., :3]) + t * p[..., 3:4]
            w = p[..., 3:4].broadcast(xyz.shape[:-1] + (1,))
            return jt.concat([xyz, w], dim=-1)
        raise ValueError(f"Expected point last dim 3 or 4, got {p.shape[-1]}")

    def matrix(self):
        """Convert SE(3) element(s) to 4x4 homogeneous transformation matrix."""
        t = self.data[..., :3]
        R = _so3_matrix(self.data[..., 3:7])
        top = jt.concat([R, t[..., None]], dim=-1)
        bottom_row = jt.array([0.0, 0.0, 0.0, 1.0], dtype=self.dtype)
        bottom_shape = self.data.shape[:-1] + (1, 4)
        bottom = bottom_row.reshape([1] * (len(bottom_shape) - 1) + [4]).broadcast(bottom_shape)
        return jt.concat([top, bottom], dim=-2)

    def translation(self):
        """Extract homogeneous translation vector [tx, ty, tz, 1.0]."""
        t = self.data[..., :3]
        ones = jt.ones(t.shape[:-1] + (1,), dtype=self.dtype)
        return jt.concat([t, ones], dim=-1)

    def scale(self, s):
        """Scale translation component by s while leaving rotation unchanged."""
        t = self.data[..., :3]
        q = self.data[..., 3:7]
        if isinstance(s, jt.Var):
            if s.ndim > 0 and s.ndim >= t.ndim:
                pass
            elif s.ndim > 0 and s.shape != (1,):
                s = s.unsqueeze(-1)
        t = t * s
        return SE3(jt.concat([t, q], dim=-1))

    def detach(self):
        return SE3(self.data.detach())

    def clone(self):
        return SE3(self.data.clone())

    def view(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], (tuple, list, jt.NanoVector)):
            dims = tuple(dims[0])
        return SE3(self.data.reshape(tuple(dims) + (self.embedded_dim,)))

    def reshape(self, *dims):
        return self.view(*dims)

    def __mul__(self, other):
        if isinstance(other, SE3):
            return self.mul(other)
        if isinstance(other, jt.Var):
            return self.act(other)
        return NotImplemented

    def __getitem__(self, index):
        if index is Ellipsis:
            index = (Ellipsis, slice(None))
        elif isinstance(index, tuple) and any(x is Ellipsis for x in index):
            index = index + (slice(None),)
        return SE3(self.data[index])

    def __setitem__(self, index, item):
        val = item.data if isinstance(item, SE3) else item
        if index is Ellipsis:
            index = (Ellipsis, slice(None))
        elif isinstance(index, tuple) and any(x is Ellipsis for x in index):
            index = index + (slice(None),)
        self.data[index] = val

    def float(self, device=None):
        return SE3(self.data.float32())

    def double(self, device=None):
        return SE3(self.data.float64())

    def float32(self):
        return SE3(self.data.float32())

    def float64(self):
        return SE3(self.data.float64())

    def float16(self):
        return SE3(self.data.float16())

    def numpy(self):
        return self.data.numpy()

    def unbind(self, dim=0):
        return [SE3(x) for x in self.data.unbind(dim=dim)]


def cat(group_list, dim=0):
    """Concatenate SE3 objects along dimension."""
    data = jt.concat([X.data for X in group_list], dim=dim)
    return group_list[0].__class__(data)


def stack(group_list, dim=0):
    """Stack SE3 objects along a new dimension."""
    data = jt.stack([X.data for X in group_list], dim=dim)
    return group_list[0].__class__(data)
