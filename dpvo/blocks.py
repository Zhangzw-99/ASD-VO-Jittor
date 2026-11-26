import jittor as jt
import jittor.nn as nn


class LayerNorm1D(nn.Module):
    def __init__(self, dim):
        super(LayerNorm1D, self).__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-4)

    def execute(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid())

        self.res = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim))

    def execute(self, x):
        return x + self.gate(x) * self.res(x)


def _soft_agg_execute(x, ix, f, g, h, dim, expand):
    if x.shape[1] == 0:
        return jt.zeros((x.shape[0], 0, dim), dtype=x.dtype)

    b = x.shape[0]
    uniq, jx = jt.unique(ix.int32(), return_inverse=True)
    k = uniq.shape[0]

    gx = g(x)
    d_g = gx.shape[2]
    max_gx = gx.reindex_reduce(
        "maximum", [b, k, d_g], ["i0", "@e0(i1)", "i2"], extras=[jx]
    )
    exp_gx = jt.exp(gx - max_gx[:, jx])
    sum_exp = exp_gx.reindex_reduce(
        "add", [b, k, d_g], ["i0", "@e0(i1)", "i2"], extras=[jx]
    )
    w = exp_gx / sum_exp[:, jx]

    y = (f(x) * w).reindex_reduce(
        "add", [b, k, dim], ["i0", "@e0(i1)", "i2"], extras=[jx]
    )
    out = h(y)
    if expand:
        return out[:, jx]
    return out


class SoftAgg(nn.Module):
    def __init__(self, dim=512, expand=True):
        super(SoftAgg, self).__init__()
        self.dim = dim
        self.expand = expand
        self.f = nn.Linear(self.dim, self.dim)
        self.g = nn.Linear(self.dim, self.dim)
        self.h = nn.Linear(self.dim, self.dim)

    def execute(self, x, ix):
        return _soft_agg_execute(x, ix, self.f, self.g, self.h, self.dim, self.expand)


class SoftAggBasic(nn.Module):
    def __init__(self, dim=512, expand=True):
        super(SoftAggBasic, self).__init__()
        self.dim = dim
        self.expand = expand
        self.f = nn.Linear(self.dim, self.dim)
        self.g = nn.Linear(self.dim, 1)
        self.h = nn.Linear(self.dim, self.dim)

    def execute(self, x, ix):
        return _soft_agg_execute(x, ix, self.f, self.g, self.h, self.dim, self.expand)


### Gradient Clipping and Zeroing Operations (identity in inference) ###

GRAD_CLIP = 0.1


class GradientClip(nn.Module):
    def __init__(self):
        super(GradientClip, self).__init__()

    def execute(self, x):
        return x


class GradientZero(nn.Module):
    def __init__(self):
        super(GradientZero, self).__init__()

    def execute(self, x):
        return x
