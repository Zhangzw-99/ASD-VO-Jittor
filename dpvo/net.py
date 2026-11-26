import numpy as np
import jittor as jt
import jittor.nn as nn

from . import fastba
from . import altcorr
from .se3 import SE3

from .extractor import BasicEncoder, BasicEncoder4
from .blocks import GradientClip, GatedResidual, SoftAgg

from .utils import coords_grid_with_index, pyramidify, set_depth, flatmeshgrid
from . import projective_ops as pops


DIM = 384


class Update(nn.Module):
    def __init__(self, p):
        super(Update, self).__init__()

        self.c1 = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(),
            nn.Linear(DIM, DIM))

        self.c2 = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(),
            nn.Linear(DIM, DIM))

        self.norm = nn.LayerNorm(DIM, eps=1e-3)

        self.agg_kk = SoftAgg(DIM)
        self.agg_ij = SoftAgg(DIM)

        self.gru = nn.Sequential(
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
        )

        self.corr = nn.Sequential(
            nn.Linear(2 * 49 * p * p, DIM),
            nn.ReLU(),
            nn.Linear(DIM, DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            nn.ReLU(),
            nn.Linear(DIM, DIM),
        )

        self.d = nn.Sequential(
            nn.ReLU(),
            nn.Linear(DIM, 2),
            GradientClip())

        self.w = nn.Sequential(
            nn.ReLU(),
            nn.Linear(DIM, 2),
            GradientClip(),
            nn.Sigmoid())

    def execute(self, net, inp, corr, flow, ii, jj, kk):
        """Update operator."""
        if net.dtype != 'float32':
            net = net.float32()
        if inp.dtype != 'float32':
            inp = inp.float32()
        if corr.dtype != 'float32':
            corr = corr.float32()

        if net.shape[1] == 0:
            b = net.shape[0]
            return net, (
                jt.zeros((b, 0, 2), dtype='float32'),
                jt.zeros((b, 0, 2), dtype='float32'),
                None,
            )

        net = net + inp + self.corr(corr)
        net = self.norm(net)

        ix, jx = fastba.neighbors(kk, jj)
        mask_ix = (ix >= 0).float32().reshape(1, -1, 1)
        mask_jx = (jx >= 0).float32().reshape(1, -1, 1)
        ix_safe = jt.maximum(ix, 0)
        jx_safe = jt.maximum(jx, 0)

        net = net + self.c1(mask_ix * net[:, ix_safe])
        net = net + self.c2(mask_jx * net[:, jx_safe])

        net = net + self.agg_kk(net, kk)
        net = net + self.agg_ij(net, ii * 12345 + jj)

        net = self.gru(net)

        return net, (self.d(net), self.w(net), None)


class Patchifier(nn.Module):
    def __init__(self, patch_size=3):
        super(Patchifier, self).__init__()
        self.patch_size = patch_size
        self.fnet = BasicEncoder4(output_dim=128, norm_fn='instance')
        self.inet = BasicEncoder4(output_dim=DIM, norm_fn='none')

    def __image_gradient(self, images):
        gray = ((images + 0.5) * (255.0 / 2)).sum(dim=2)
        dx = gray[..., :-1, 1:] - gray[..., :-1, :-1]
        dy = gray[..., 1:, :-1] - gray[..., :-1, :-1]
        g = jt.sqrt(dx ** 2 + dy ** 2)
        g = nn.avg_pool2d(g, 4, stride=4)
        return g

    def execute(self, images, patches_per_image=80, disps=None, centroid_sel_strat='RANDOM', return_color=False):
        """Extract patches from input images."""
        fmap = self.fnet(images) / 4.0
        imap = self.inet(images) / 4.0

        b, n, c, h, w = fmap.shape
        P = self.patch_size

        # bias patch selection towards regions with high gradient
        if centroid_sel_strat == 'GRADIENT_BIAS':
            g = self.__image_gradient(images.float32())
            x = jt.randint(1, w - 1, shape=(n, 3 * patches_per_image))
            y = jt.randint(1, h - 1, shape=(n, 3 * patches_per_image))

            coords = jt.stack([x, y], dim=-1).float32()
            g = altcorr.patchify(g[0, :, None], coords, 0).reshape(n, 3 * patches_per_image)

            ix, _ = jt.argsort(g, dim=1)
            x = jt.gather(x, 1, ix[:, -patches_per_image:])
            y = jt.gather(y, 1, ix[:, -patches_per_image:])

        elif centroid_sel_strat == 'RANDOM':
            x = jt.randint(1, w - 1, shape=(n, patches_per_image))
            y = jt.randint(1, h - 1, shape=(n, patches_per_image))

        else:
            raise NotImplementedError(f"Patch centroid selection not implemented: {centroid_sel_strat}")

        coords = jt.stack([x, y], dim=-1).float32()
        imap = altcorr.patchify(imap[0], coords, 0).reshape(b, -1, DIM, 1, 1)
        gmap = altcorr.patchify(fmap[0], coords, P // 2).reshape(b, -1, 128, P, P)

        if return_color:
            clr = altcorr.patchify(images[0].float32(), 4 * (coords + 0.5), 0).reshape(b, -1, 3)

        if disps is None:
            disps = jt.ones((b, n, h, w), dtype='float32')

        grid, _ = coords_grid_with_index(disps)
        patches = altcorr.patchify(grid[0], coords, P // 2).reshape(b, -1, 3, P, P)

        index = jt.arange(n).reshape(n, 1)
        index = index.repeat(1, patches_per_image).reshape(-1)

        if return_color:
            return fmap, gmap, imap, patches, index, clr

        return fmap, gmap, imap, patches, index


class CorrBlock:
    def __init__(self, fmap, gmap, radius=3, dropout=0.2, levels=[1, 4]):
        self.dropout = dropout
        self.radius = radius
        self.levels = levels

        self.gmap = gmap
        self.pyramid = pyramidify(fmap, lvls=levels)

    def __call__(self, ii, jj, coords):
        corrs = []
        for i in range(len(self.levels)):
            corrs += [altcorr.corr(self.gmap, self.pyramid[i], coords / self.levels[i], ii, jj, self.radius, self.dropout)]
        return jt.stack(corrs, -1).reshape(1, len(ii), -1)


class VONet(nn.Module):
    def __init__(self, use_viewer=False):
        super(VONet, self).__init__()
        self.P = 3
        self.patchify = Patchifier(self.P)
        self.update = Update(self.P)

        self.DIM = DIM
        self.RES = 4

    def execute(self, images, poses, disps, intrinsics, M=1024, STEPS=12, P=1, structure_only=False, rescale=False):
        """Estimates SE3 between pairs of frames (inference uses Patchifier and Update directly)."""
        raise NotImplementedError("VONet training forward is not supported in inference-only build")
