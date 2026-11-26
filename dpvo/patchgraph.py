import numpy as np
import jittor as jt

from . import projective_ops as pops
from .se3 import SE3
from .loop_closure.optim_utils import reduce_edges
from .utils import *


class PatchGraph:
    """Dataclass for storing variables."""

    def __init__(self, cfg, P, DIM, pmem, **kwargs):
        self.cfg = cfg
        self.P = P
        self.pmem = pmem
        self.DIM = DIM

        self.n = 0      # number of frames
        self.m = 0      # number of patches

        self.M = self.cfg.PATCHES_PER_FRAME
        self.N = self.cfg.BUFFER_SIZE

        self.tstamps_ = np.zeros(self.N, dtype=np.int64)
        self.poses_ = jt.zeros((self.N, 7), dtype=jt.float32)
        self.patches_ = jt.zeros((self.N, self.M, 3, self.P, self.P), dtype=jt.float32)
        self.intrinsics_ = jt.zeros((self.N, 4), dtype=jt.float32)

        self.points_ = jt.zeros((self.N * self.M, 3), dtype=jt.float32)
        self.colors_ = jt.zeros((self.N, self.M, 3), dtype=jt.uint8)

        self.index_ = jt.zeros((self.N, self.M), dtype=jt.int64)
        self.index_map_ = jt.zeros((self.N,), dtype=jt.int64)

        # initialize poses to identity matrix
        self.poses_[:, 6] = 1.0

        # store relative poses for removed frames
        self.delta = {}

        ### edge information ###
        self.net = jt.zeros((1, 0, DIM), **kwargs)
        self.ii = jt.zeros((0,), dtype=jt.int64)
        self.jj = jt.zeros((0,), dtype=jt.int64)
        self.kk = jt.zeros((0,), dtype=jt.int64)

        ### inactive edge information (i.e., no longer updated, but useful for BA) ###
        self.ii_inac = jt.zeros((0,), dtype=jt.int64)
        self.jj_inac = jt.zeros((0,), dtype=jt.int64)
        self.kk_inac = jt.zeros((0,), dtype=jt.int64)
        self.weight_inac = jt.zeros((1, 0, 2), dtype=jt.float32)
        self.target_inac = jt.zeros((1, 0, 2), dtype=jt.float32)

    def edges_loop(self):
        """Adding edges from old patches to new frames."""
        lc_range = self.cfg.MAX_EDGE_AGE
        l = self.n - self.cfg.REMOVAL_WINDOW  # upper bound for "old" patches

        if l <= 0:
            return jt.zeros((0,), dtype=jt.int64), jt.zeros((0,), dtype=jt.int64)

        j_start = self.n - self.cfg.GLOBAL_OPT_FREQ
        j_end = self.n - self.cfg.KEYFRAME_INDEX
        k_start = max(l - lc_range, 0) * self.M
        k_end = l * self.M
        if j_start >= j_end or k_start >= k_end:
            return jt.zeros((0,), dtype=jt.int64), jt.zeros((0,), dtype=jt.int64)

        # create candidate edges
        jj, kk = flatmeshgrid(
            jt.arange(j_start, j_end, dtype=jt.int64),
            jt.arange(k_start, k_end, dtype=jt.int64),
            indexing="ij",
        )
        if jj.numel() == 0 or kk.numel() == 0:
            return jt.zeros((0,), dtype=jt.int64), jt.zeros((0,), dtype=jt.int64)
        ii = self.ix[kk]

        # Remove edges which have too large flow magnitude
        center = self.P // 2
        flow_mg, val = pops.flow_mag(
            SE3(self.poses),
            self.patches[..., center, center].view(1, -1, 3, 1, 1),
            self.intrinsics,
            ii,
            jj,
            kk,
            beta=0.5,
        )
        fl = flow_mg.shape[1] // self.M
        flow_mg_sum = (flow_mg * val).reshape(fl, self.M).sum(dim=1).float32()
        num_val = val.reshape(fl, self.M).sum(dim=1).float32().maximum(1.0)
        flow_mag = jt.where(
            num_val > (self.M * 0.75),
            flow_mg_sum / num_val,
            jt.full_like(flow_mg_sum, float("inf")),
        )

        mask = flow_mag < self.cfg.BACKEND_THRESH
        if int(mask.sum().item()) == 0:
            return jt.zeros((0,), dtype=jt.int64), jt.zeros((0,), dtype=jt.int64)

        mask_np = mask.numpy().astype(bool)
        es = reduce_edges(
            flow_mag.numpy()[mask_np],
            ii.numpy()[::self.M][mask_np],
            jj.numpy()[::self.M][mask_np],
            max_num_edges=1000,
            nms=1,
        )
        if len(es) == 0:
            return jt.zeros((0,), dtype=jt.int64), jt.zeros((0,), dtype=jt.int64)

        edges = jt.array(es, dtype=jt.int64)
        ii_e = edges[:, 0:1].repeat(1, self.M)
        jj_e = edges[:, 1:2].repeat(1, self.M)
        kk_e = ii_e * self.M + jt.arange(self.M, dtype=jt.int64).unsqueeze(0)
        return kk_e.reshape(-1), jj_e.reshape(-1)

    def normalize(self):
        """Normalize depth and poses."""
        s = self.patches_[:self.n, :, 2].mean()
        s_val = float(s.item())
        if not np.isfinite(s_val) or s_val <= 0:
            raise RuntimeError("DPVO numerical: invalid depth normalization scale")
        self.patches_[:self.n, :, 2] = self.patches_[:self.n, :, 2] / s_val
        self.poses_[:self.n, :3] = self.poses_[:self.n, :3] * s_val
        scaled_delta = {t: (t0, dP.scale(s_val)) for t, (t0, dP) in self.delta.items()}
        if scaled_delta:
            stacked = jt.stack([dP.data for _, dP in scaled_delta.values()])
            if not bool(jt.isfinite(stacked).all().item()):
                raise RuntimeError("DPVO numerical: invalid removed-frame scale after normalization")
        self.delta = scaled_delta
        self.poses_[:self.n] = (SE3(self.poses_[:self.n]) * SE3(self.poses_[:1]).inv()).data

        points = pops.point_cloud(
            SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m]
        )
        center = self.P // 2
        points = (points[..., center, center, :3] / points[..., center, center, 3:]).reshape(-1, 3)
        self.points_[:points.shape[0]] = points[:]

    @property
    def poses(self):
        return self.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.patches_.view(1, self.N * self.M, 3, self.P, self.P)

    @property
    def intrinsics(self):
        return self.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.index_.view(-1)
