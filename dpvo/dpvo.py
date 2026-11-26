import logging

import numpy as np
import jittor as jt

from . import altcorr, fastba, se3
from . import projective_ops as pops
from .se3 import SE3
from .net import VONet
from .patchgraph import PatchGraph
from .utils import *


Id = SE3.Identity(1)


class DPVO:

    def __init__(self, cfg, network, ht=480, wd=640, viz=False):
        self.cfg = cfg
        backend = getattr(self.cfg, "BACKEND", "AUTO")
        if hasattr(altcorr, "set_backend"):
            altcorr.set_backend(backend)
        if hasattr(fastba, "set_backend"):
            fastba.set_backend(backend)

        self.huber_delta = float(cfg.BA_HUBER_DELTA)
        if not np.isfinite(self.huber_delta) or self.huber_delta < 0.0:
            raise ValueError("BA_HUBER_DELTA must be finite and nonnegative")
        self.relative_depth_beta = float(cfg.BA_RELATIVE_DEPTH_BETA)
        if not np.isfinite(self.relative_depth_beta) or self.relative_depth_beta < 0.0:
            raise ValueError("BA_RELATIVE_DEPTH_BETA must be finite and nonnegative")
        self._bootstrap_updates_complete = False
        self.load_weights(network)
        self.is_initialized = False
        self.enable_timing = False

        self.M = self.cfg.PATCHES_PER_FRAME
        self.N = self.cfg.BUFFER_SIZE

        self.ht = ht    # image height
        self.wd = wd    # image width

        DIM = self.DIM
        RES = self.RES

        ### state attributes ###
        self.tlist = []
        self.counter = 0
        self.hard_skip_since_retained = False
        self.hard_skip_recovery_remaining = 0
        self.numerical_events = []
        self._terminating = False

        # keep track of global-BA calls
        self.ran_global_ba = np.zeros(100000, dtype=bool)

        ht = ht // RES
        wd = wd // RES

        # dummy image placeholder
        self.image_ = jt.zeros((self.ht, self.wd, 3), dtype=jt.uint8)

        ### network attributes ###
        if self.cfg.MIXED_PRECISION:
            self.kwargs = kwargs = {"dtype": jt.float16}
        else:
            self.kwargs = kwargs = {"dtype": jt.float32}

        ### frame memory size ###
        self.pmem = self.mem = 36  # 32 was too small given default settings
        if self.cfg.LOOP_CLOSURE:
            self.last_global_ba = -1000  # keep track of time since last global opt
            self.pmem = self.cfg.MAX_EDGE_AGE  # patch memory

        self.imap_ = jt.zeros((self.pmem, self.M, DIM), **kwargs)
        self.gmap_ = jt.zeros((self.pmem, self.M, 128, self.P, self.P), **kwargs)

        self.pg = PatchGraph(self.cfg, self.P, self.DIM, self.pmem, **kwargs)
        self.frame_quality = jt.ones((self.N,), dtype=jt.float32)
        self.soft_frame_count = 0

        # classic backend
        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.load_long_term_loop_closure()

        self.fmap1_ = jt.zeros((1, self.mem, 128, ht // 1, wd // 1), **kwargs)
        self.fmap2_ = jt.zeros((1, self.mem, 128, ht // 4, wd // 4), **kwargs)

        # feature pyramid
        self.pyramid = (self.fmap1_, self.fmap2_)
        self.viewer = None

    def load_long_term_loop_closure(self):
        try:
            from .loop_closure.long_term import LongTermLoopClosure
            self.long_term_lc = LongTermLoopClosure(self.cfg, self.pg)
        except ModuleNotFoundError:
            self.cfg.CLASSIC_LOOP_CLOSURE = False

    def load_weights(self, network):
        # load network from checkpoint file
        if isinstance(network, str):
            from collections import OrderedDict
            state_dict = jt.load(network)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "update.lmbda" not in k:
                    new_state_dict[k.replace("module.", "")] = v

            self.network = VONet()
            self.network.load_state_dict(new_state_dict)

        else:
            self.network = network

        # steal network attributes
        self.DIM = self.network.DIM
        self.RES = self.network.RES
        self.P = self.network.P

        self.network.eval()

    @property
    def poses(self):
        return self.pg.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.pg.patches_.view(1, self.N * self.M, 3, self.P, self.P)

    @property
    def intrinsics(self):
        return self.pg.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.pg.index_.view(-1)

    @property
    def imap(self):
        return self.imap_.view(1, self.pmem * self.M, self.DIM)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.pmem * self.M, 128, self.P, self.P)

    @property
    def n(self):
        return self.pg.n

    @n.setter
    def n(self, val):
        self.pg.n = val

    @property
    def m(self):
        return self.pg.m

    @m.setter
    def m(self, val):
        self.pg.m = val

    def get_pose(self, t):
        # Long stationary prefixes can exceed Python's recursion limit.
        chain = []
        while t not in self.traj:
            t0, dP = self.pg.delta[t]
            chain.append((t, dP))
            t = t0
        pose = SE3(self.traj[t])
        for t, dP in reversed(chain):
            pose = dP * pose
            self.traj[t] = pose.data
        return pose

    def terminate(self):
        self._terminating = True

        if self.is_initialized:
            if self.cfg.CLASSIC_LOOP_CLOSURE:
                self.long_term_lc.terminate(self.n)

            if self.cfg.LOOP_CLOSURE:
                self.append_factors(*self.pg.edges_loop())

            for _ in range(12):
                self.ran_global_ba[self.n] = False
                if self.update() is False:
                    break

        """ interpolate missing poses """
        self.traj = {}
        for i in range(self.n):
            self.traj[self.pg.tstamps_[i]] = self.pg.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = se3.stack(poses, dim=0)
        poses = poses.inv().data.numpy()
        tstamps = np.array(self.tlist, dtype=np.float64)

        # Poses: x y z qx qy qz qw
        return poses, tstamps

    def corr(self, coords, indicies=None, fp32=False):
        """local correlation volume"""
        ii, jj = indicies if indicies is not None else (self.pg.kk, self.pg.jj)
        ii1 = ii % (self.M * self.pmem)
        jj1 = jj % (self.mem)
        gmap = self.gmap.float32() if fp32 else self.gmap
        pyramid = tuple(fmap.float32() for fmap in self.pyramid) if fp32 else self.pyramid
        corr1 = altcorr.corr(gmap, pyramid[0], coords / 1, ii1, jj1, 3)
        corr2 = altcorr.corr(gmap, pyramid[1], coords / 4, ii1, jj1, 3)
        return jt.stack([corr1, corr2], -1).view(1, ii.shape[0], -1)

    def reproject(self, indicies=None):
        """reproject patch k from i -> j"""
        (ii, jj, kk) = indicies if indicies is not None else (self.pg.ii, self.pg.jj, self.pg.kk)
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3)

    def append_factors(self, ii, jj):
        if ii.numel() == 0:
            return
        ii = ii.int64()
        jj = jj.int64()
        self.pg.jj = jt.concat([self.pg.jj, jj])
        self.pg.kk = jt.concat([self.pg.kk, ii])
        self.pg.ii = jt.concat([self.pg.ii, self.ix[ii]])

        net_dtype = self.pg.net.dtype if self.pg.net.shape[1] > 0 else self.kwargs["dtype"]
        net = jt.zeros((1, ii.shape[0], self.DIM), dtype=net_dtype)
        self.pg.net = jt.concat([self.pg.net, net], dim=1)

    def remove_factors(self, m, store: bool):
        assert self.pg.ii.numel() == self.pg.weight.shape[1]
        if self.pg.ii.numel() == 0:
            return
        m = m.bool()
        if store:
            removed_count = int(m.sum().item())
            if removed_count > 0:
                removed = jt.where(m)[0].int64()
                self.pg.ii_inac = jt.concat((self.pg.ii_inac, self.pg.ii[removed]))
                self.pg.jj_inac = jt.concat((self.pg.jj_inac, self.pg.jj[removed]))
                self.pg.kk_inac = jt.concat((self.pg.kk_inac, self.pg.kk[removed]))
                self.pg.weight_inac = jt.concat((self.pg.weight_inac, self.pg.weight[:, removed]), dim=1)
                self.pg.target_inac = jt.concat((self.pg.target_inac, self.pg.target[:, removed]), dim=1)

        keep_mask = jt.logical_not(m)
        keep_count = int(keep_mask.sum().item())
        if keep_count == 0:
            self.pg.weight = jt.zeros((1, 0, 2), dtype=self.pg.weight.dtype)
            self.pg.target = jt.zeros((1, 0, 2), dtype=self.pg.target.dtype)
            self.pg.ii = jt.zeros((0,), dtype=jt.int64)
            self.pg.jj = jt.zeros((0,), dtype=jt.int64)
            self.pg.kk = jt.zeros((0,), dtype=jt.int64)
            self.pg.net = jt.zeros((1, 0, self.DIM), dtype=self.pg.net.dtype)
        elif keep_count < m.shape[0]:
            keep = jt.where(keep_mask)[0].int64()
            self.pg.weight = self.pg.weight[:, keep]
            self.pg.target = self.pg.target[:, keep]
            self.pg.ii = self.pg.ii[keep]
            self.pg.jj = self.pg.jj[keep]
            self.pg.kk = self.pg.kk[keep]
            self.pg.net = self.pg.net[:, keep]
        assert self.pg.ii.numel() == self.pg.weight.shape[1]

    def motion_probe(self):
        """kinda hacky way to ensure enough motion for initialization"""
        kk = jt.arange(self.m - self.M, self.m, dtype=jt.int64)
        jj = jt.full(kk.shape, self.n, dtype=jt.int64)
        ii = self.ix[kk]

        net = jt.zeros((1, ii.shape[0], self.DIM), dtype=jt.float32)
        coords = self.reproject(indicies=(ii, jj, kk))

        corr = self.corr(coords, indicies=(kk, jj))
        ctx = self.imap[:, kk % (self.M * self.pmem)]
        net, (delta, weight, _) = self.network.update(
            net, ctx.float32(), corr.float32(), None, ii, jj, kk
        )

        delta_norm = jt.sqrt((delta.float32() * delta.float32()).sum(dim=-1)).reshape(-1)
        return float(quantile_05(delta_norm).item())

    def motionmag(self, i, j):
        k = (self.pg.ii == i) & (self.pg.jj == j)
        if int(k.sum().item()) == 0:
            return float("nan")
        index = jt.where(k)[0].int64()
        ii = self.pg.ii[index]
        jj = self.pg.jj[index]
        kk = self.pg.kk[index]

        flow, _ = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        return float(flow.mean().item())

    def keyframe(self):
        i = self.n - self.cfg.KEYFRAME_INDEX - 1
        j = self.n - self.cfg.KEYFRAME_INDEX + 1
        m = self.motionmag(i, j) + self.motionmag(j, i)

        if m / 2 < self.cfg.KEYFRAME_THRESH:
            k = self.n - self.cfg.KEYFRAME_INDEX
            t0 = self.pg.tstamps_[k - 1]
            t1 = self.pg.tstamps_[k]

            dP = SE3(self.pg.poses_[k]) * SE3(self.pg.poses_[k - 1]).inv()
            self.pg.delta[t1] = (t0, dP)

            to_remove = (self.pg.ii == k) | (self.pg.jj == k)
            self.remove_factors(to_remove, store=False)

            source_shift = (self.pg.ii > k).int64()
            self.pg.kk = self.pg.kk - source_shift * self.M
            self.pg.ii = self.pg.ii - source_shift
            self.pg.jj = self.pg.jj - (self.pg.jj > k).int64()

            for i in range(k, self.n - 1):
                self.pg.tstamps_[i] = self.pg.tstamps_[i + 1]
                self.pg.colors_[i] = self.pg.colors_[i + 1].clone()
                self.pg.poses_[i] = self.pg.poses_[i + 1].clone()
                self.pg.patches_[i] = self.pg.patches_[i + 1].clone()
                self.pg.intrinsics_[i] = self.pg.intrinsics_[i + 1].clone()

                self.imap_[i % self.pmem] = self.imap_[(i + 1) % self.pmem].clone()
                self.gmap_[i % self.pmem] = self.gmap_[(i + 1) % self.pmem].clone()
                self.fmap1_[:, i % self.mem] = self.fmap1_[:, (i + 1) % self.mem].clone()
                self.fmap2_[:, i % self.mem] = self.fmap2_[:, (i + 1) % self.mem].clone()

            self._remove_frame_quality(k)
            self.n -= 1
            self.m -= self.M

            if self.cfg.CLASSIC_LOOP_CLOSURE:
                self.long_term_lc.keyframe(k)

        if self.pg.kk.numel() > 0:
            to_remove = self.ix[self.pg.kk] < self.n - self.cfg.REMOVAL_WINDOW
            if self.cfg.LOOP_CLOSURE:
                # ...unless they are being used for loop closure
                lc_edges = ((self.pg.jj - self.pg.ii) > 30) & (
                    self.pg.jj > (self.n - self.cfg.OPTIMIZATION_WINDOW)
                )
                to_remove = to_remove & jt.logical_not(lc_edges)
            self.remove_factors(to_remove, store=True)

    def __run_global_BA(self):
        """Global bundle adjustment. Includes both active and inactive edges."""
        if self.pg.ii_inac.numel() > 0:
            full_target = jt.concat((self.pg.target_inac, self.pg.target), dim=1)
            full_weight = jt.concat((self.pg.weight_inac, self.pg.weight), dim=1)
            full_ii = jt.concat((self.pg.ii_inac, self.pg.ii))
            full_jj = jt.concat((self.pg.jj_inac, self.pg.jj))
            full_kk = jt.concat((self.pg.kk_inac, self.pg.kk))
        else:
            full_target = self.pg.target
            full_weight = self.pg.weight
            full_ii = self.pg.ii
            full_jj = self.pg.jj
            full_kk = self.pg.kk

        self.pg.normalize()
        lmbda = jt.array([1e-4], dtype=jt.float32)
        t0 = int(self.pg.ii.min().item())
        fastba.BA(
            self.pg.poses_,
            self.pg.patches_,
            self.intrinsics,
            full_target,
            full_weight,
            lmbda,
            full_ii,
            full_jj,
            full_kk,
            t0,
            self.n,
            M=self.M,
            iterations=2,
            eff_impl=True,
            huber_delta=self.huber_delta if self._bootstrap_updates_complete else 0.0,
            relative_depth_beta=self.relative_depth_beta if self._bootstrap_updates_complete else 0.0,
        )
        self.ran_global_ba[self.n] = True

    def update(self):
        if self.pg.ii.numel() == 0:
            return False
        with Timer("other", enabled=self.enable_timing):
            coords = self.reproject()
            if not bool(jt.isfinite(coords).all().item()):
                self._reject_update("projection", "non-finite projected coordinates")
                return False

            corr = self.corr(coords)
            ctx = self.imap[:, self.pg.kk % (self.M * self.pmem)]
            net, (delta, weight, _) = self.network.update(
                self.pg.net.float32(), ctx.float32(), corr.float32(), None,
                self.pg.ii, self.pg.jj, self.pg.kk,
            )

            retried_fp32 = False
            if not self._finite_network_output(net, delta, weight):
                retried_fp32 = True
                self._record_numerical_event("network", "retrying non-finite update in float32")
                if not bool(jt.isfinite(corr).all().item()):
                    corr = self.corr(coords, fp32=True)
                net, (delta, weight, _) = self.network.update(
                    self.pg.net.float32(), ctx.float32(), corr.float32(), None,
                    self.pg.ii, self.pg.jj, self.pg.kk,
                )
                if not self._finite_network_output(net, delta, weight):
                    self._reject_update("network", "float32 update is non-finite")
                    return False

            weight = weight.float32()
            weight = self._apply_frame_quality(weight, self.pg.ii, self.pg.jj)
            target = coords[..., self.P // 2, self.P // 2] + delta.float32()
            if not bool(jt.isfinite(target).all().item()):
                self._reject_update("network", "non-finite target coordinates")
                return False

        if retried_fp32 and self.cfg.MIXED_PRECISION:
            net_half = net.float16()
            self.pg.net = net_half if bool(jt.isfinite(net_half).all().item()) else net.float32()
        else:
            self.pg.net = net.cast(self.kwargs["dtype"])
        self.pg.target = target
        self.pg.weight = weight

        with Timer("BA", enabled=self.enable_timing):
            accepted = self._bundle_adjust()

            points = pops.point_cloud(
                SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m]
            )
            center = self.P // 2
            points = (points[..., center, center, :3] / points[..., center, center, 3:]).reshape(-1, 3)
            self.pg.points_[:points.shape[0]] = points[:]
        return accepted

    @staticmethod
    def _finite_network_output(net, delta, weight):
        return all(bool(jt.isfinite(value).all().item()) for value in (net, delta, weight))

    @staticmethod
    def _valid_poses(poses):
        norms = jt.sqrt((poses[..., 3:7] * poses[..., 3:7]).sum(dim=-1))
        return bool(
            jt.isfinite(poses).all().item()
            and jt.isfinite(norms).all().item()
            and bool((norms > 1e-12).all().item())
        )

    def _record_numerical_event(self, stage, reason, tstamp=None):
        if not hasattr(self, "numerical_events"):
            self.numerical_events = []
        if getattr(self, "_terminating", False):
            stage = "terminate/" + stage
        frame = self.tlist[-1] if tstamp is None else tstamp
        event = {"frame": int(frame), "stage": stage, "reason": reason}
        self.numerical_events.append(event)
        if len(self.numerical_events) <= 20:
            logging.error("DPVO numerical recovery: %s", event)

    def _reject_update(self, stage, reason):
        # Keep edge arrays aligned without storing invalid constraints for global BA.
        self.pg.target = jt.zeros((1, self.pg.ii.numel(), 2), dtype=jt.float32)
        self.pg.weight = jt.zeros_like(self.pg.target)
        self._record_numerical_event(stage, reason)

    def _bundle_adjust(self):
        global_ba = bool(
            (self.pg.ii < self.n - self.cfg.REMOVAL_WINDOW - 1).any().item()
        ) and not self.ran_global_ba[self.n]
        old_poses = self.pg.poses_[:self.n].clone()
        old_depths = self.pg.patches_[:self.n, :, 2].clone()
        # normalize() also rescales removed-frame transforms; restore them together.
        old_delta = self.pg.delta.copy() if global_ba else None
        old_points = self.pg.points_[:self.m].clone() if global_ba else None
        old_ran_global_ba = self.ran_global_ba[self.n]
        try:
            if global_ba:
                self.__run_global_BA()
            else:
                t0 = max(self.n - self.cfg.OPTIMIZATION_WINDOW, 1) if self.is_initialized else 1
                lmbda = jt.array([1e-4], dtype=jt.float32)
                fastba.BA(
                    self.pg.poses_,
                    self.pg.patches_,
                    self.intrinsics,
                    self.pg.target,
                    self.pg.weight,
                    lmbda,
                    self.pg.ii,
                    self.pg.jj,
                    self.pg.kk,
                    t0,
                    self.n,
                    M=self.M,
                    iterations=2,
                    eff_impl=self.cfg.EFF_IMPL,
                    huber_delta=self.huber_delta if self._bootstrap_updates_complete else 0.0,
                    relative_depth_beta=self.relative_depth_beta if self._bootstrap_updates_complete else 0.0,
                )
            depths = self.pg.patches_[:self.n, :, 2]
            if not (
                self._valid_poses(self.pg.poses_[:self.n])
                and bool(jt.isfinite(depths).all().item())
                and bool((depths > 0).all().item())
            ):
                raise RuntimeError("DPVO numerical: invalid pose or depth after BA")
        except (RuntimeError, np.linalg.LinAlgError) as exc:
            if "DPVO numerical:" not in str(exc) and not isinstance(exc, np.linalg.LinAlgError):
                raise
            self.pg.poses_[:self.n] = old_poses
            self.pg.patches_[:self.n, :, 2] = old_depths
            if old_delta is not None:
                self.pg.delta = old_delta
                self.pg.points_[:self.m] = old_points
            self.ran_global_ba[self.n] = old_ran_global_ba
            self._record_numerical_event(
                "global-BA" if global_ba else "local-BA", str(exc).splitlines()[0]
            )
            return False
        return True

    def _apply_frame_quality(self, weight, ii, jj):
        """Downweight edges incident to retained soft-quality frames."""
        if self.soft_frame_count == 0:
            return weight

        source_quality = self.frame_quality[ii]
        target_quality = self.frame_quality[jj]
        edge_quality = jt.minimum(source_quality, target_quality).cast(weight.dtype)
        return weight * edge_quality.view(1, -1, 1)

    def _remove_frame_quality(self, k):
        """Keep graph-frame quality aligned after keyframe slot compaction."""
        if self.soft_frame_count == 0:
            return

        removed_was_soft = bool(self.frame_quality[k].item() < 1.0)
        for i in range(k, self.n - 1):
            self.frame_quality[i] = self.frame_quality[i + 1].clone()
        self.frame_quality[self.n - 1] = 1.0
        if removed_was_soft:
            self.soft_frame_count -= 1

    def __edges_forw(self):
        r = self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        if t0 >= t1 or (self.n - 1) >= self.n:
            return jt.zeros((0,), dtype=jt.int64), jt.zeros((0,), dtype=jt.int64)
        return flatmeshgrid(
            jt.arange(t0, t1, dtype=jt.int64),
            jt.arange(self.n - 1, self.n, dtype=jt.int64),
            indexing="ij",
        )

    def __edges_back(self):
        r = self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        j0 = max(self.n - r, 0)
        if t0 >= t1 or j0 >= self.n:
            return jt.zeros((0,), dtype=jt.int64), jt.zeros((0,), dtype=jt.int64)
        return flatmeshgrid(
            jt.arange(t0, t1, dtype=jt.int64),
            jt.arange(j0, self.n, dtype=jt.int64),
            indexing="ij",
        )

    def _motion_delta(self, tstamp):
        """Predict motion relative to the latest retained graph frame."""
        if self.n < 2 or self.cfg.MOTION_MODEL != "DAMPED_LINEAR":
            return Id[0]

        i1 = int(self.pg.tstamps_[self.n - 1])
        i2 = int(self.pg.tstamps_[self.n - 2])
        try:
            dt_prev = float(self.tlist[i1]) - float(self.tlist[i2])
            dt_cur = float(tstamp) - float(self.tlist[i1])
        except (IndexError, TypeError, ValueError):
            return Id[0]

        if (
            not np.isfinite(dt_prev)
            or not np.isfinite(dt_cur)
            or dt_prev <= 0.0
            or dt_cur < 0.0
        ):
            return Id[0]

        P1 = SE3(self.pg.poses_[self.n - 1])
        P2 = SE3(self.pg.poses_[self.n - 2])
        xi = self.cfg.MOTION_DAMPING * (dt_cur / dt_prev) * (P1 * P2.inv()).log()
        return self._checked_motion(SE3.exp(xi), tstamp=tstamp)

    def _v1_motion_delta(self):
        """Original DPVO prediction for an ordinary accepted-frame prefix."""
        P1 = SE3(self.pg.poses_[self.n - 1])
        P2 = SE3(self.pg.poses_[self.n - 2])

        # Preserve the original implementation's timestamp path exactly when
        # no hard-skipped frame needs bridging.
        *_, a, b, c = [1] * 3 + self.tlist
        fac = (c - b) / (b - a)
        xi = self.cfg.MOTION_DAMPING * fac * (P1 * P2.inv()).log()
        return self._checked_motion(SE3.exp(xi))

    def _checked_motion(self, motion, tstamp=None):
        if not self._valid_poses(motion.data):
            self._record_numerical_event(
                "motion", "invalid extrapolation; keeping retained pose", tstamp=tstamp
            )
            return Id[0]
        return motion

    def _accepted_motion_delta(self, tstamp):
        """Use retained timestamps until skipped inputs leave the v1 triplet."""
        if self.hard_skip_recovery_remaining > 0:
            dP = self._motion_delta(tstamp)
            self.hard_skip_recovery_remaining -= 1
        else:
            dP = self._v1_motion_delta()
        self.hard_skip_since_retained = self.hard_skip_recovery_remaining > 0
        return dP

    def skip_frame(self, tstamp):
        """Record a frame without adding it to the optimization graph."""
        if self.n == 0:
            raise RuntimeError("cannot skip a frame before DPVO has retained a pose")

        root = int(self.pg.tstamps_[self.n - 1])
        dP = self._motion_delta(tstamp)
        self.tlist.append(tstamp)
        self.pg.delta[self.counter] = (root, dP)
        self.counter += 1
        self.hard_skip_since_retained = True
        self.hard_skip_recovery_remaining = 2

    def __call__(self, tstamp, image, intrinsics, frame_weight=1.0):
        """track new frame"""

        frame_weight = float(frame_weight)
        if not np.isfinite(frame_weight) or not 0.0 < frame_weight <= 1.0:
            raise ValueError(
                "frame_weight must be finite and in (0, 1], "
                f"got {frame_weight!r}"
            )

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc(image, self.n)

        if (self.n + 1) >= self.N:
            raise Exception(
                f'The buffer size is too small. You can increase it using "--opts BUFFER_SIZE={self.N*2}"'
            )

        image = 2 * (image[None, None].float32() / 255.0) - 0.5

        fmap, gmap, imap, patches, _, clr = self.network.patchify(
            image,
            patches_per_image=self.cfg.PATCHES_PER_FRAME,
            centroid_sel_strat=self.cfg.CENTROID_SEL_STRAT,
            return_color=True,
        )

        ### update state attributes ###
        self.tlist.append(tstamp)
        self.pg.tstamps_[self.n] = self.counter
        self.pg.intrinsics_[self.n] = intrinsics.float32() / self.RES

        # color info for visualization
        clr = (clr[0, :, [2, 1, 0]] + 0.5) * (255.0 / 2)
        self.pg.colors_[self.n] = clr.uint8()

        self.pg.index_[self.n + 1] = self.n + 1
        self.pg.index_map_[self.n + 1] = self.m + self.M
        self.frame_quality[self.n] = frame_weight

        if self.n > 1:
            P1 = SE3(self.pg.poses_[self.n - 1])
            if self.cfg.MOTION_MODEL == "DAMPED_LINEAR":
                dP = self._accepted_motion_delta(tstamp)
                self.pg.poses_[self.n] = (dP * P1).data
                if not self._valid_poses(self.pg.poses_[self.n]):
                    self._record_numerical_event(
                        "motion", "invalid composition; keeping retained pose"
                    )
                    self.pg.poses_[self.n] = P1.data
            else:
                self.pg.poses_[self.n] = self.pg.poses_[self.n - 1].clone()
                self.hard_skip_since_retained = False
                self.hard_skip_recovery_remaining = 0
        else:
            self.hard_skip_since_retained = False
            self.hard_skip_recovery_remaining = 0

        initial_depth = jt.rand((1, self.M, 1, 1), dtype=jt.float32)
        initial_depth = jt.where(
            initial_depth == 0, jt.full_like(initial_depth, 1e-4), initial_depth
        )
        patches = patches.float32().clone()
        patches[:, :, 2] = initial_depth.broadcast((1, self.M, self.P, self.P))
        if self.is_initialized:
            s = float(jt.median(self.pg.patches_[self.n - 3:self.n, :, 2]).item())
            patches[:, :, 2] = s

        self.pg.patches_[self.n] = patches[0]

        ### update network attributes ###
        self.imap_[self.n % self.pmem] = imap.reshape(self.M, self.DIM).cast(self.imap_.dtype)
        self.gmap_[self.n % self.pmem] = gmap.reshape(self.M, 128, self.P, self.P).cast(self.gmap_.dtype)
        self.fmap1_[:, self.n % self.mem] = jt.nn.avg_pool2d(fmap[0], 1, stride=1).cast(self.fmap1_.dtype)
        self.fmap2_[:, self.n % self.mem] = jt.nn.avg_pool2d(fmap[0], 4, stride=4).cast(self.fmap2_.dtype)

        self.counter += 1
        if self.n > 0 and not self.is_initialized:
            if self.motion_probe() < 2.0:
                self.pg.delta[self.counter - 1] = (self.counter - 2, Id[0])
                self.frame_quality[self.n] = 1.0
                return

        self.n += 1
        self.m += self.M
        if frame_weight < 1.0:
            self.soft_frame_count += 1

        if self.cfg.LOOP_CLOSURE:
            if self.n - self.last_global_ba >= self.cfg.GLOBAL_OPT_FREQ:
                """ Add loop closure factors """
                lii, ljj = self.pg.edges_loop()
                if lii.numel() > 0:
                    self.last_global_ba = self.n
                    self.append_factors(lii, ljj)

        # Add forward and backward factors
        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())

        if self.n == 8 and not self.is_initialized:
            self.is_initialized = True

            for itr in range(12):
                self.update()
            self._bootstrap_updates_complete = True

        elif self.is_initialized:
            self.update()
            self.keyframe()

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc.attempt_loop_closure(self.n)
            self.long_term_lc.lc_callback()
