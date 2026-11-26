import jittor as jt
import numpy as np
import scipy.linalg

_NEIGHBORS_CPU_HEADER = """
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>
"""

_NEIGHBORS_CPU_SRC = """
const int64_t E = in0_shape0;
std::vector<int64_t> order(E);
std::iota(order.begin(), order.end(), 0);
std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
    const int64_t ka = @in0(a);
    const int64_t kb = @in0(b);
    if (ka != kb) return ka < kb;
    return @in1(a) < @in1(b);
});
for (int64_t p = 0; p < E; p++) {
    const int64_t cur = order[p];
    const int64_t k_cur = @in0(cur);
    @out0(cur) = (p > 0 && @in0(order[p - 1]) == k_cur) ? order[p - 1] : -1;
    @out1(cur) = (p + 1 < E && @in0(order[p + 1]) == k_cur) ? order[p + 1] : -1;
}
"""

_INDEX_TENSOR_CPU_SRC = """
const int64_t U = in0_shape0;
const int64_t n_frames = (int64_t)data["n_frames"];
int64_t cx = 0;
int64_t start = 0;
while (start < U) {
    const int64_t i = @in0(start) / n_frames;
    int64_t end = start + 1;
    while (end < U && (@in0(end) / n_frames) == i) {
        end++;
    }
    for (int64_t a = start; a < end; a++) {
        const int64_t j1 = @in0(a) % n_frames;
        for (int64_t b = start; b < end; b++) {
            const int64_t j2 = @in0(b) % n_frames;
            @out0(cx, 0) = i;
            @out0(cx, 1) = j1;
            @out0(cx, 2) = j2;
            @out0(cx, 3) = a;
            @out0(cx, 4) = b;
            cx++;
        }
    }
    start = end;
}
"""

_COMMON_SE3_CUDA_HEADER = """
#include <math.h>
#include <stdint.h>

#ifndef DPVO_GPU_1D_KERNEL_LOOP
#define DPVO_GPU_1D_KERNEL_LOOP(i, n) \\
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < (int64_t)(n); i += (int64_t)blockDim.x * gridDim.x)
#endif

__device__ static inline void dpvo_actSO3(const float *q, const float *X, float *Y) {
  float uv[3];
  uv[0] = 2.0f * (q[1] * X[2] - q[2] * X[1]);
  uv[1] = 2.0f * (q[2] * X[0] - q[0] * X[2]);
  uv[2] = 2.0f * (q[0] * X[1] - q[1] * X[0]);

  Y[0] = X[0] + q[3] * uv[0] + (q[1] * uv[2] - q[2] * uv[1]);
  Y[1] = X[1] + q[3] * uv[1] + (q[2] * uv[0] - q[0] * uv[2]);
  Y[2] = X[2] + q[3] * uv[2] + (q[0] * uv[1] - q[1] * uv[0]);
}

__device__ static inline void dpvo_actSE3(const float *t, const float *q, const float *X, float *Y) {
  dpvo_actSO3(q, X, Y);
  Y[3] = X[3];
  Y[0] += X[3] * t[0];
  Y[1] += X[3] * t[1];
  Y[2] += X[3] * t[2];
}

__device__ static inline void dpvo_adjSE3(const float *t, const float *q, const float *X, float *Y) {
  float qinv[4] = {-q[0], -q[1], -q[2], q[3]};
  dpvo_actSO3(qinv, &X[0], &Y[0]);
  dpvo_actSO3(qinv, &X[3], &Y[3]);

  float u[3], v[3];
  u[0] = t[2] * X[1] - t[1] * X[2];
  u[1] = t[0] * X[2] - t[2] * X[0];
  u[2] = t[1] * X[0] - t[0] * X[1];

  dpvo_actSO3(qinv, u, v);
  Y[3] += v[0];
  Y[4] += v[1];
  Y[5] += v[2];
}

__device__ static inline void dpvo_relSE3(
    const float *ti, const float *qi,
    const float *tj, const float *qj,
    float *tij, float *qij) {
  qij[0] = -qj[3] * qi[0] + qj[0] * qi[3] - qj[1] * qi[2] + qj[2] * qi[1];
  qij[1] = -qj[3] * qi[1] + qj[1] * qi[3] - qj[2] * qi[0] + qj[0] * qi[2];
  qij[2] = -qj[3] * qi[2] + qj[2] * qi[3] - qj[0] * qi[1] + qj[1] * qi[0];
  qij[3] =  qj[3] * qi[3] + qj[0] * qi[0] + qj[1] * qi[1] + qj[2] * qi[2];

  dpvo_actSO3(qij, ti, tij);
  tij[0] = tj[0] - tij[0];
  tij[1] = tj[1] - tij[1];
  tij[2] = tj[2] - tij[2];
}

__device__ static inline void dpvo_expSO3(const float *phi, float *q) {
  float theta_sq = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2];
  float theta_p4 = theta_sq * theta_sq;
  float theta = sqrtf(theta_sq);
  float imag, real;

  if (theta_sq < 1e-8f) {
    imag = 0.5f - (1.0f / 48.0f) * theta_sq + (1.0f / 3840.0f) * theta_p4;
    real = 1.0f - (1.0f / 8.0f) * theta_sq + (1.0f / 384.0f) * theta_p4;
  } else {
    imag = sinf(0.5f * theta) / theta;
    real = cosf(0.5f * theta);
  }

  q[0] = imag * phi[0];
  q[1] = imag * phi[1];
  q[2] = imag * phi[2];
  q[3] = real;
}

__device__ static inline void dpvo_crossInplace(const float *a, float *b) {
  float x[3] = {
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]
  };
  b[0] = x[0];
  b[1] = x[1];
  b[2] = x[2];
}

__device__ static inline void dpvo_expSE3(const float *xi, float *t, float *q) {
  dpvo_expSO3(xi + 3, q);
  float tau[3] = {xi[0], xi[1], xi[2]};
  float phi[3] = {xi[3], xi[4], xi[5]};

  float theta_sq = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2];
  float theta = sqrtf(theta_sq);

  t[0] = tau[0];
  t[1] = tau[1];
  t[2] = tau[2];

  if (theta > 1e-4f) {
    float a = (1.0f - cosf(theta)) / theta_sq;
    dpvo_crossInplace(phi, tau);
    t[0] += a * tau[0];
    t[1] += a * tau[1];
    t[2] += a * tau[2];

    float b = (theta - sinf(theta)) / (theta * theta_sq);
    dpvo_crossInplace(phi, tau);
    t[0] += b * tau[0];
    t[1] += b * tau[1];
    t[2] += b * tau[2];
  }
}

__device__ static inline void dpvo_retrSE3(
    const float *xi, const float *t, const float *q, float *t1, float *q1) {
  float dt[3] = {0.0f, 0.0f, 0.0f};
  float dq[4] = {0.0f, 0.0f, 0.0f, 1.0f};
  dpvo_expSE3(xi, dt, dq);

  q1[0] = dq[3] * q[0] + dq[0] * q[3] + dq[1] * q[2] - dq[2] * q[1];
  q1[1] = dq[3] * q[1] + dq[1] * q[3] + dq[2] * q[0] - dq[0] * q[2];
  q1[2] = dq[3] * q[2] + dq[2] * q[3] + dq[0] * q[1] - dq[1] * q[0];
  q1[3] = dq[3] * q[3] - dq[0] * q[0] - dq[1] * q[1] - dq[2] * q[2];

  dpvo_actSO3(dq, t, t1);
  t1[0] += dt[0];
  t1[1] += dt[1];
  t1[2] += dt[2];
}
"""

_REPROJECT_CUDA_SRC = """
__global__ static void reproject_kernel(@ARGS_DEF) {
  @PRECALC
  __shared__ float fx, fy, cx, cy;
  if (threadIdx.x == 0) {
    fx = @in2(0, 0);
    fy = @in2(0, 1);
    cx = @in2(0, 2);
    cy = @in2(0, 3);
  }
  __syncthreads();

  const int64_t E = in3_shape0;
  const int P_h = in1_shape2;
  const int P_w = in1_shape3;

  DPVO_GPU_1D_KERNEL_LOOP(n, E) {
    const int64_t ix = @in3(n);
    const int64_t jx = @in4(n);
    const int64_t kx = @in5(n);

    float ti[3] = { @in0(ix, 0), @in0(ix, 1), @in0(ix, 2) };
    float tj[3] = { @in0(jx, 0), @in0(jx, 1), @in0(jx, 2) };
    float qi[4] = { @in0(ix, 3), @in0(ix, 4), @in0(ix, 5), @in0(ix, 6) };
    float qj[4] = { @in0(jx, 3), @in0(jx, 4), @in0(jx, 5), @in0(jx, 6) };

    float tij[3], qij[4];
    dpvo_relSE3(ti, qi, tj, qj, tij, qij);

    float Xi[4], Xj[4];
    for (int i = 0; i < P_h; i++) {
      for (int j = 0; j < P_w; j++) {
        Xi[0] = (@in1(kx, 0, i, j) - cx) / fx;
        Xi[1] = (@in1(kx, 1, i, j) - cy) / fy;
        Xi[2] = 1.0f;
        Xi[3] = @in1(kx, 2, i, j);

        dpvo_actSE3(tij, qij, Xi, Xj);

        @out(n, 0, i, j) = fx * (Xj[0] / Xj[2]) + cx;
        @out(n, 1, i, j) = fy * (Xj[1] / Xj[2]) + cy;
      }
    }
  }
}

const int64_t E = in3_shape0;
const int threads = 256;
int blocks = (int)((E + threads - 1) / threads);
if (blocks > 65535) blocks = 65535;
if (blocks > 0) {
  reproject_kernel<<<blocks, threads>>>(@ARGS);
}
"""

_HESSIAN_CUDA_SRC = """
__global__ static void reprojection_residuals_and_hessian_kernel(
    @ARGS_DEF, const int64_t t0, const int64_t N, const int ppf, const float huber_delta) {
  @PRECALC
  __shared__ float fx, fy, cx, cy;
  if (threadIdx.x == 0) {
    fx = @in2(0, 0);
    fy = @in2(0, 1);
    cx = @in2(0, 2);
    cy = @in2(0, 3);
  }
  const bool eff_impl = (ppf > 0);
  __syncthreads();

  const int64_t E_edges = in6_shape0;
  DPVO_GPU_1D_KERNEL_LOOP(n, E_edges) {
    const int64_t k  = @in9(n);
    int64_t ix = @in6(n);
    int64_t jx = @in7(n);
    const int64_t kx = @in8(n);

    int64_t ijx = 0, ijs = 0;
    if (eff_impl) {
      ijx = @in5(0, n);
      ijs = @in5(1, n);
    }

    float ti[3] = { @in0(ix, 0), @in0(ix, 1), @in0(ix, 2) };
    float tj[3] = { @in0(jx, 0), @in0(jx, 1), @in0(jx, 2) };
    float qi[4] = { @in0(ix, 3), @in0(ix, 4), @in0(ix, 5), @in0(ix, 6) };
    float qj[4] = { @in0(jx, 3), @in0(jx, 4), @in0(jx, 5), @in0(jx, 6) };

    float Xi[4], Xj[4];
    Xi[0] = (@in1(kx, 0, 1, 1) - cx) / fx;
    Xi[1] = (@in1(kx, 1, 1, 1) - cy) / fy;
    Xi[2] = 1.0f;
    Xi[3] = @in1(kx, 2, 1, 1);

    float tij[3], qij[4];
    dpvo_relSE3(ti, qi, tj, qj, tij, qij);
    dpvo_actSE3(tij, qij, Xi, Xj);

    const float X = Xj[0];
    const float Y = Xj[1];
    const float Z = Xj[2];
    const float W = Xj[3];

    const float tx = @in3(n, 0);
    const float ty = @in3(n, 1);
    const float wx = @in4(n, 0);
    const float wy = @in4(n, 1);

    if (!isfinite(X) || !isfinite(Y) || !isfinite(Z) || !isfinite(W) || Z <= 0.2f ||
        !isfinite(tx) || !isfinite(ty) ||
        !isfinite(wx) || !isfinite(wy) ||
        wx < 0.0f || wy < 0.0f) {
      continue;
    }

    const float d = 1.0f / Z;
    const float d2 = d * d;

    const float x1 = fx * (X * d) + cx;
    const float y1 = fy * (Y * d) + cy;

    const float rx = tx - x1;
    const float ry = ty - y1;

    const bool in_bounds = (sqrtf(rx * rx + ry * ry) < 128.0f) && (Z > 0.2f) &&
      (x1 > -64.0f) && (y1 > -64.0f) && (x1 < 2.0f * cx + 64.0f) && (y1 < 2.0f * cy + 64.0f);

    if (!in_bounds) {
      continue;
    }

    float robust_weight = 1.0f;
    if (huber_delta > 0.0f) {
      const float mean_weight = 0.5f * (wx + wy);
      if (mean_weight > 0.0f) {
        const float scaled_residual_sq = (wx * rx * rx + wy * ry * ry) / mean_weight;
        if (scaled_residual_sq > huber_delta * huber_delta) {
          robust_weight = huber_delta / sqrtf(scaled_residual_sq);
        }
      }
    }

    ix -= t0;
    jx -= t0;
    const bool act_i = (ix >= 0) && (ix < N);
    const bool act_j = (jx >= 0) && (jx < N);

    for (int row = 0; row < 2; row++) {
      const float w_raw = (row == 0) ? wx : wy;
      if (w_raw == 0.0f) {
        continue;
      }

      float Jj[6], Ji[6], Jz, r;
      if (row == 0) {
        r = rx;
        Jz = fx * (tij[0] * d - tij[2] * (X * d2));
        Jj[0] = fx * W * d;
        Jj[1] = 0.0f;
        Jj[2] = fx * (-X) * W * d2;
        Jj[3] = fx * (-X) * Y * d2;
        Jj[4] = fx * (1.0f + X * X * d2);
        Jj[5] = fx * (-Y) * d;
      } else {
        r = ry;
        Jz = fy * (tij[1] * d - tij[2] * (Y * d2));
        Jj[0] = 0.0f;
        Jj[1] = fy * W * d;
        Jj[2] = fy * (-Y) * W * d2;
        Jj[3] = fy * (-1.0f - Y * Y * d2);
        Jj[4] = fy * (X * Y * d2);
        Jj[5] = fy * X * d;
      }

      const float w = w_raw * robust_weight;
      dpvo_adjSE3(tij, qij, Jj, Ji);

      for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
          if (act_i) {
            atomicAdd(&@out0(6 * ix + i, 6 * ix + j), w * Ji[i] * Ji[j]);
          }
          if (act_j) {
            atomicAdd(&@out0(6 * jx + i, 6 * jx + j), w * Jj[i] * Jj[j]);
          }
          if (act_i && act_j) {
            atomicAdd(&@out0(6 * ix + i, 6 * jx + j), -w * Ji[i] * Jj[j]);
            atomicAdd(&@out0(6 * jx + i, 6 * ix + j), -w * Jj[i] * Ji[j]);
          }
        }
      }

      if (eff_impl) {
        const int64_t kp = kx % ppf;
        for (int i = 0; i < 6; i++) {
          atomicAdd(&@out5(ijs, kp, i), -w * Jz * Ji[i]);
          atomicAdd(&@out5(ijx, kp, i),  w * Jz * Jj[i]);
        }
      } else {
        for (int i = 0; i < 6; i++) {
          if (act_i) {
            atomicAdd(&@out1(6 * ix + i, k), -w * Jz * Ji[i]);
          }
          if (act_j) {
            atomicAdd(&@out1(6 * jx + i, k),  w * Jz * Jj[i]);
          }
        }
      }

      for (int i = 0; i < 6; i++) {
        if (act_i) {
          atomicAdd(&@out3(6 * ix + i), -w * r * Ji[i]);
        }
        if (act_j) {
          atomicAdd(&@out3(6 * jx + i),  w * r * Jj[i]);
        }
      }

      atomicAdd(&@out2(k), w * Jz * Jz);
      atomicAdd(&@out4(k), w * r * Jz);
    }
  }
}

cudaMemsetAsync(out0_p, 0, out0->size);
cudaMemsetAsync(out1_p, 0, out1->size);
cudaMemsetAsync(out2_p, 0, out2->size);
cudaMemsetAsync(out3_p, 0, out3->size);
cudaMemsetAsync(out4_p, 0, out4->size);
cudaMemsetAsync(out5_p, 0, out5->size);

const int64_t t0 = (int64_t)data["t0"];
const int64_t N = (int64_t)data["N"];
const int ppf = (int)data["ppf"];
const float huber_delta = (float)data["huber_delta"];
const int64_t E_edges = in6_shape0;
const int threads = 256;
int blocks = (int)((E_edges + threads - 1) / threads);
if (blocks > 65535) blocks = 65535;
if (blocks > 0) {
  reprojection_residuals_and_hessian_kernel<<<blocks, threads>>>(@ARGS, t0, N, ppf, huber_delta);
}
"""

_EET_CUDA_SRC = """
__global__ static void EEt_kernel(@ARGS_DEF, const int64_t t0, const int64_t N, const int ppf) {
  @PRECALC
  const int64_t total = in2_shape0 * (int64_t)ppf;
  DPVO_GPU_1D_KERNEL_LOOP(n, total) {
    const int k = (int)(n % ppf);
    const int64_t idx = n / ppf;
    const int64_t i = @in2(idx, 0);
    const int64_t j1 = @in2(idx, 1) - t0;
    const int64_t j2 = @in2(idx, 2) - t0;
    if (j1 >= 0 && j1 < N && j2 >= 0 && j2 < N) {
      const int64_t q_idx = @in3(i, k);
      if (q_idx >= 0) {
        const int64_t j1_idx = @in2(idx, 3);
        const int64_t j2_idx = @in2(idx, 4);
        const float q = @in1(q_idx);
        for (int xi = 0; xi < 6; xi++) {
          const float a = @in0(j1_idx, k, xi) * q;
          for (int xj = 0; xj < 6; xj++) {
            atomicAdd(&@out(6 * j1 + xi, 6 * j2 + xj), a * @in0(j2_idx, k, xj));
          }
        }
      }
    }
  }
}

cudaMemsetAsync(out_p, 0, out->size);
const int64_t t0 = (int64_t)data["t0"];
const int64_t N = (int64_t)data["N"];
const int ppf = (int)data["ppf"];
const int64_t total = in2_shape0 * (int64_t)ppf;
const int threads = 256;
int blocks = (int)((total + threads - 1) / threads);
if (blocks > 65535) blocks = 65535;
if (blocks > 0) {
  EEt_kernel<<<blocks, threads>>>(@ARGS, t0, N, ppf);
}
"""

_EV_CUDA_SRC = """
__global__ static void Ev_kernel(@ARGS_DEF, const int64_t t0, const int64_t N, const int ppf) {
  @PRECALC
  const int64_t total = in0_shape0 * (int64_t)ppf;
  DPVO_GPU_1D_KERNEL_LOOP(n, total) {
    const int k = (int)(n % ppf);
    const int64_t idx = n / ppf;
    const int64_t i = @in2(idx, 0);
    const int64_t j = @in2(idx, 1) - t0;
    if (j >= 0 && j < N) {
      const int64_t q_idx = @in3(i, k);
      if (q_idx >= 0) {
        const float v = @in1(q_idx);
        for (int r = 0; r < 6; r++) {
          atomicAdd(&@out(j * 6 + r), @in0(idx, k, r) * v);
        }
      }
    }
  }
}

cudaMemsetAsync(out_p, 0, out->size);
const int64_t t0 = (int64_t)data["t0"];
const int64_t N = (int64_t)data["N"];
const int ppf = (int)data["ppf"];
const int64_t total = in0_shape0 * (int64_t)ppf;
const int threads = 256;
int blocks = (int)((total + threads - 1) / threads);
if (blocks > 65535) blocks = 65535;
if (blocks > 0) {
  Ev_kernel<<<blocks, threads>>>(@ARGS, t0, N, ppf);
}
"""

_ETV_CUDA_SRC = """
__global__ static void Etv_kernel(@ARGS_DEF, const int64_t t0, const int64_t N, const int ppf) {
  @PRECALC
  const int64_t total = in0_shape0 * (int64_t)ppf;
  DPVO_GPU_1D_KERNEL_LOOP(n, total) {
    const int k = (int)(n % ppf);
    const int64_t idx = n / ppf;
    const int64_t i = @in2(idx, 0);
    const int64_t j = @in2(idx, 1) - t0;
    if (j >= 0 && j < N) {
      const int64_t q_idx = @in3(i, k);
      if (q_idx >= 0) {
        for (int r = 0; r < 6; r++) {
          atomicAdd(&@out(q_idx), @in0(idx, k, r) * @in1(j * 6 + r));
        }
      }
    }
  }
}

cudaMemsetAsync(out_p, 0, out->size);
const int64_t t0 = (int64_t)data["t0"];
const int64_t N = (int64_t)data["N"];
const int ppf = (int)data["ppf"];
const int64_t total = in0_shape0 * (int64_t)ppf;
const int threads = 256;
int blocks = (int)((total + threads - 1) / threads);
if (blocks > 65535) blocks = 65535;
if (blocks > 0) {
  Etv_kernel<<<blocks, threads>>>(@ARGS, t0, N, ppf);
}
"""

_POSE_RETR_CUDA_SRC = """
__global__ static void pose_retr_kernel(@ARGS_DEF, const int64_t t0) {
  @PRECALC
  const int64_t N = in0_shape0;
  DPVO_GPU_1D_KERNEL_LOOP(i, N) {
    const int64_t t = t0 + i;
    float p_t0[3] = { @out0(t, 0), @out0(t, 1), @out0(t, 2) };
    float p_q0[4] = { @out0(t, 3), @out0(t, 4), @out0(t, 5), @out0(t, 6) };
    float xi[6] = {
      @in0(i, 0), @in0(i, 1), @in0(i, 2),
      @in0(i, 3), @in0(i, 4), @in0(i, 5)
    };
    float p_t1[3], p_q1[4];
    dpvo_retrSE3(xi, p_t0, p_q0, p_t1, p_q1);

    @out0(t, 0) = p_t1[0];
    @out0(t, 1) = p_t1[1];
    @out0(t, 2) = p_t1[2];
    @out0(t, 3) = p_q1[0];
    @out0(t, 4) = p_q1[1];
    @out0(t, 5) = p_q1[2];
    @out0(t, 6) = p_q1[3];
  }
}

const int64_t t0 = (int64_t)data["t0"];
const int64_t N = in0_shape0;
const int threads = 256;
int blocks = (int)((N + threads - 1) / threads);
if (blocks > 65535) blocks = 65535;
if (blocks > 0) {
  pose_retr_kernel<<<blocks, threads>>>(@ARGS, t0);
}
"""

_PATCH_RETR_CUDA_SRC = """
__global__ static void patch_retr_kernel(@ARGS_DEF) {
  @PRECALC
  const int64_t M = in0_shape0;
  const int p_h = out0_shape2;
  const int p_w = out0_shape3;
  DPVO_GPU_1D_KERNEL_LOOP(n, M) {
    const int64_t ix = @in0(n);
    float d = @out0(ix, 2, 0, 0) + @in1(n);
    d = (d > 20.0f) ? 1.0f : d;
    d = fmaxf(d, 1e-4f);
    for (int i = 0; i < p_h; i++) {
      for (int j = 0; j < p_w; j++) {
        @out0(ix, 2, i, j) = d;
      }
    }
  }
}

const int64_t M = in0_shape0;
const int threads = 256;
int blocks = (int)((M + threads - 1) / threads);
if (blocks > 65535) blocks = 65535;
if (blocks > 0) {
  patch_retr_kernel<<<blocks, threads>>>(@ARGS);
}
"""


def _extract_var(obj):
    return obj if isinstance(obj, jt.Var) else obj.data


def neighbors_cuda(kk, jj):
    """Find previous/next temporal neighbor edge indices grouped by patch kk and sorted by jj."""
    kk_i = kk.reshape(-1).int64()
    jj_i = jj.reshape(-1).int64()
    E = int(kk_i.numel())
    if E == 0:
        return jt.zeros((0,), dtype=jt.int64), jt.zeros((0,), dtype=jt.int64)

    ix, jx = jt.code(
        [(E,), (E,)],
        ["int64", "int64"],
        [kk_i, jj_i],
        cpu_header=_NEIGHBORS_CPU_HEADER,
        cpu_src=_NEIGHBORS_CPU_SRC,
    )
    return ix, jx


def reproject_cuda(poses, patches, intrinsics, ii, jj, kk):
    """Reproject patches from frame ii to frame jj matching ba_cuda.cu reproject."""
    poses_var = _extract_var(poses).reshape(-1, 7).float32()
    patches_var = _extract_var(patches)
    P = int(patches_var.shape[-1])
    patches_var = patches_var.reshape(-1, 3, P, P).float32()
    intr_var = intrinsics.reshape(-1, 4).float32()

    ii_i = ii.reshape(-1).int64()
    jj_i = jj.reshape(-1).int64()
    kk_i = kk.reshape(-1).int64()
    E = int(ii_i.numel())
    if E == 0:
        return jt.zeros((1, 0, 2, P, P), dtype=jt.float32)

    coords = jt.code(
        (E, 2, P, P),
        "float32",
        [poses_var, patches_var, intr_var, ii_i, jj_i, kk_i],
        cuda_header=_COMMON_SE3_CUDA_HEADER,
        cuda_src=_REPROJECT_CUDA_SRC,
    )
    return coords.reshape(1, E, 2, P, P)


def _build_efficient_e_tables(ii_np, jj_np, kx_np, ppf):
    """Prebuild EfficentE lookup tables without Eigen or dynamic recompilation."""
    E_edges = int(ii_np.shape[0])
    n_frames = int(max(int(ii_np.max()), int(jj_np.max()), int(kx_np.max() // ppf))) + 1
    ij_pair = np.concatenate([ii_np * n_frames + jj_np, ii_np * n_frames + ii_np])
    ij_uniq_np, ij_inv_np = np.unique(ij_pair, return_inverse=True)

    U = int(ij_uniq_np.shape[0])
    ij_xself = jt.array(ij_inv_np.reshape(2, E_edges).astype(np.int64, copy=False)).int64()

    patch_to_ku_np = np.full((n_frames, ppf), -1, dtype=np.int64)
    patch_to_ku_np[kx_np // ppf, kx_np % ppf] = np.arange(kx_np.shape[0], dtype=np.int64)
    patch_to_ku = jt.array(patch_to_ku_np).int64()

    block_index_np = np.stack(
        [ij_uniq_np // n_frames, ij_uniq_np % n_frames], axis=1
    ).astype(np.int64, copy=False)
    block_index_tensor = jt.array(block_index_np).int64()

    _, counts = np.unique(ij_uniq_np // n_frames, return_counts=True)
    total_pairs = int(np.sum(counts * counts))
    ij_uniq_jt = jt.array(ij_uniq_np.astype(np.int64, copy=False)).int64()
    index_tensor = jt.code(
        (max(total_pairs, 1), 5),
        "int64",
        [ij_uniq_jt],
        data={"n_frames": float(n_frames)},
        cpu_src=_INDEX_TENSOR_CPU_SRC,
    )
    if total_pairs == 0:
        index_tensor = jt.zeros((0, 5), dtype=jt.int64)

    return U, ij_xself, patch_to_ku, index_tensor, block_index_tensor


def BA_cuda(
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
    """Gauss-Newton bundle adjustment using inline CUDA kernels and exact Schur Cholesky solve."""
    relative_depth_beta = float(relative_depth_beta)
    if not np.isfinite(relative_depth_beta) or relative_depth_beta < 0.0:
        raise RuntimeError("relative_depth_beta must be finite and nonnegative")

    huber_delta = float(huber_delta)
    poses_var = _extract_var(poses)
    patches_var = _extract_var(patches)

    E_edges = int(ii.numel())
    if E_edges == 0:
        return poses_var, patches_var

    P = int(patches_var.shape[-1])
    poses_w = poses_var.reshape(-1, 7).float32().clone()
    patches_w = patches_var.reshape(-1, 3, P, P).float32().clone()
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

    # Jittor 1.3.11 jt.unique corrupts int64 tensors, so compute sorted unique via NumPy.
    kk_np = kk_w.numpy().astype(np.int64, copy=False)
    kx_np, ku_np = np.unique(kk_np, return_inverse=True)
    kx = jt.array(kx_np).int64()
    ku = jt.array(ku_np).int64()

    t0 = int(t0)
    t1 = int(t1)
    N = int(t1 - t0)
    M_uniq = int(kx_np.shape[0])
    ppf_val = int(M)
    use_eff = bool(eff_impl) and (N > 0) and (ppf_val > 0)

    if use_eff:
        ii_np = ii_w.numpy().astype(np.int64, copy=False)
        jj_np = jj_w.numpy().astype(np.int64, copy=False)
        U, ij_xself, patch_to_ku, index_tensor, block_index_tensor = _build_efficient_e_tables(
            ii_np, jj_np, kx_np, ppf_val
        )
        E_shape = (1, 1)
        E_lookup_shape = (U, ppf_val, 6)
        kernel_ppf = ppf_val
    else:
        ij_xself = jt.zeros((2, 1), dtype=jt.int64)
        E_shape = (6 * max(N, 1), M_uniq)
        E_lookup_shape = (1, 1, 6)
        kernel_ppf = 0

    B_shape = (6 * max(N, 1), 6 * max(N, 1))
    C_shape = (M_uniq,)
    v_shape = (6 * max(N, 1),)
    u_shape = (M_uniq,)

    diag_idx = np.arange(6 * N) if N > 0 else None

    for itr in range(iterations):
        B, E, C, v, u, E_lookup = jt.code(
            [B_shape, E_shape, C_shape, v_shape, u_shape, E_lookup_shape],
            ["float32", "float32", "float32", "float32", "float32", "float32"],
            [poses_w, patches_w, intr_w, target_w, weight_w, ij_xself, ii_w, jj_w, kk_w, ku],
            data={
                "t0": float(t0),
                "N": float(N),
                "ppf": float(kernel_ppf),
                "huber_delta": huber_delta,
            },
            cuda_header=_COMMON_SE3_CUDA_HEADER,
            cuda_src=_HESSIAN_CUDA_SRC,
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
            Q_flat = 1.0 / (C + lmbda_w + extra)
        else:
            Q_flat = 1.0 / (C + lmbda_w)

        if N == 0:
            dZ = Q_flat * u
            dZ_np = dZ.numpy()
            if not np.isfinite(dZ_np).all():
                raise RuntimeError(
                    f"DPVO numerical: non-finite depth update at BA iteration {itr}"
                )
        else:
            if use_eff:
                EQEt = jt.code(
                    (6 * N, 6 * N),
                    "float32",
                    [E_lookup, Q_flat, index_tensor, patch_to_ku],
                    data={"t0": float(t0), "N": float(N), "ppf": float(ppf_val)},
                    cuda_header=_COMMON_SE3_CUDA_HEADER,
                    cuda_src=_EET_CUDA_SRC,
                )
                Qu = Q_flat * u
                EQu = jt.code(
                    (6 * N,),
                    "float32",
                    [E_lookup, Qu, block_index_tensor, patch_to_ku],
                    data={"t0": float(t0), "N": float(N), "ppf": float(ppf_val)},
                    cuda_header=_COMMON_SE3_CUDA_HEADER,
                    cuda_src=_EV_CUDA_SRC,
                )
                S = B - EQEt
                y = v - EQu
            else:
                EQ = E * Q_flat.reshape(1, M_uniq)
                Et = E.transpose(0, 1)
                S = B - jt.matmul(EQ, Et)
                y = v - jt.matmul(EQ, u.reshape(M_uniq, 1)).reshape(6 * N)

            S_np = S.numpy()
            y_np = y.numpy()
            S_np[diag_idx, diag_idx] += 1e-4 * S_np[diag_idx, diag_idx] + 1.0

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
            if use_eff:
                EtdX = jt.code(
                    (M_uniq,),
                    "float32",
                    [E_lookup, dX.reshape(-1), block_index_tensor, patch_to_ku],
                    data={"t0": float(t0), "N": float(N), "ppf": float(ppf_val)},
                    cuda_header=_COMMON_SE3_CUDA_HEADER,
                    cuda_src=_ETV_CUDA_SRC,
                )
                dZ = Q_flat * (u - EtdX)
            else:
                dZ = Q_flat * (u - jt.matmul(Et, dX.reshape(6 * N, 1)).reshape(M_uniq))

            dZ_np = dZ.numpy()
            if not (np.isfinite(dX_np).all() and np.isfinite(dZ_np).all()):
                raise RuntimeError(
                    f"DPVO numerical: non-finite pose/depth update at BA iteration {itr}"
                )

            jt.code(
                [dX],
                [poses_w],
                data={"t0": float(t0)},
                cuda_header=_COMMON_SE3_CUDA_HEADER,
                cuda_src=_POSE_RETR_CUDA_SRC,
            )

        jt.code(
            [kx, dZ],
            [patches_w],
            cuda_header=_COMMON_SE3_CUDA_HEADER,
            cuda_src=_PATCH_RETR_CUDA_SRC,
        )

    poses_var.assign(poses_w.reshape(poses_var.shape).cast(poses_var.dtype))
    patches_var.assign(patches_w.reshape(patches_var.shape).cast(patches_var.dtype))
    return poses_var, patches_var
