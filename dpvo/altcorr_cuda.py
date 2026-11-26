import jittor as jt

_PATCHIFY_CUDA_HEADER = r"""
#include <type/fp16_compute.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <math.h>

__forceinline__ __device__
bool altcorr_within_bounds(int h, int w, int H, int W) {
    return h >= 0 && h < H && w >= 0 && w < W;
}

template <typename scalar_t>
__global__ void patchify_fwd_kernel(
    int R,
    int B,
    int M,
    int C,
    int H,
    int W,
    const scalar_t* __restrict__ net,
    const float* __restrict__ coords,
    scalar_t* __restrict__ patches)
{
    const int D = 2 * R + 2;
    const int64_t total = static_cast<int64_t>(B) * M * D * D;
    int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n < total) {
        const int ii = static_cast<int>(n % D); n /= D;
        const int jj = static_cast<int>(n % D); n /= D;
        const int  m = static_cast<int>(n % M); n /= M;
        const int  b = static_cast<int>(n);

        const int64_t c_idx = (static_cast<int64_t>(b) * M + m) * 2;
        const float x = coords[c_idx + 0];
        const float y = coords[c_idx + 1];
        const int i = static_cast<int>(floorf(y)) + (ii - R);
        const int j = static_cast<int>(floorf(x)) + (jj - R);

        if (altcorr_within_bounds(i, j, H, W)) {
            const int64_t DD = static_cast<int64_t>(D) * D;
            const int64_t HW = static_cast<int64_t>(H) * W;
            const int64_t patch_base =
                (static_cast<int64_t>(b) * M + m) * C * DD
                + static_cast<int64_t>(ii) * D + jj;
            const int64_t net_base =
                static_cast<int64_t>(b) * C * HW
                + static_cast<int64_t>(i) * W + j;
            for (int k = 0; k < C; ++k) {
                patches[patch_base + static_cast<int64_t>(k) * DD] =
                    net[net_base + static_cast<int64_t>(k) * HW];
            }
        }
    }
}
"""

_PATCHIFY_CUDA_SRC = r"""
const int R = static_cast<int>(data["R"]);
const int B = static_cast<int>(in0_shape0);
const int C = static_cast<int>(in0_shape1);
const int H = static_cast<int>(in0_shape2);
const int W = static_cast<int>(in0_shape3);
const int M = static_cast<int>(in1_shape1);
const int D = 2 * R + 2;

cudaMemsetAsync(out0_p, 0, out0->size, 0);
const int64_t total = static_cast<int64_t>(B) * M * D * D;
if (total > 0) {
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    patchify_fwd_kernel<in0_type><<<blocks, threads>>>(
        R, B, M, C, H, W, in0_p, in1_p, out0_p
    );
}
"""

_CORR_CUDA_HEADER = r"""
#include <type/fp16_compute.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <math.h>

__forceinline__ __device__
bool corr_within_bounds(int h, int w, int H, int W) {
    return h >= 0 && h < H && w >= 0 && w < W;
}

template <typename scalar_t>
__global__ void corr_forward_kernel(
    int R,
    int B,
    int N1,
    int N2,
    int C,
    int Hp,
    int Wp,
    int H2,
    int W2,
    int M,
    const scalar_t* __restrict__ fmap1,
    const scalar_t* __restrict__ fmap2,
    const float* __restrict__ coords,
    const int* __restrict__ us,
    const int* __restrict__ vs,
    scalar_t* __restrict__ corr)
{
    const int D = 2 * R + 2;
    const int64_t total = static_cast<int64_t>(B) * M * Hp * Wp * D * D;
    int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n < total) {
        const int jj = static_cast<int>(n % D); n /= D;
        const int ii = static_cast<int>(n % D); n /= D;
        const int j0 = static_cast<int>(n % Wp); n /= Wp;
        const int i0 = static_cast<int>(n % Hp); n /= Hp;
        const int  m = static_cast<int>(n % M); n /= M;
        const int  b = static_cast<int>(n);

        const int ix = us[m];
        const int jx = vs[m];

        const int64_t HW1 = static_cast<int64_t>(Hp) * Wp;
        const int64_t coord_base =
            (static_cast<int64_t>(b) * M + m) * 2 * HW1
            + static_cast<int64_t>(i0) * Wp + j0;
        const float x = coords[coord_base];
        const float y = coords[coord_base + HW1];

        const int i1 = static_cast<int>(floorf(y)) + (ii - R);
        const int j1 = static_cast<int>(floorf(x)) + (jj - R);

        float s = 0.0f;
        if (corr_within_bounds(i1, j1, H2, W2)) {
            const int64_t HW2 = static_cast<int64_t>(H2) * W2;
            const int64_t f1_base =
                ((static_cast<int64_t>(b) * N1 + ix) * C * Hp + i0) * Wp + j0;
            const int64_t f2_base =
                ((static_cast<int64_t>(b) * N2 + jx) * C * H2 + i1) * W2 + j1;

            int i = 0;
            #pragma unroll 8
            for (; i + 8 <= C; i += 8) {
                scalar_t f1[8];
                scalar_t f2[8];
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    f1[j] = fmap1[f1_base + static_cast<int64_t>(i + j) * HW1];
                    f2[j] = fmap2[f2_base + static_cast<int64_t>(i + j) * HW2];
                }
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    s += static_cast<float>(f1[j]) * static_cast<float>(f2[j]);
                }
            }
            for (; i < C; ++i) {
                s += static_cast<float>(fmap1[f1_base + static_cast<int64_t>(i) * HW1]) *
                     static_cast<float>(fmap2[f2_base + static_cast<int64_t>(i) * HW2]);
            }
        }

        const int64_t out_idx =
            (((((static_cast<int64_t>(b) * M + m) * D + ii) * D + jj) * Hp + i0) * Wp + j0);
        corr[out_idx] = static_cast<scalar_t>(s);
    }
}
"""

_CORR_CUDA_SRC = r"""
const int R = static_cast<int>(data["R"]);
const int B = static_cast<int>(in0_shape0);
const int N1 = static_cast<int>(in0_shape1);
const int C = static_cast<int>(in0_shape2);
const int Hp = static_cast<int>(in0_shape3);
const int Wp = static_cast<int>(in0_shape4);
const int N2 = static_cast<int>(in1_shape1);
const int H2 = static_cast<int>(in1_shape3);
const int W2 = static_cast<int>(in1_shape4);
const int M = static_cast<int>(in2_shape1);
const int D = 2 * R + 2;

const int64_t total = static_cast<int64_t>(B) * M * Hp * Wp * D * D;
if (total > 0) {
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    corr_forward_kernel<in0_type><<<blocks, threads>>>(
        R, B, N1, N2, C, Hp, Wp, H2, W2, M,
        in0_p, in1_p, in2_p, in3_p, in4_p, out0_p
    );
}
"""


def patchify_raw_cuda(net, coords, radius):
    """Extract raw (D x D) patches using inline CUDA kernel without bilinear interpolation."""
    radius = int(radius)
    B, C, _, _ = net.shape
    M = coords.shape[1]
    D = 2 * radius + 2
    if B == 0 or M == 0:
        return jt.zeros((B, M, C, D, D), dtype=net.dtype)

    net_c = net.contiguous()
    coords_f = coords.float32().contiguous()
    return jt.code(
        (B, M, C, D, D),
        net_c.dtype,
        [net_c, coords_f],
        cuda_header=_PATCHIFY_CUDA_HEADER,
        cuda_src=_PATCHIFY_CUDA_SRC,
        data={"R": radius},
    )


def corr_cuda(fmap1, fmap2, coords, ii, jj, radius=3, dropout=1.0):
    """Compute local correlation volumes with inline CUDA kernel and bilinear interpolation."""
    del dropout  # Unused during inference forward pass.
    radius = int(radius)
    B, M, _, Hp, Wp = coords.shape
    d = 2 * radius + 1
    D = 2 * radius + 2
    out_dtype = fmap1.dtype
    if B == 0 or M == 0:
        return jt.zeros((B, M, d, d, Hp, Wp), dtype=out_dtype)

    fmap1_c = fmap1.contiguous()
    fmap2_c = (fmap2 if fmap2.dtype == out_dtype else fmap2.cast(out_dtype)).contiguous()
    coords_f = coords.float32().contiguous()
    ii_i = ii.reshape(-1).int32().contiguous()
    jj_i = jj.reshape(-1).int32().contiguous()

    corr_raw = jt.code(
        (B, M, D, D, Hp, Wp),
        out_dtype,
        [fmap1_c, fmap2_c, coords_f, ii_i, jj_i],
        cuda_header=_CORR_CUDA_HEADER,
        cuda_src=_CORR_CUDA_SRC,
        data={"R": radius},
    )

    x = coords_f[:, :, 0].reshape(B, M, 1, 1, Hp, Wp)
    y = coords_f[:, :, 1].reshape(B, M, 1, 1, Hp, Wp)
    dx = (x - jt.floor(x)).cast(out_dtype)
    dy = (y - jt.floor(y)).cast(out_dtype)
    one = jt.ones((), dtype=out_dtype)

    out = (
        (one - dx) * (one - dy) * corr_raw[:, :, :d, :d]
        + dx * (one - dy) * corr_raw[:, :, :d, 1:]
        + (one - dx) * dy * corr_raw[:, :, 1:, :d]
        + dx * dy * corr_raw[:, :, 1:, 1:]
    )
    return out.cast(out_dtype).permute(0, 1, 3, 2, 4, 5)
