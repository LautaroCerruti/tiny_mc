#include "photon_cuda.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math_constants.h>
#include <math.h>
#include "params.h"
#include "helper_cuda.h"
#include <math_constants.h>
#include <cstdint>

struct Xoshiro128pState {
    uint32_t s0, s1, s2, s3;
};

__device__ __forceinline__ uint32_t rotl32(uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

__device__ __forceinline__ uint64_t splitmix64(uint64_t &state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__device__ __forceinline__ void xoshiro128p_init(uint64_t seed, Xoshiro128pState *st) {
    uint64_t sm = seed;
    st->s0 = (uint32_t)splitmix64(sm);
    st->s1 = (uint32_t)splitmix64(sm);
    st->s2 = (uint32_t)splitmix64(sm);
    st->s3 = (uint32_t)splitmix64(sm);
}

__device__ __forceinline__ float4 xoshiro128p_next4(Xoshiro128pState *st) {
    const float norm = 2.3283064365386963e-10f; // 1/2^32

    float4 r;
    uint32_t result, t;

    result = st->s0 + st->s3;
    r.x    = result * norm;
    t      = st->s1 << 9;
    st->s2 ^= st->s0;  st->s3 ^= st->s1;
    st->s1 ^= st->s2;  st->s0 ^= st->s3;
    st->s2 ^= t;       st->s3 = rotl32(st->s3, 11);

    result = st->s0 + st->s3;
    r.y    = result * norm;
    t      = st->s1 << 9;
    st->s2 ^= st->s0;  st->s3 ^= st->s1;
    st->s1 ^= st->s2;  st->s0 ^= st->s3;
    st->s2 ^= t;       st->s3 = rotl32(st->s3, 11);

    result = st->s0 + st->s3;
    r.z    = result * norm;
    t      = st->s1 << 9;
    st->s2 ^= st->s0;  st->s3 ^= st->s1;
    st->s1 ^= st->s2;  st->s0 ^= st->s3;
    st->s2 ^= t;       st->s3 = rotl32(st->s3, 11);

    result = st->s0 + st->s3;
    r.w    = result * norm;
    t      = st->s1 << 9;
    st->s2 ^= st->s0;  st->s3 ^= st->s1;
    st->s1 ^= st->s2;  st->s0 ^= st->s3;
    st->s2 ^= t;       st->s3 = rotl32(st->s3, 11);

    return r;
}

__device__ __forceinline__ float xoshiro128p_next(Xoshiro128pState *st) {
    uint32_t result = st->s0 + st->s3;
    uint32_t t      = st->s1 << 9;
    st->s2 ^= st->s0;  st->s3 ^= st->s1;
    st->s1 ^= st->s2;  st->s0 ^= st->s3;
    st->s2 ^= t;       st->s3 = rotl32(st->s3, 11);
    return result * (1.0f / 4294967296.0f);
}

__global__ void photon_kernel(float* heats, float* heats_squared, unsigned int photons_per_thread) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lid  = tid%warpSize;

    __shared__ float s_heats[SHELLS][32];
    __shared__ float s_heats_sq[SHELLS][32];

    if (GPU_THREADS < SHELLS) {
        if (tid < warpSize) {
            for (int i = 0; i < SHELLS; i++) {
                s_heats[i][lid]    = 0.0f;
                s_heats_sq[i][lid] = 0.0f;
            }
        }
    } else {
        if (tid < SHELLS) {
            for (int j = 0; j < warpSize; j++) {
                s_heats[tid][j]    = 0.0f;
                s_heats_sq[tid][j] = 0.0f;
            }
        }
    }

    __syncthreads();

    Xoshiro128pState state;
    uint64_t seed = clock64() + (uint64_t)idx;
    xoshiro128p_init(seed, &state);

    const float albedo = MU_S / (MU_S + MU_A);
    const float shells_per_mfp = 1e4f / MICRONS_PER_SHELL / (MU_A + MU_S);

    float x = 0.0f, y = 0.0f, z = 0.0f;
    float u = 0.0f, v = 0.0f, w = 1.0f;
    float weight = 1.0f;

    unsigned int remaining_photons = photons_per_thread;

    while (remaining_photons) {
        float t = -logf(xoshiro128p_next(&state));
        x += t * u;
        y += t * v;
        z += t * w;

        int shell = min(int(sqrtf(x*x + y*y + z*z) * shells_per_mfp), SHELLS-1);

        float deposit = (1.0f - albedo) * weight;
        atomicAdd(&s_heats[shell][lid], deposit);
        atomicAdd(&s_heats_sq[shell][lid], deposit*deposit);

        weight *= albedo;

        u = 2.0f * xoshiro128p_next(&state) - 1.0f;
        float temp = sqrtf(1.0f - u*u);
        float sin, cos;
        sincosf(2.0f * CUDART_PI_F * xoshiro128p_next(&state), &sin, &cos);
        v = sin * temp;
        w = cos * temp;

        if (weight < 0.001f) {
            weight *= 10.0f;
            if (xoshiro128p_next(&state) > 0.1f) {
                x = 0.0f;
                y = 0.0f;
                z = 0.0f;
                u = 0.0f;
                v = 0.0f;
                w = 1.0f;
                weight = 1.0f;
                remaining_photons--;
            }
        }
    }

    __syncthreads();

    if (tid < warpSize) {
        for (int i = 0; i < SHELLS; i++) {
            float warp_heat = s_heats[i][lid];
            #define FULL_MASK 0xffffffff
            warp_heat += __shfl_down_sync(FULL_MASK, warp_heat, 16);
            warp_heat += __shfl_down_sync(FULL_MASK, warp_heat, 8);
            warp_heat += __shfl_down_sync(FULL_MASK, warp_heat, 4);
            warp_heat += __shfl_down_sync(FULL_MASK, warp_heat, 2);
            warp_heat += __shfl_down_sync(FULL_MASK, warp_heat, 1);
            if (lid == 0) {
                atomicAdd(&heats[i], warp_heat);
            }

            float warp_heat_sq = s_heats_sq[i][lid];
            warp_heat_sq += __shfl_down_sync(FULL_MASK, warp_heat_sq, 16);
            warp_heat_sq += __shfl_down_sync(FULL_MASK, warp_heat_sq, 8);
            warp_heat_sq += __shfl_down_sync(FULL_MASK, warp_heat_sq, 4);
            warp_heat_sq += __shfl_down_sync(FULL_MASK, warp_heat_sq, 2);
            warp_heat_sq += __shfl_down_sync(FULL_MASK, warp_heat_sq, 1);
            if (lid == 0) {
                atomicAdd(&heats_squared[i], warp_heat_sq);
            }
        }
    }
}

void launch_simulation(float *h_heats, float *h_heats_sq, double *elapsed_time) {
    float *d_heats, *d_heats_sq;
    checkCudaCall(cudaMalloc(&d_heats,  SHELLS * sizeof(float)));
    checkCudaCall(cudaMalloc(&d_heats_sq, SHELLS * sizeof(float)));
    checkCudaCall(cudaMemset(d_heats,  0, SHELLS * sizeof(float)));
    checkCudaCall(cudaMemset(d_heats_sq, 0, SHELLS * sizeof(float)));

    cudaEvent_t start, stop;
    checkCudaCall(cudaEventCreate(&start));
    checkCudaCall(cudaEventCreate(&stop));

    unsigned int photons_per_thread = PHOTONS / (GPU_BLOCKS * GPU_THREADS);
    printf("# Launching %d blocks with %d threads each and %d photons per thread\n", GPU_BLOCKS, GPU_THREADS, photons_per_thread);

    checkCudaCall(cudaEventRecord(start, 0));

    photon_kernel<<<GPU_BLOCKS, GPU_THREADS>>>(d_heats, d_heats_sq, photons_per_thread);

    checkCudaCall(cudaEventRecord(stop, 0));
    checkCudaCall(cudaEventSynchronize(stop));

    float ms = 0.0f;
    checkCudaCall(cudaEventElapsedTime(&ms, start, stop));

    *elapsed_time = ms / 1000.0;

    checkCudaCall(cudaMemcpy(h_heats, d_heats, SHELLS * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaCall(cudaMemcpy(h_heats_sq, d_heats_sq, SHELLS * sizeof(float), cudaMemcpyDeviceToHost));

    checkCudaCall(cudaEventDestroy(start));
    checkCudaCall(cudaEventDestroy(stop));
    checkCudaCall(cudaFree(d_heats));
    checkCudaCall(cudaFree(d_heats_sq));
}
