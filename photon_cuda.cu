#include "photon_cuda.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math_constants.h>
#include <math.h>
#include "params.h"
#include "helper_cuda.h"
#include <cstdint>

__global__ void photon_kernel_atomic_global(float* heats, float* heats_squared, unsigned int photons_per_thread) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize per-thread RNG
    curandState state;
    //curand_init(seed, idx, 0, &state);
    curand_init(clock64(), idx, 0, &state);

    // Precompute constants
    const float albedo = MU_S / (MU_S + MU_A);
    const float shells_per_mfp = 1e4f / MICRONS_PER_SHELL / (MU_A + MU_S);

    // Photon state
    float x = 0.0f, y = 0.0f, z = 0.0f;
    float u = 0.0f, v = 0.0f, w = 1.0f;
    float weight = 1.0f;

    unsigned int remaining_photons = photons_per_thread;

    // Photon life loop
    while (remaining_photons) {
        // Sample step length
        float rnd = curand_uniform(&state);
        float t = -logf(rnd);
        x += t * u;
        y += t * v;
        z += t * w;

        // Determine shell
        int shell = (int)(sqrtf(x*x + y*y + z*z) * shells_per_mfp);
        if (shell >= SHELLS) shell = SHELLS - 1;

        // Deposit energy
        float deposit = (1.0f - albedo) * weight;
        atomicAdd(&heats[shell], deposit);
        atomicAdd(&heats_squared[shell], deposit * deposit);

        // Update weight
        weight *= albedo;

        // Roulette for low-weight photons
        if (weight < 0.001f) {
            if (curand_uniform(&state) > 0.1f) {
                // Photon is absorbed
                remaining_photons--;
                x = 0.0f;
                y = 0.0f;
                z = 0.0f;
                u = 0.0f;
                v = 0.0f;
                w = 1.0f;
                weight = 1.0f;
                continue;
            } else {
                weight *= 10.0f;
            }
        }

        // Scatter: sample new direction using rejection method
        float xi1, xi2, s;
        do {
            xi1 = 2.0f * curand_uniform(&state) - 1.0f;
            xi2 = 2.0f * curand_uniform(&state) - 1.0f;
            s = xi1*xi1 + xi2*xi2;
        } while (s > 1.0f);

        u = 2.0f * s - 1.0f;
        //float factor = sqrtf((1.0f - u*u) / s);
        float factor = 2.0f * sqrtf(1.0f - s);
        v = xi1 * factor;
        w = xi2 * factor;
    }
}

__global__ void photon_kernel_shared(float* heats, float* heats_squared, unsigned int photons_per_thread) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // apuntadores en shared memory
    __shared__ float s_heats[SHELLS];
    __shared__ float s_heats_sq[SHELLS];

    // Inicialización de shared (por los primeros SHELLS hilos)
    if (tid < SHELLS) {
        s_heats[tid]    = 0.0f;
        s_heats_sq[tid] = 0.0f;
    }
    
    __syncthreads();

    // Initialize per-thread RNG
    curandState state;
    //curand_init(seed, idx, 0, &state);
    curand_init(clock64(), idx, 0, &state);

    // Precompute constants
    const float albedo = MU_S / (MU_S + MU_A);
    const float shells_per_mfp = 1e4f / MICRONS_PER_SHELL / (MU_A + MU_S);

    // Photon state
    float x = 0.0f, y = 0.0f, z = 0.0f;
    float u = 0.0f, v = 0.0f, w = 1.0f;
    float weight = 1.0f;

    unsigned int remaining_photons = photons_per_thread;

    // Photon life loop
    while (remaining_photons) {
        // Sample step length
        float rnd = curand_uniform(&state);
        float t = -logf(rnd);
        x += t * u;
        y += t * v;
        z += t * w;

        // Determine shell
        int shell = (int)(sqrtf(x*x + y*y + z*z) * shells_per_mfp);
        if (shell >= SHELLS) shell = SHELLS - 1;

        // Deposit energy
        float deposit = (1.0f - albedo) * weight;
        atomicAdd(&s_heats[shell], deposit);
        atomicAdd(&s_heats_sq[shell], deposit*deposit);

        // Update weight
        weight *= albedo;

        // Roulette for low-weight photons
        if (weight < 0.001f) {
            if (curand_uniform(&state) > 0.1f) {
                // Photon is absorbed
                remaining_photons--;
                x = 0.0f;
                y = 0.0f;
                z = 0.0f;
                u = 0.0f;
                v = 0.0f;
                w = 1.0f;
                weight = 1.0f;
                continue;
            } else {
                weight *= 10.0f;
            }
        }

        // Scatter: sample new direction using rejection method
        float xi1, xi2, s;
        do {
            xi1 = 2.0f * curand_uniform(&state) - 1.0f;
            xi2 = 2.0f * curand_uniform(&state) - 1.0f;
            s = xi1*xi1 + xi2*xi2;
        } while (s > 1.0f);

        u = 2.0f * s - 1.0f;
        //float factor = sqrtf((1.0f - u*u) / s);
        float factor = 2.0f * sqrtf(1.0f - s);
        v = xi1 * factor;
        w = xi2 * factor;
    }

    __syncthreads();

    if (tid < SHELLS) {
        atomicAdd(&heats[tid], s_heats[tid]);
        atomicAdd(&heats_squared[tid], s_heats_sq[tid]);
    }
}

__global__ void photon_kernel_polares(float* heats, float* heats_squared, unsigned int photons_per_thread) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // apuntadores en shared memory
    __shared__ float s_heats[SHELLS];
    __shared__ float s_heats_sq[SHELLS];

    // Inicialización de shared (por los primeros SHELLS hilos)
    if (tid < SHELLS) {
        s_heats[tid]    = 0.0f;
        s_heats_sq[tid] = 0.0f;
    }
    
    __syncthreads();

    // Initialize per-thread RNG
    curandState state;
    //curand_init(seed, idx, 0, &state);
    curand_init(clock64(), idx, 0, &state);

    // Precompute constants
    const float albedo = MU_S / (MU_S + MU_A);
    const float shells_per_mfp = 1e4f / MICRONS_PER_SHELL / (MU_A + MU_S);

    // Photon state
    float x = 0.0f, y = 0.0f, z = 0.0f;
    float u = 0.0f, v = 0.0f, w = 1.0f;
    float weight = 1.0f;

    unsigned int remaining_photons = photons_per_thread;

    // Photon life loop
    while (remaining_photons) {
        // Sample step length
        float rnd = curand_uniform(&state);
        float t = -logf(rnd);
        x += t * u;
        y += t * v;
        z += t * w;

        // Determine shell
        int shell = (int)(sqrtf(x*x + y*y + z*z) * shells_per_mfp);
        if (shell >= SHELLS) shell = SHELLS - 1;

        // Deposit energy
        float deposit = (1.0f - albedo) * weight;
        atomicAdd(&s_heats[shell], deposit);
        atomicAdd(&s_heats_sq[shell], deposit*deposit);

        // Update weight
        weight *= albedo;

        // Roulette for low-weight photons
        if (weight < 0.001f) {
            if (curand_uniform(&state) > 0.1f) {
                // Photon is absorbed
                remaining_photons--;
                x = 0.0f;
                y = 0.0f;
                z = 0.0f;
                u = 0.0f;
                v = 0.0f;
                w = 1.0f;
                weight = 1.0f;
                continue;
            } else {
                weight *= 10.0f;
            }
        }

        float xi1, xi2, s, r, sin, cos;

        s = curand_uniform(&state);
        r = sqrtf(s);
        sincospif(2.0f * curand_uniform(&state), &sin, &cos);
        xi1 = r * cos;
        xi2 = r * sin;

        u = 2.0f * s - 1.0f;
        float factor = 2.0f * sqrtf(1.0f - s);
        v = xi1 * factor;
        w = xi2 * factor;
    }

    __syncthreads();

    if (tid < SHELLS) {
        atomicAdd(&heats[tid], s_heats[tid]);
        atomicAdd(&heats_squared[tid], s_heats_sq[tid]);
    }
}

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
    // Usamos SplitMix64 para generar cuatro palabras de 32 bits
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

    // — primera muestra —
    result = st->s0 + st->s3;
    r.x    = result * norm;
    t      = st->s1 << 9;
    st->s2 ^= st->s0;  st->s3 ^= st->s1;
    st->s1 ^= st->s2;  st->s0 ^= st->s3;
    st->s2 ^= t;       st->s3 = rotl32(st->s3, 11);

    // — segunda muestra —
    result = st->s0 + st->s3;
    r.y    = result * norm;
    t      = st->s1 << 9;
    st->s2 ^= st->s0;  st->s3 ^= st->s1;
    st->s1 ^= st->s2;  st->s0 ^= st->s3;
    st->s2 ^= t;       st->s3 = rotl32(st->s3, 11);

    // — tercera muestra —
    result = st->s0 + st->s3;
    r.z    = result * norm;
    t      = st->s1 << 9;
    st->s2 ^= st->s0;  st->s3 ^= st->s1;
    st->s1 ^= st->s2;  st->s0 ^= st->s3;
    st->s2 ^= t;       st->s3 = rotl32(st->s3, 11);

    // — cuarta muestra —
    result = st->s0 + st->s3;
    r.w    = result * norm;
    // estado ya modificado

    return r;
}

__global__ void photon_kernel_xoshiro(float* heats, float* heats_squared, unsigned int photons_per_thread) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // apuntadores en shared memory
    __shared__ float s_heats[SHELLS];
    __shared__ float s_heats_sq[SHELLS];

    // Inicialización de shared (por los primeros SHELLS hilos)
    if (tid < SHELLS) {
        s_heats[tid]    = 0.0f;
        s_heats_sq[tid] = 0.0f;
    }
    
    __syncthreads();

    Xoshiro128pState state;
    uint64_t seed = clock64() + (uint64_t)idx;
    xoshiro128p_init(seed, &state);

    // Precompute constants
    const float albedo = MU_S / (MU_S + MU_A);
    const float shells_per_mfp = 1e4f / MICRONS_PER_SHELL / (MU_A + MU_S);

    // Photon state
    float x = 0.0f, y = 0.0f, z = 0.0f;
    float u = 0.0f, v = 0.0f, w = 1.0f;
    float weight = 1.0f;

    unsigned int remaining_photons = photons_per_thread;

    // Photon life loop
    while (remaining_photons) {
        float4 rnd4 = xoshiro128p_next4(&state);
        // Sample step length
        float rnd = rnd4.w;
        float t = -logf(rnd);
        x += t * u;
        y += t * v;
        z += t * w;

        // Determine shell
        int shell = (int)(sqrtf(x*x + y*y + z*z) * shells_per_mfp);
        if (shell >= SHELLS) shell = SHELLS - 1;

        // Deposit energy
        float deposit = (1.0f - albedo) * weight;
        atomicAdd(&s_heats[shell], deposit);
        atomicAdd(&s_heats_sq[shell], deposit*deposit);

        // Update weight
        weight *= albedo;

        // Roulette for low-weight photons
        if (weight < 0.001f) {
            if (rnd4.x > 0.1f) {
                // Photon is absorbed
                remaining_photons--;
                x = 0.0f;
                y = 0.0f;
                z = 0.0f;
                u = 0.0f;
                v = 0.0f;
                w = 1.0f;
                weight = 1.0f;
                continue;
            } else {
                weight *= 10.0f;
            }
        }

        float xi1, xi2, s, r, sin, cos;

        s = rnd4.y;
        r = sqrtf(s);
        sincospif(2.0f * rnd4.z, &sin, &cos);
        xi1 = r * cos;
        xi2 = r * sin;

        u = 2.0f * s - 1.0f;
        float factor = 2.0f * sqrtf(1.0f - s);
        v = xi1 * factor;
        w = xi2 * factor;
    }

    __syncthreads();

    if (tid < SHELLS) {
        atomicAdd(&heats[tid], s_heats[tid]);
        atomicAdd(&heats_squared[tid], s_heats_sq[tid]);
    }
}

__global__ void photon_kernel_rsqrt(float* heats, float* heats_squared, unsigned int photons_per_thread) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // apuntadores en shared memory
    __shared__ float s_heats[SHELLS];
    __shared__ float s_heats_sq[SHELLS];

    if (GPU_THREADS < SHELLS) {
        if (tid == 0) {
            for (int i = 0; i < SHELLS; i++) {
                s_heats[i]    = 0.0f;
                s_heats_sq[i] = 0.0f;
            }
        }
    } else {
        if (tid < SHELLS) {
            s_heats[tid]    = 0.0f;
            s_heats_sq[tid] = 0.0f;
        }
    }
    
    __syncthreads();

    Xoshiro128pState state;
    uint64_t seed = clock64() + (uint64_t)idx;
    xoshiro128p_init(seed, &state);

    // Precompute constants
    const float albedo = MU_S / (MU_S + MU_A);
    const float shells_per_mfp = 1e4f / MICRONS_PER_SHELL / (MU_A + MU_S);

    // Photon state
    float x = 0.0f, y = 0.0f, z = 0.0f;
    float u = 0.0f, v = 0.0f, w = 1.0f;
    float weight = 1.0f;

    unsigned int remaining_photons = photons_per_thread;

    // Photon life loop
    while (remaining_photons) {
        float4 rnd4 = xoshiro128p_next4(&state);
        // Sample step length
        float rnd = rnd4.w;
        float t = -logf(rnd);
        x += t * u;
        y += t * v;
        z += t * w;

        // Determine shell
        float aux = x*x + y*y + z*z;
        float inv_r = rsqrtf(aux);
        int shell = min(int(inv_r * aux * shells_per_mfp), SHELLS-1);

        // Deposit energy
        float deposit = (1.0f - albedo) * weight;
        atomicAdd(&s_heats[shell], deposit);
        atomicAdd(&s_heats_sq[shell], deposit*deposit);

        // Update weight
        weight *= albedo;

        // Roulette for low-weight photons
        if (weight < 0.001f) {
            if (rnd4.x > 0.1f) {
                // Photon is absorbed
                remaining_photons--;
                x = 0.0f;
                y = 0.0f;
                z = 0.0f;
                u = 0.0f;
                v = 0.0f;
                w = 1.0f;
                weight = 1.0f;
                continue;
            } else {
                weight *= 10.0f;
            }
        }

        float xi1, xi2, s, r, sin, cos;

        s = rnd4.y;
        r = rsqrtf(s) * s;
        sincospif(2.0f * rnd4.z, &sin, &cos);
        xi1 = r * cos;
        xi2 = r * sin;

        u = 2.0f * s - 1.0f;
        float temp = 1.0f - s;
        float factor = 2.0f * rsqrtf(temp) * temp;
        v = xi1 * factor;
        w = xi2 * factor;
    }

    __syncthreads();

    if (tid < SHELLS) {
        atomicAdd(&heats[tid], s_heats[tid]);
        atomicAdd(&heats_squared[tid], s_heats_sq[tid]);
    }

    if (GPU_THREADS < SHELLS) {
        if (tid == 0) {
            for (int i = 0; i < SHELLS; i++) {
                atomicAdd(&heats[i], s_heats[i]);
                atomicAdd(&heats_squared[i], s_heats_sq[i]);
            }
        }
    } else {
        if (tid < SHELLS) {
            atomicAdd(&heats[tid], s_heats[tid]);
            atomicAdd(&heats_squared[tid], s_heats_sq[tid]);
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

    //int blocks = (PHOTONS / GPU_PHOTONS_PER_THREAD) / GPU_THREADS;
    unsigned int photons_per_thread = PHOTONS / (GPU_BLOCKS * GPU_THREADS);
    printf("# Launching %d blocks with %d threads each and %d photons per thread\n", GPU_BLOCKS, GPU_THREADS, photons_per_thread);

    checkCudaCall(cudaEventRecord(start, 0));

    // kernel v1
    //photon_kernel_atomic_global<<<GPU_BLOCKS, GPU_THREADS>>>(d_heats, d_heats_sq, photons_per_thread);

    // kernel v2
    //photon_kernel_shared<<<GPU_BLOCKS, GPU_THREADS>>>(d_heats, d_heats_sq, photons_per_thread);

    // kernel v3
    //photon_kernel_polares<<<GPU_BLOCKS, GPU_THREADS>>>(d_heats, d_heats_sq, photons_per_thread);

    // kernel v7
    //photon_kernel_xoshiro<<<GPU_BLOCKS, GPU_THREADS>>>(d_heats, d_heats_sq, photons_per_thread);

    // kernel v8
    photon_kernel_rsqrt<<<GPU_BLOCKS, GPU_THREADS>>>(d_heats, d_heats_sq, photons_per_thread);

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
