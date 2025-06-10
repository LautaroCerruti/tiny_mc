#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math_constants.h>
#include "params.h"

// CUDA kernel: one thread simulates one photon
__global__ void photon_kernel(float* heats, float* heats_squared, unsigned int photons_per_thread) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PHOTONS/photons_per_thread) return;

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
                weight = 1.0f; // Reset weight for next photon
                continue; // Skip to next photon
            } else {
                weight *= 10f;
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

void launch_simulation(float *h_heats, float *h_heats_sq) {
    // Allocate and zero device arrays
    float *d_heats, *d_heats_sq;
    cudaMalloc(&d_heats,  SHELLS * sizeof(float));
    cudaMalloc(&d_heats_sq, SHELLS * sizeof(float));
    cudaMemset(d_heats,  0, SHELLS * sizeof(float));
    cudaMemset(d_heats_sq, 0, SHELLS * sizeof(float));

    // Choose launch parameters
    int threads = 256;
    int blocks = (PHOTONS + threads - 1) / threads;

    // Launch kernel with a time- or user-defined seed
    photon_kernel<<<blocks, threads>>>(d_heats, d_heats_sq, 12345ULL);
    cudaDeviceSynchronize();

    // Copy results back to host...
    // float h_heats[SHELLS], h_heats_sq[SHELLS];
    // cudaMemcpy(h_heats,    d_heats,    SHELLS * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_heats_sq, d_heats_sq, SHELLS * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_heats);
    cudaFree(d_heats_sq);
}
