#include <math.h>
#include <stdlib.h>

#include "params.h"
#include "xoshiro.h"

#define PI 3.14159265358979323846f

void photon_lab1(float* heats, float* heats_squared)
{
    const float albedo = MU_S / (MU_S + MU_A);
    const float shells_per_mfp = 1e4 / MICRONS_PER_SHELL / (MU_A + MU_S);

    /* launch */
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float u = 0.0f;
    float v = 0.0f;
    float w = 1.0f;
    float weight = 1.0f;

    for (;;) {
        float t = -logf(next_float()); /* move */
        x += t * u;
        y += t * v;
        z += t * w;

        unsigned int shell = sqrtf(x * x + y * y + z * z) * shells_per_mfp; /* absorb */
        if (shell > SHELLS - 1) {
            shell = SHELLS - 1;
        }
        heats[shell] += (1.0f - albedo) * weight;
        heats_squared[shell] += (1.0f - albedo) * (1.0f - albedo) * weight * weight; /* add up squares */
        weight *= albedo;

        /* New direction, rejection method */
        float xi1, xi2;
        do {
            xi1 = 2.0f * next_float() - 1.0f;
            xi2 = 2.0f * next_float() - 1.0f;
            t = xi1 * xi1 + xi2 * xi2;
        } while (1.0f < t);
        u = 2.0f * t - 1.0f;
        v = xi1 * sqrtf((1.0f - u * u) / t);
        w = xi2 * sqrtf((1.0f - u * u) / t);

        if (weight < 0.001f) { /* roulette */
            if (next_float() > 0.1f)
                break;
            weight /= 0.1f;
        }
    }
}

void photon(float* heats, float* heats_squared)
{
    const float albedo = MU_S / (MU_S + MU_A);
    const float shells_per_mfp = 1e4 / MICRONS_PER_SHELL / (MU_A + MU_S);

    /* launch */
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float u = 0.0f;
    float v = 0.0f;
    float w = 1.0f;
    float weight = 1.0f;

    for (;;) {
        float t = -logf(next_float()); /* move */
        x += t * u;
        y += t * v;
        z += t * w;

        unsigned int shell = sqrtf(x * x + y * y + z * z) * shells_per_mfp; /* absorb */
        if (shell > SHELLS - 1) {
            shell = SHELLS - 1;
        }
        heats[shell] += (1.0f - albedo) * weight;
        heats_squared[shell] += (1.0f - albedo) * (1.0f - albedo) * weight * weight; /* add up squares */
        weight *= albedo;

        float r, theta, xi1, xi2;
        t = next_float();
        r = sqrtf(t);
        theta = 2.0f * PI * next_float();
        xi1 = r * cosf(theta);
        xi2 = r * sinf(theta);

        u = 2.0f * t - 1.0f;
        float sqrt_val = 2.0f * sqrtf(1.0f - t);
        v = xi1 * sqrt_val;
        w = xi2 * sqrt_val;

        if (weight < 0.001f) { /* roulette */
            if (next_float() > 0.1f)
                break;
            weight *= 10.0f;
        }
    }
}

void photon_vectorized(float* heats, float* heats_squared) {
    float x[BLOCK_SIZE] = {0.0f};
    float y[BLOCK_SIZE] = {0.0f};
    float z[BLOCK_SIZE] = {0.0f};
    float u[BLOCK_SIZE] = {0.0f};
    float v[BLOCK_SIZE] = {0.0f};
    float w[BLOCK_SIZE];
    float weight[BLOCK_SIZE];
    int active[BLOCK_SIZE];

    unsigned int shell[BLOCK_SIZE];
    
    // Se vectoriza el siguiente for
    for (int i = 0; i < BLOCK_SIZE; i++) {
        w[i] = 1.0f;
        weight[i] = 1.0f;
        active[i] = 1;
    }

    const float albedo = MU_S / (MU_S + MU_A);
    const float shells_per_mfp = 1e4f / MICRONS_PER_SHELL / (MU_A + MU_S);

    int photons_remaining = BLOCK_SIZE;
    while (photons_remaining > 0) {
        photons_remaining = 0;
        float rands[BLOCK_SIZE*4];
        next_float_vector_4_times_block(rands);
        float h[BLOCK_SIZE];
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float t;
            t = -logf(rands[i]);
            x[i] += t * u[i];
            y[i] += t * v[i];
            z[i] += t * w[i];
            shell[i] = (unsigned int)(sqrtf(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]) * shells_per_mfp);
            if (shell[i] >= SHELLS)
                shell[i] = SHELLS - 1;
            h[i] = (1.0f - albedo) * weight[i];
        }

        for (int i = 0; i < BLOCK_SIZE; i++) {
            if (active[i]) {
                heats[shell[i]] += h[i];
                heats_squared[shell[i]] += h[i]*h[i];
            }
        }

        // Se vectoriza el siguiente for
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float r, theta, xi1, xi2, t_val;
            t_val = rands[i+BLOCK_SIZE];
            r = sqrtf(t_val);
            theta = 2.0f * PI * rands[i+BLOCK_SIZE*2];
            xi1 = r * cosf(theta);
            xi2 = r * sinf(theta);
            u[i] = 2.0f * t_val - 1.0f;
            float sqrt_val = 2.0f * sqrtf(1.0f - t_val);
            v[i] = xi1 * sqrt_val;
            w[i] = xi2 * sqrt_val;
            
            weight[i] *= albedo;

            if (active[i]) {
                if (weight[i] < 0.001f) {
                    if (rands[i+BLOCK_SIZE*3] > 0.1f)
                        active[i] = 0;  // Se desactiva el fotón
                    else {
                        weight[i] *= 10.0f;
                    }
                }
            }
            if(active[i]) 
                photons_remaining++;
        }
    }
}