#define _GNU_SOURCE
#include <math.h>
#include <stdlib.h>
#include <stdio.h> // printf()
#include <omp.h> // omp_get_wtime()

#include "params.h"
#include "xoshiro.h"
#define PI 3.14159265358979323846f

static float heat[SHELLS] __attribute__((aligned(64)));
static float heat2[SHELLS] __attribute__((aligned(64)));

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

void photon_vectorized_lab2(float* heats, float* heats_squared) {
    float x[BLOCK_SIZE] = {0.0f};
    float y[BLOCK_SIZE] = {0.0f};
    float z[BLOCK_SIZE] = {0.0f};
    float u[BLOCK_SIZE] = {0.0f};
    float v[BLOCK_SIZE] = {0.0f};
    float w[BLOCK_SIZE];
    float weight[BLOCK_SIZE];
    int active[BLOCK_SIZE];

    unsigned int shell[BLOCK_SIZE];

    // Se vectoriza el siguiente for 32 byte
    // ICX Se vectoriza el siguiente for 32 byte
    for (int i = 0; i < BLOCK_SIZE; i++) {
        w[i] = 1.0f;
        weight[i] = 1.0f;
        active[i] = 1;
    }

    const float albedo = MU_S / (MU_S + MU_A);
    const float shells_per_mfp = 1e4f / MICRONS_PER_SHELL / (MU_A + MU_S);

    int photons_remaining = BLOCK_SIZE;
    // no se vectoriza el siguiente while
    while (photons_remaining > 0) {
        photons_remaining = 0;
        float rands[BLOCK_SIZE*4];
        next_float_vector_4_times_block(rands);
        float h[BLOCK_SIZE];
        // Se vectoriza el siguiente for 32 byte
        // ICX Se vectoriza el siguiente for 32 byte
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

        // no se vectoriza el siguiente for
        // ICX no se vectoriza el siguiente for
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if (active[i]) {
                heats[shell[i]] += h[i];
                heats_squared[shell[i]] += h[i]*h[i];
            }
        }

        // Se vectoriza el siguiente for 32 byte
        // ICX Se vectoriza el siguiente for 32 byte
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
        }
        // Se vectoriza el siguiente for 32 byte
        // ICX no se vectoriza el siguiente for
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if (active[i]) {
                if (weight[i] < 0.001f) {
                    if (rands[i+BLOCK_SIZE*3] > 0.1f)
                        active[i] = 0;
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

void photon_vectorized_lab2_v2(float* heats, float* heats_squared) {
    float x[BLOCK_SIZE] __attribute__((aligned(64))) = {0.0f};
    float y[BLOCK_SIZE] __attribute__((aligned(32))) = {0.0f};
    float z[BLOCK_SIZE] __attribute__((aligned(32))) = {0.0f};
    float u[BLOCK_SIZE] __attribute__((aligned(32))) = {0.0f};
    float v[BLOCK_SIZE] __attribute__((aligned(32))) = {0.0f};
    float w[BLOCK_SIZE] __attribute__((aligned(32)));
    float weight[BLOCK_SIZE] __attribute__((aligned(32)));
    int active[BLOCK_SIZE] __attribute__((aligned(32)));

    unsigned int shell[BLOCK_SIZE] __attribute__((aligned(32)));

    // Se vectoriza el siguiente for 32 byte
    // ICX Se vectoriza el siguiente for 32 byte
    for (int i = 0; i < BLOCK_SIZE; i++) {
        w[i] = 1.0f;
        weight[i] = 1.0f;
        active[i] = 1;
    }

    const float albedo = MU_S / (MU_S + MU_A);
    const float shells_per_mfp = 1e4f / MICRONS_PER_SHELL / (MU_A + MU_S);

    int photons_remaining = BLOCK_SIZE;
    // no se vectoriza el siguiente while
    while (photons_remaining > 0) {
        photons_remaining = 0;
        float rands[BLOCK_SIZE*4] __attribute__((aligned(32)));
        next_float_vector_4_times_block(rands);
        float h[BLOCK_SIZE] __attribute__((aligned(32)));
        // Se vectoriza el siguiente for 32 byte
        // ICX Se vectoriza el siguiente for 32 byte
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

        // no se vectoriza el siguiente for
        // ICX no se vectoriza el siguiente for
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if (active[i]) {
                heats[shell[i]] += h[i];
                heats_squared[shell[i]] += h[i]*h[i];
            }
        }

        // Se vectoriza el siguiente for 32 byte
        // ICX Se vectoriza el siguiente for 32 byte
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
        }
        // Se vectoriza el siguiente for 32 byte
        // ICX vectoriza el siguiente for 32 byte
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if (weight[i] < 0.001f) {
                if (rands[i+BLOCK_SIZE*3] > 0.1f)
                    active[i] = 0;
                else {
                    weight[i] *= 10.0f;
                }
            }
        }
        // ICX vectoriza el siguiente for 32 byte
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if(active[i])
                photons_remaining++;
        }
    }
}

void photon_vectorized_mejora_v2(float *__restrict__ heats, float *__restrict__ heats_squared) {
    float x[BLOCK_SIZE]      __attribute__((aligned(64))) = {0.0f};
    float y[BLOCK_SIZE] __attribute__((aligned(32))) = {0.0f};
    float z[BLOCK_SIZE] __attribute__((aligned(32))) = {0.0f};
    float u[BLOCK_SIZE] __attribute__((aligned(32))) = {0.0f};
    float v[BLOCK_SIZE] __attribute__((aligned(32))) = {0.0f};
    float w[BLOCK_SIZE] __attribute__((aligned(32)));
    float weight[BLOCK_SIZE] __attribute__((aligned(32)));
    unsigned int active[BLOCK_SIZE] __attribute__((aligned(32)));

    // Se vectoriza el siguiente for 32 byte
    // ICX vectoriza el siguiente for 32 byte
    for (int i = 0; i < BLOCK_SIZE; i++) {
        w[i] = 1.0f;
        weight[i] = 1.0f;
        active[i] = 1;   // 1 indica fotón activo
    }

    const float albedo = MU_S / (MU_S + MU_A);
    const float shells_per_mfp = 1e4f / MICRONS_PER_SHELL / (MU_A + MU_S);

    float heat_update[MAGIC_N] __attribute__((aligned(64)));
    unsigned int shell_update[MAGIC_N];
    unsigned int update_count = 0;

    unsigned int photons_remaining = BLOCK_SIZE;
    while (photons_remaining > 0) {
        float rands[BLOCK_SIZE*4] __attribute__((aligned(64)));
        next_float_vector_4_times_block(rands);
        float h[BLOCK_SIZE] __attribute__((aligned(64)));
        unsigned int shell[BLOCK_SIZE] __attribute__((aligned(32)));
        // Se vectoriza el siguiente for 32 byte
        // ICX Se vectoriza el siguiente for 32 byte
        for (unsigned int i = 0; i < BLOCK_SIZE; i++) {
            float t = -logf(rands[i]);
            x[i] += t * u[i];
            y[i] += t * v[i];
            z[i] += t * w[i];
            float r = sqrtf(x[i]*x[i] + y[i]*y[i] + z[i]*z[i]);
            unsigned int s = (unsigned int)(r * shells_per_mfp);
            shell[i] = (s >= SHELLS) ? SHELLS-1 : s;
            h[i] = (1.0f - albedo) * weight[i];
        }

        if(update_count+photons_remaining > MAGIC_N) {
            // no se vectoriza el siguiente for
            // ICX no se vectoriza el siguiente for
            for (unsigned int j = 0; j < update_count; j++) {
                heats[shell_update[j]] += heat_update[j];
                heats_squared[shell_update[j]] += heat_update[j]*heat_update[j];
            }
            update_count = 0;
        }
        photons_remaining = 0;

        // no se vectoriza el siguiente for
        // ICX no se vectoriza el siguiente for
        for (unsigned int i = 0; i < BLOCK_SIZE; i++) {
            if (active[i]) {
                shell_update[update_count] = shell[i];
                heat_update[update_count] = h[i];
                update_count++;
            }
        }

        // Se vectoriza el siguiente for 32 byte
        // ICX Se vectoriza el siguiente for 32 byte
        for (unsigned int i = 0; i < BLOCK_SIZE; i++) {
            float t_val = rands[i+BLOCK_SIZE];
            float r = sqrtf(t_val);
            float theta = 2.0f * PI * rands[i+BLOCK_SIZE*2];
            float xi1 = r * cosf(theta);
            float xi2 = r * sinf(theta);
            u[i] = 2.0f * t_val - 1.0f;
            float sqrt_val = 2.0f * sqrtf(1.0f - t_val);
            v[i] = xi1 * sqrt_val;
            w[i] = xi2 * sqrt_val;

            weight[i] *= albedo;
        }
        // Se vectoriza el siguiente for 32 byte
        // ICX vectoriza el siguiente for 32 byte
        for (unsigned int i = 0; i < BLOCK_SIZE; i++) {
            if (weight[i] < 0.001f) {
                if (rands[i+BLOCK_SIZE*3] > 0.1f)
                    active[i] = 0;  // Se desactiva el fotón
                else {
                    weight[i] *= 10.0f;
                }
            }
            // active[i] = weight[i] < 0.001f && rands[i+BLOCK_SIZE*3] > 0.1f ? 0 : active[i];
            // weight[i] = weight[i] < 0.001f && rands[i+BLOCK_SIZE*3] <= 0.1f ? weight[i]*10.0f : weight[i];
        }
        // ICX vectoriza el siguiente for 32 byte
        for (unsigned int i = 0; i < BLOCK_SIZE; i++) {
            if(active[i])
                photons_remaining++;
        }
    }

    // no se vectoriza el siguiente for
    // ICX no se vectoriza el siguiente for
    for (unsigned int j = 0; j < update_count; j++) {
        heats[shell_update[j]] += heat_update[j];
        heats_squared[shell_update[j]] += heat_update[j]*heat_update[j];
    }
}

void photon_vectorized_prueba_v6(float *__restrict__ heats, float *__restrict__ heats_squared, unsigned int simulationCount) {
    float x[BLOCK_SIZE]      __attribute__((aligned(64))) = {0.0f};
    float y[BLOCK_SIZE]      __attribute__((aligned(32))) = {0.0f};
    float z[BLOCK_SIZE]      __attribute__((aligned(32))) = {0.0f};
    float u[BLOCK_SIZE]      __attribute__((aligned(32))) = {0.0f};
    float v[BLOCK_SIZE]      __attribute__((aligned(32))) = {0.0f};
    float w[BLOCK_SIZE]      __attribute__((aligned(32)));
    float weight[BLOCK_SIZE] __attribute__((aligned(32)));
    unsigned int active[BLOCK_SIZE] __attribute__((aligned(32)));

    // Se vectoriza el siguiente for 32 byte
    // ICX vectoriza el siguiente for 32 byte
    for (int i = 0; i < BLOCK_SIZE; i++) {
        w[i] = 1.0f;
        weight[i] = 1.0f;
        active[i] = 1;   // 1 indica fotón activo
    }
    simulationCount -= BLOCK_SIZE;

    const float albedo = MU_S / (MU_S + MU_A);
    const float shells_per_mfp = 1e4f / MICRONS_PER_SHELL / (MU_A + MU_S);

    unsigned short cont = 1;
    while (cont) {
        float rands[BLOCK_SIZE*4] __attribute__((aligned(64)));
        next_float_vector_4_times_block(rands);
        float h[BLOCK_SIZE] __attribute__((aligned(64)));
        unsigned short shell[BLOCK_SIZE] __attribute__((aligned(32)));
        // Se vectoriza el siguiente for 16 byte
        // ICX Se vectoriza el siguiente for 32 byte
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float t = -logf(rands[i]);
            x[i] += t * u[i];
            y[i] += t * v[i];
            z[i] += t * w[i];
            float r = sqrtf(x[i]*x[i] + y[i]*y[i] + z[i]*z[i]);
            unsigned short s = (unsigned short)(r * shells_per_mfp);
            shell[i] = (s >= SHELLS) ? SHELLS-1 : s;
            h[i] = (1.0f - albedo) * weight[i];
        }

        // no se vectoriza el siguiente for
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if (active[i]) {
                heats[shell[i]] += h[i];
                heats_squared[shell[i]] += h[i]*h[i];
            }
        }

        // Se vectoriza el siguiente for 32 byte
        // ICX Se vectoriza el siguiente for 32 byte
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float t_val = rands[i+BLOCK_SIZE*2];
            float r = sqrtf(t_val);
            float theta = 2.0f * PI * rands[i+BLOCK_SIZE*3];
            float xi1 = r * cosf(theta);
            float xi2 = r * sinf(theta);
            u[i] = 2.0f * t_val - 1.0f;
            float sqrt_val = 2.0f * sqrtf(1.0f - t_val);
            v[i] = xi1 * sqrt_val;
            w[i] = xi2 * sqrt_val;

            weight[i] *= albedo;
        }

        // Se vectoriza el siguiente for 32 byte
        // ICX Se vectoriza el siguiente for 32 byte
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if (active[i] && weight[i] < 0.001f) {
                if (rands[i+BLOCK_SIZE] > 0.1f) {
                    x[i] = 0.0f;
                    y[i] = 0.0f;
                    z[i] = 0.0f;
                    u[i] = 0.0f;
                    v[i] = 0.0f;
                    w[i] = 1.0f;
                    weight[i] = 1.0f;
                    active[i] = 0;  // Se desactiva el fotón
                }
                else {
                    weight[i] *= 10.0f;
                }
            }
        }
        // no se vectoriza el siguiente for
        // ICX no se vectoriza el siguiente for
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if(!active[i] && simulationCount > 0) {
                active[i] = 1;   // 1 indica fotón activo
                simulationCount--;
            }
        }
        cont = 0;
        // Se vectoriza el siguiente for 32 byte
        // ICX vectoriza el siguiente for 32 byte
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if(active[i])
                cont++;
        }
    }
}

void photon_vectorized_prueba_v7(float *__restrict__ heats, float *__restrict__ heats_squared, unsigned int simulationCount) {
    float x[BLOCK_SIZE]      __attribute__((aligned(64))) = {0.0f};
    float y[BLOCK_SIZE]      __attribute__((aligned(32))) = {0.0f};
    float z[BLOCK_SIZE]      __attribute__((aligned(32))) = {0.0f};
    float u[BLOCK_SIZE]      __attribute__((aligned(32))) = {0.0f};
    float v[BLOCK_SIZE]      __attribute__((aligned(32))) = {0.0f};
    float w[BLOCK_SIZE]      __attribute__((aligned(32)));
    float weight[BLOCK_SIZE] __attribute__((aligned(32)));
    unsigned int simPerTrack[BLOCK_SIZE] __attribute__((aligned(32)));

    // Se vectoriza el siguiente for 32 byte
    // ICX vectoriza el siguiente for 32 byte
    for (int i = 0; i < BLOCK_SIZE; i++) {
        simPerTrack[i] = simulationCount/BLOCK_SIZE;
        w[i] = 1.0f;
        weight[i] = 1.0f;
    }

    const float albedo = MU_S / (MU_S + MU_A);
    const float shells_per_mfp = 1e4f / MICRONS_PER_SHELL / (MU_A + MU_S);
    unsigned int hasToSim = BLOCK_SIZE;
    while (hasToSim > 0) {
        hasToSim = 0;
        float rands[BLOCK_SIZE*4] __attribute__((aligned(64)));
        next_float_vector_4_times_block(rands);
        float h[BLOCK_SIZE] __attribute__((aligned(64)));
        unsigned int shell[BLOCK_SIZE] __attribute__((aligned(32)));
        // Se vectoriza el siguiente for 32 byte
        // ICX Se vectoriza el siguiente for 32 byte
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float t = -logf(rands[i]);
            x[i] += t * u[i];
            y[i] += t * v[i];
            z[i] += t * w[i];
            float r = sqrtf(x[i]*x[i] + y[i]*y[i] + z[i]*z[i]);
            unsigned int s = (unsigned int)(r * shells_per_mfp);
            shell[i] = (s >= SHELLS) ? SHELLS-1 : s;
            h[i] = (1.0f - albedo) * weight[i];
        }

        // no se vectoriza el siguiente for
        // ICX no se vectoriza el siguiente for
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if (simPerTrack[i]) {
                heats[shell[i]] += h[i];
                heats_squared[shell[i]] += h[i]*h[i];
            }
        }

        // Se vectoriza el siguiente for 32 byte
        // ICX Se vectoriza el siguiente for 32 byte
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float t_val = rands[i+BLOCK_SIZE*2];
            float theta = 2.0f * PI * rands[i+BLOCK_SIZE*3];
            float r = sqrtf(t_val);
            float cos = cosf(theta);
            float sin = sinf(theta);
            u[i] = 2.0f * t_val - 1.0f;
            float sqrt_val = 2.0f * sqrtf(1.0f - t_val);
            v[i] = r * cos * sqrt_val;
            w[i] = r * sin * sqrt_val;

            weight[i] *= albedo;
            
            if (weight[i] < 0.001f) {
                if (rands[i+BLOCK_SIZE] > 0.1f) {
                    if (simPerTrack[i]>0) simPerTrack[i]--;
                    x[i] = simPerTrack[i] ? 0.0f : x[i];
                    y[i] = simPerTrack[i] ? 0.0f : y[i];
                    z[i] = simPerTrack[i] ? 0.0f : z[i];
                    u[i] = simPerTrack[i] ? 0.0f : u[i];
                    v[i] = simPerTrack[i] ? 0.0f : v[i];
                    w[i] = simPerTrack[i] ? 1.0f : w[i];
                    weight[i] = simPerTrack[i] ? 1.0f : weight[i];

                } else {
                    weight[i] = simPerTrack[i] ? weight[i] * 10.0f : weight[i];
                }
            }
            if(simPerTrack[i])
                hasToSim++;
        }
    }
}

void photon_vectorized_prueba_v8(float *__restrict__ heats, float *__restrict__ heats_squared, unsigned int simulationCount) {
    float x[BLOCK_SIZE]      __attribute__((aligned(64))) = {0.0f};
    float y[BLOCK_SIZE]      __attribute__((aligned(32))) = {0.0f};
    float z[BLOCK_SIZE]      __attribute__((aligned(32))) = {0.0f};
    float u[BLOCK_SIZE]      __attribute__((aligned(32))) = {0.0f};
    float v[BLOCK_SIZE]      __attribute__((aligned(32))) = {0.0f};
    float w[BLOCK_SIZE]      __attribute__((aligned(32)));
    float weight[BLOCK_SIZE] __attribute__((aligned(32)));
    unsigned int simPerTrack[BLOCK_SIZE] __attribute__((aligned(32)));

    // Se vectoriza el siguiente for 32 byte
    for (int i = 0; i < BLOCK_SIZE; i++) {
        simPerTrack[i] = simulationCount/BLOCK_SIZE;
        w[i] = 1.0f;
        weight[i] = 1.0f;
    }

    const float albedo = MU_S / (MU_S + MU_A);
    const float shells_per_mfp = 1e4f / MICRONS_PER_SHELL / (MU_A + MU_S);

    unsigned int hasToSim = 1;
    while (hasToSim > 0) {
        hasToSim = 0;
        float rands[BLOCK_SIZE*4] __attribute__((aligned(64)));
        next_float_vector_4_times_block(rands);
        float r[BLOCK_SIZE] __attribute__((aligned(64)));
        float h[BLOCK_SIZE] __attribute__((aligned(32)));
        unsigned short shell[BLOCK_SIZE] __attribute__((aligned(32)));
        // Se vectoriza el siguiente for 32 byte
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float t = -logf(rands[i]);
            x[i] += t * u[i];
            y[i] += t * v[i];
            z[i] += t * w[i];
            r[i] = sqrtf(x[i]*x[i] + y[i]*y[i] + z[i]*z[i]);
            h[i] = (1.0f - albedo) * weight[i];
        }
        for (int i = 0; i < BLOCK_SIZE; i++) {
            unsigned short s = (unsigned short)(r[i] * shells_per_mfp);
            shell[i] = (s >= SHELLS) ? SHELLS-1 : s;
        }

        // no se vectoriza el siguiente for
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if (simPerTrack[i]) {
                heats[shell[i]] += h[i];
                heats_squared[shell[i]] += h[i]*h[i];
            }
        }

        // Se vectoriza el siguiente for 32 byte
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float t_val = rands[i+BLOCK_SIZE*2];
            float r = sqrtf(t_val);
            float theta = 2.0f * PI * rands[i+BLOCK_SIZE*3];
            float xi1 = r * cosf(theta);
            float xi2 = r * sinf(theta);
            u[i] = 2.0f * t_val - 1.0f;
            float sqrt_val = 2.0f * sqrtf(1.0f - t_val);
            v[i] = xi1 * sqrt_val;
            w[i] = xi2 * sqrt_val;

            weight[i] *= albedo;
            
            if (weight[i] < 0.001f) {
                if (rands[i+BLOCK_SIZE] > 0.1f) {
                    if (simPerTrack[i]>0) simPerTrack[i]--;
                    x[i] = simPerTrack[i] ? 0.0f : x[i];
                    y[i] = simPerTrack[i] ? 0.0f : y[i];
                    z[i] = simPerTrack[i] ? 0.0f : z[i];
                    u[i] = simPerTrack[i] ? 0.0f : u[i];
                    v[i] = simPerTrack[i] ? 0.0f : v[i];
                    w[i] = simPerTrack[i] ? 1.0f : w[i];
                    weight[i] = simPerTrack[i] ? 1.0f : weight[i];

                } else {
                    weight[i] = simPerTrack[i] ? weight[i] * 10.0f : weight[i];
                }
            }
        }
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if(simPerTrack[i])
                hasToSim++;
        }
    }
}

void photon_vectorized_prueba_v9(float *__restrict__ heats, float *__restrict__ heats_squared, unsigned int simulationCount) {
    float x[BLOCK_SIZE]      __attribute__((aligned(64))) = {0.0f};
    float y[BLOCK_SIZE]      __attribute__((aligned(32))) = {0.0f};
    float z[BLOCK_SIZE]      __attribute__((aligned(32))) = {0.0f};
    float u[BLOCK_SIZE]      __attribute__((aligned(32))) = {0.0f};
    float v[BLOCK_SIZE]      __attribute__((aligned(32))) = {0.0f};
    float w[BLOCK_SIZE]      __attribute__((aligned(32)));
    float weight[BLOCK_SIZE] __attribute__((aligned(32)));
    unsigned int simPerTrack[BLOCK_SIZE] __attribute__((aligned(32)));

    // Se vectoriza el siguiente for 32 byte
    // ICX vectoriza el siguiente for 32 byte
    for (int i = 0; i < BLOCK_SIZE; i++) {
        simPerTrack[i] = simulationCount/BLOCK_SIZE;
        w[i] = 1.0f;
        weight[i] = 1.0f;
    }

    float heat_update[MAGIC_N] __attribute__((aligned(64)));
    unsigned short shell_update[MAGIC_N] __attribute__((aligned(32)));
    unsigned short update_count = 0;

    const float albedo = MU_S / (MU_S + MU_A);
    const float shells_per_mfp = 1e4f / MICRONS_PER_SHELL / (MU_A + MU_S);

    unsigned int hasToSim = 1;
    while (hasToSim > 0) {
        float rands[BLOCK_SIZE*4] __attribute__((aligned(64)));
        next_float_vector_4_times_block(rands);
        float r[BLOCK_SIZE] __attribute__((aligned(64)));
        float h[BLOCK_SIZE] __attribute__((aligned(32)));
        unsigned short shell[BLOCK_SIZE] __attribute__((aligned(32)));
        // Se vectoriza el siguiente for 32 byte
        // ICX Se vectoriza el siguiente for 32 byte
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float t = -logf(rands[i]);
            x[i] += t * u[i];
            y[i] += t * v[i];
            z[i] += t * w[i];
            r[i] = sqrtf(x[i]*x[i] + y[i]*y[i] + z[i]*z[i]);
            h[i] = (1.0f - albedo) * weight[i];
        }
        // Se vectoriza el siguiente for 16 byte
        // ICX se vectoriza el siguiente for
        for (int i = 0; i < BLOCK_SIZE; i++) {
            unsigned short s = (unsigned short)(r[i] * shells_per_mfp);
            shell[i] = (s >= SHELLS) ? SHELLS-1 : s;
        }

        if(update_count+hasToSim > MAGIC_N) {
            // no se vectoriza el siguiente for
            // ICX no se vectoriza el siguiente for
            for (unsigned int j = 0; j < update_count; j++) {
                heats[shell_update[j]] += heat_update[j];
                heats_squared[shell_update[j]] += heat_update[j]*heat_update[j];
            }
            update_count = 0;
        }

        if (hasToSim == BLOCK_SIZE) {
            // gcc ??????
            // ICX se vectoriza el siguiente for
            for (unsigned int i = 0; i < BLOCK_SIZE; i++) {
                shell_update[update_count+i] = shell[i];
                heat_update[update_count+i] = h[i];
            }
            update_count+=8;
        } else {
            // gcc no se vectoriza el siguiente for
            // ICX no se vectoriza el siguiente for
            for (unsigned int i = 0; i < BLOCK_SIZE; i++) {
                if (simPerTrack[i]) {
                    shell_update[update_count] = shell[i];
                    heat_update[update_count] = h[i];
                    update_count++;
                }
            }
        }

        hasToSim = 0;
        // Se vectoriza el siguiente for 32 byte
        // ICX se vecotoriza el siguiente for
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float t_val = rands[i+BLOCK_SIZE*2];
            float r = sqrtf(t_val);
            float theta = 2.0f * PI * rands[i+BLOCK_SIZE*3];
            float xi1 = r * cosf(theta);
            float xi2 = r * sinf(theta);
            u[i] = 2.0f * t_val - 1.0f;
            float sqrt_val = 2.0f * sqrtf(1.0f - t_val);
            v[i] = xi1 * sqrt_val;
            w[i] = xi2 * sqrt_val;

            weight[i] *= albedo;
            
            if (weight[i] < 0.001f) {
                if (rands[i+BLOCK_SIZE] > 0.1f) {
                    if (simPerTrack[i]>0) simPerTrack[i]--;
                    x[i] = simPerTrack[i] ? 0.0f : x[i];
                    y[i] = simPerTrack[i] ? 0.0f : y[i];
                    z[i] = simPerTrack[i] ? 0.0f : z[i];
                    u[i] = simPerTrack[i] ? 0.0f : u[i];
                    v[i] = simPerTrack[i] ? 0.0f : v[i];
                    w[i] = simPerTrack[i] ? 1.0f : w[i];
                    weight[i] = simPerTrack[i] ? 1.0f : weight[i];

                } else {
                    weight[i] = simPerTrack[i] ? weight[i] * 10.0f : weight[i];
                }
            }
        }
        // se vectoriza el siguiente for 32 byte
        // ICX se vectoriza el siguiente for
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if(simPerTrack[i])
                hasToSim++;
        }
    }

    // no se vectoriza el siguiente for
    // ICX no se vectoriza el siguiente for
    for (unsigned int j = 0; j < update_count; j++) {
        heats[shell_update[j]] += heat_update[j];
        heats_squared[shell_update[j]] += heat_update[j]*heat_update[j];
    }
}

void photon_vectorized_prueba_v10_all_int(float *__restrict__ heats, float *__restrict__ heats_squared, unsigned int simulationCount) {
    float x[BLOCK_SIZE]      __attribute__((aligned(64))) = {0.0f};
    float y[BLOCK_SIZE]      __attribute__((aligned(32))) = {0.0f};
    float z[BLOCK_SIZE]      __attribute__((aligned(32))) = {0.0f};
    float u[BLOCK_SIZE]      __attribute__((aligned(32))) = {0.0f};
    float v[BLOCK_SIZE]      __attribute__((aligned(32))) = {0.0f};
    float w[BLOCK_SIZE]      __attribute__((aligned(32)));
    float weight[BLOCK_SIZE] __attribute__((aligned(32)));
    unsigned int simPerTrack[BLOCK_SIZE] __attribute__((aligned(32)));

    // Se vectoriza el siguiente for 32 byte
    // ICX vectoriza el siguiente for 32 byte
    for (int i = 0; i < BLOCK_SIZE; i++) {
        simPerTrack[i] = simulationCount/BLOCK_SIZE;
        w[i] = 1.0f;
        weight[i] = 1.0f;
    }

    float heat_update[MAGIC_N] __attribute__((aligned(64)));
    unsigned int shell_update[MAGIC_N] __attribute__((aligned(32)));
    unsigned int update_count = 0;

    const float albedo = MU_S / (MU_S + MU_A);
    const float shells_per_mfp = 1e4f / MICRONS_PER_SHELL / (MU_A + MU_S);

    unsigned int hasToSim = 1;
    while (hasToSim > 0) {
        float rands[BLOCK_SIZE*4] __attribute__((aligned(64)));
        next_float_vector_4_times_block(rands);
        float r[BLOCK_SIZE] __attribute__((aligned(64)));
        float h[BLOCK_SIZE] __attribute__((aligned(32)));
        unsigned int shell[BLOCK_SIZE] __attribute__((aligned(32)));
        // Se vectoriza el siguiente for 32 byte
        // ICX Se vectoriza el siguiente for 32 byte
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float t = -logf(rands[i]);
            x[i] += t * u[i];
            y[i] += t * v[i];
            z[i] += t * w[i];
            r[i] = sqrtf(x[i]*x[i] + y[i]*y[i] + z[i]*z[i]);
            h[i] = (1.0f - albedo) * weight[i];
        }
        // Se vectoriza el siguiente for 32 byte
        // ICX se vectoriza el siguiente for
        for (int i = 0; i < BLOCK_SIZE; i++) {
            unsigned int s = (unsigned int)(r[i] * shells_per_mfp);
            shell[i] = (s >= SHELLS) ? SHELLS-1 : s;
        }

        if(update_count+hasToSim > MAGIC_N) {
            // no se vectoriza el siguiente for
            // ICX no se vectoriza el siguiente for
            for (unsigned int j = 0; j < update_count; j++) {
                heats[shell_update[j]] += heat_update[j];
                heats_squared[shell_update[j]] += heat_update[j]*heat_update[j];
            }
            update_count = 0;
        }

        if (hasToSim == BLOCK_SIZE) {
            // gcc ??????
            // ICX se vectoriza el siguiente for
            for (unsigned int i = 0; i < BLOCK_SIZE; i++) {
                shell_update[update_count+i] = shell[i];
                heat_update[update_count+i] = h[i];
            }
            update_count+=8;
        } else {
            // gcc no se vectoriza el siguiente for
            // ICX no se vectoriza el siguiente for
            for (unsigned int i = 0; i < BLOCK_SIZE; i++) {
                if (simPerTrack[i]) {
                    shell_update[update_count] = shell[i];
                    heat_update[update_count] = h[i];
                    update_count++;
                }
            }
        }

        hasToSim = 0;
        // Se vectoriza el siguiente for 32 byte
        // ICX se vecotoriza el siguiente for
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float t_val = rands[i+BLOCK_SIZE*2];
            float r = sqrtf(t_val);
            float theta = 2.0f * PI * rands[i+BLOCK_SIZE*3];
            float xi1 = r * cosf(theta);
            float xi2 = r * sinf(theta);
            u[i] = 2.0f * t_val - 1.0f;
            float sqrt_val = 2.0f * sqrtf(1.0f - t_val);
            v[i] = xi1 * sqrt_val;
            w[i] = xi2 * sqrt_val;

            weight[i] *= albedo;
            
            if (weight[i] < 0.001f) {
                if (rands[i+BLOCK_SIZE] > 0.1f) {
                    if (simPerTrack[i]>0) simPerTrack[i]--;
                    x[i] = simPerTrack[i] ? 0.0f : x[i];
                    y[i] = simPerTrack[i] ? 0.0f : y[i];
                    z[i] = simPerTrack[i] ? 0.0f : z[i];
                    u[i] = simPerTrack[i] ? 0.0f : u[i];
                    v[i] = simPerTrack[i] ? 0.0f : v[i];
                    w[i] = simPerTrack[i] ? 1.0f : w[i];
                    weight[i] = simPerTrack[i] ? 1.0f : weight[i];

                } else {
                    weight[i] = simPerTrack[i] ? weight[i] * 10.0f : weight[i];
                }
            }
        }
        // se vectoriza el siguiente for 32 byte
        // ICX se vectoriza el siguiente for
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if(simPerTrack[i])
                hasToSim++;
        }
    }

    // no se vectoriza el siguiente for
    // ICX no se vectoriza el siguiente for
    for (unsigned int j = 0; j < update_count; j++) {
        heats[shell_update[j]] += heat_update[j];
        heats_squared[shell_update[j]] += heat_update[j]*heat_update[j];
    }
}

void photon_vectorized_prueba_v11_all_short(float *__restrict__ heats, float *__restrict__ heats_squared, unsigned int simulationCount) {
    float x[BLOCK_SIZE]      __attribute__((aligned(64))) = {0.0f};
    float y[BLOCK_SIZE]      __attribute__((aligned(32))) = {0.0f};
    float z[BLOCK_SIZE]      __attribute__((aligned(32))) = {0.0f};
    float u[BLOCK_SIZE]      __attribute__((aligned(32))) = {0.0f};
    float v[BLOCK_SIZE]      __attribute__((aligned(32))) = {0.0f};
    float w[BLOCK_SIZE]      __attribute__((aligned(32)));
    float weight[BLOCK_SIZE] __attribute__((aligned(32)));
    unsigned int simPerTrack[BLOCK_SIZE] __attribute__((aligned(32)));

    // Se vectoriza el siguiente for 32 byte
    // ICX vectoriza el siguiente for 32 byte
    for (int i = 0; i < BLOCK_SIZE; i++) {
        simPerTrack[i] = simulationCount/BLOCK_SIZE;
        w[i] = 1.0f;
        weight[i] = 1.0f;
    }

    float heat_update[MAGIC_N] __attribute__((aligned(64)));
    unsigned short shell_update[MAGIC_N] __attribute__((aligned(32)));
    unsigned short update_count = 0;

    const float albedo = MU_S / (MU_S + MU_A);
    const float shells_per_mfp = 1e4f / MICRONS_PER_SHELL / (MU_A + MU_S);

    unsigned short hasToSim = 1;
    while (hasToSim > 0) {
        float rands[BLOCK_SIZE*4] __attribute__((aligned(64)));
        next_float_vector_4_times_block(rands);
        float r[BLOCK_SIZE] __attribute__((aligned(64)));
        float h[BLOCK_SIZE] __attribute__((aligned(32)));
        unsigned short shell[BLOCK_SIZE] __attribute__((aligned(32)));
        // Se vectoriza el siguiente for 32 byte
        // ICX Se vectoriza el siguiente for 32 byte
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float t = -logf(rands[i]);
            x[i] += t * u[i];
            y[i] += t * v[i];
            z[i] += t * w[i];
            r[i] = sqrtf(x[i]*x[i] + y[i]*y[i] + z[i]*z[i]);
            h[i] = (1.0f - albedo) * weight[i];
        }
        // Se vectoriza el siguiente for 16 byte
        // ICX se vectoriza el siguiente for
        for (int i = 0; i < BLOCK_SIZE; i++) {
            unsigned short s = (unsigned short)(r[i] * shells_per_mfp);
            shell[i] = (s >= SHELLS) ? SHELLS-1 : s;
        }

        if(update_count+hasToSim > MAGIC_N) {
            // no se vectoriza el siguiente for
            // ICX no se vectoriza el siguiente for
            for (unsigned int j = 0; j < update_count; j++) {
                heats[shell_update[j]] += heat_update[j];
                heats_squared[shell_update[j]] += heat_update[j]*heat_update[j];
            }
            update_count = 0;
        }

        if (hasToSim == BLOCK_SIZE) {
            // gcc se vectoriza el siguiente for 16 byte
            // ICX se vectoriza el siguiente for
            for (unsigned int i = 0; i < BLOCK_SIZE; i++) {
                shell_update[update_count+i] = shell[i];
                heat_update[update_count+i] = h[i];
            }
            update_count+=8;
        } else {
            // gcc no se vectoriza el siguiente for
            // ICX no se vectoriza el siguiente for
            for (unsigned int i = 0; i < BLOCK_SIZE; i++) {
                if (simPerTrack[i]) {
                    shell_update[update_count] = shell[i];
                    heat_update[update_count] = h[i];
                    update_count++;
                }
            }
        }

        hasToSim = 0;
        // Se vectoriza el siguiente for 32 byte
        // ICX se vecotoriza el siguiente for
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float t_val = rands[i+BLOCK_SIZE*2];
            float r = sqrtf(t_val);
            float theta = 2.0f * PI * rands[i+BLOCK_SIZE*3];
            float xi1 = r * cosf(theta);
            float xi2 = r * sinf(theta);
            u[i] = 2.0f * t_val - 1.0f;
            float sqrt_val = 2.0f * sqrtf(1.0f - t_val);
            v[i] = xi1 * sqrt_val;
            w[i] = xi2 * sqrt_val;

            weight[i] *= albedo;
            
            if (weight[i] < 0.001f) {
                if (rands[i+BLOCK_SIZE] > 0.1f) {
                    if (simPerTrack[i]>0) simPerTrack[i]--;
                    x[i] = simPerTrack[i] ? 0.0f : x[i];
                    y[i] = simPerTrack[i] ? 0.0f : y[i];
                    z[i] = simPerTrack[i] ? 0.0f : z[i];
                    u[i] = simPerTrack[i] ? 0.0f : u[i];
                    v[i] = simPerTrack[i] ? 0.0f : v[i];
                    w[i] = simPerTrack[i] ? 1.0f : w[i];
                    weight[i] = simPerTrack[i] ? 1.0f : weight[i];

                } else {
                    weight[i] = simPerTrack[i] ? weight[i] * 10.0f : weight[i];
                }
            }
        }
        // se vectoriza el siguiente for 16 byte
        // ICX se vectoriza el siguiente for
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if(simPerTrack[i])
                hasToSim++;
        }
    }

    // no se vectoriza el siguiente for
    // ICX no se vectoriza el siguiente for
    for (unsigned int j = 0; j < update_count; j++) {
        heats[shell_update[j]] += heat_update[j];
        heats_squared[shell_update[j]] += heat_update[j]*heat_update[j];
    }
}

void photon_vectorized_prueba_v12_arrays_for_lanes(float *__restrict__ heats, float *__restrict__ heats_squared, unsigned int simulationCount) {
    float x[BLOCK_SIZE]      __attribute__((aligned(64))) = {0.0f};
    float y[BLOCK_SIZE]      __attribute__((aligned(32))) = {0.0f};
    float z[BLOCK_SIZE]      __attribute__((aligned(32))) = {0.0f};
    float u[BLOCK_SIZE]      __attribute__((aligned(32))) = {0.0f};
    float v[BLOCK_SIZE]      __attribute__((aligned(32))) = {0.0f};
    float w[BLOCK_SIZE]      __attribute__((aligned(32)));
    float weight[BLOCK_SIZE] __attribute__((aligned(32)));
    unsigned int simPerTrack[BLOCK_SIZE] __attribute__((aligned(32)));

    // Se vectoriza el siguiente for 32 byte
    // ICX vectoriza el siguiente for 32 byte
    for (int i = 0; i < BLOCK_SIZE; i++) {
        simPerTrack[i] = simulationCount/BLOCK_SIZE;
        w[i] = 1.0f;
        weight[i] = 1.0f;
    }

    const float albedo = MU_S / (MU_S + MU_A);
    const float shells_per_mfp = 1e4f / MICRONS_PER_SHELL / (MU_A + MU_S);

    static float heatTrack[BLOCK_SIZE][SHELLS] __attribute__((aligned(64)));
    static float heat2Track[BLOCK_SIZE][SHELLS] __attribute__((aligned(64)));

    unsigned int hasToSim = 1;
    while (hasToSim > 0) {
        float rands[BLOCK_SIZE*4] __attribute__((aligned(64)));
        next_float_vector_4_times_block(rands);
        float r[BLOCK_SIZE] __attribute__((aligned(64)));
        float h[BLOCK_SIZE] __attribute__((aligned(32)));
        unsigned short shell[BLOCK_SIZE] __attribute__((aligned(32)));
        // Se vectoriza el siguiente for 32 byte
        // ICX Se vectoriza el siguiente for 32 byte
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float t = -logf(rands[i]);
            x[i] += t * u[i];
            y[i] += t * v[i];
            z[i] += t * w[i];
            r[i] = sqrtf(x[i]*x[i] + y[i]*y[i] + z[i]*z[i]);
            h[i] = (1.0f - albedo) * weight[i];
        }
        // Se vectoriza el siguiente for 16 byte
        // ICX se vectoriza el siguiente for
        for (int i = 0; i < BLOCK_SIZE; i++) {
            unsigned short s = (unsigned short)(r[i] * shells_per_mfp);
            shell[i] = (s >= SHELLS) ? SHELLS-1 : s;
        }

        for (int i = 0; i < BLOCK_SIZE; i++) {
            if (simPerTrack[i]) {
                heatTrack[i][shell[i]] += h[i];
                heat2Track[i][shell[i]] += h[i]*h[i];
            }
        }

        hasToSim = 0;
        // Se vectoriza el siguiente for 32 byte
        // ICX se vecotoriza el siguiente for
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float t_val = rands[i+BLOCK_SIZE*2];
            float r = sqrtf(t_val);
            float theta = 2.0f * PI * rands[i+BLOCK_SIZE*3];
            float xi1 = r * cosf(theta);
            float xi2 = r * sinf(theta);
            u[i] = 2.0f * t_val - 1.0f;
            float sqrt_val = 2.0f * sqrtf(1.0f - t_val);
            v[i] = xi1 * sqrt_val;
            w[i] = xi2 * sqrt_val;

            weight[i] *= albedo;
            
            if (weight[i] < 0.001f) {
                if (rands[i+BLOCK_SIZE] > 0.1f) {
                    if (simPerTrack[i]>0) simPerTrack[i]--;
                    x[i] = simPerTrack[i] ? 0.0f : x[i];
                    y[i] = simPerTrack[i] ? 0.0f : y[i];
                    z[i] = simPerTrack[i] ? 0.0f : z[i];
                    u[i] = simPerTrack[i] ? 0.0f : u[i];
                    v[i] = simPerTrack[i] ? 0.0f : v[i];
                    w[i] = simPerTrack[i] ? 1.0f : w[i];
                    weight[i] = simPerTrack[i] ? 1.0f : weight[i];

                } else {
                    weight[i] = simPerTrack[i] ? weight[i] * 10.0f : weight[i];
                }
            }
        }
        // se vectoriza el siguiente for 32 byte
        // ICX se vectoriza el siguiente for
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if(simPerTrack[i])
                hasToSim++;
        }
    }

    // for (int i = 0; i < SHELLS; i++) {
    //     for (int j = 0; j < BLOCK_SIZE; j++) {
    //         heats[i] += heatTrack[j][i];
    //         heats_squared[i] += heat2Track[j][i];
    //     }
    // }

    for (int j = 0; j < BLOCK_SIZE; j++) {
        for (int i = 0; i < SHELLS; i++) {
            heats[i] += heatTrack[j][i];
            heats_squared[i] += heat2Track[j][i];
        }
    }
}

 /***
  * Main matter
  ***/
 int main()
 {
    seed_vector((uint64_t) SEED);
    double time = omp_get_wtime();

    // time = omp_get_wtime();
    // for (unsigned int i = 0; i < PHOTONS; ++i) {
    //     photon_lab1(heat, heat2);
    // }
	// printf("photon: %lf\n", (omp_get_wtime()-time));

    // time = omp_get_wtime();
    // for (unsigned int i = 0; i < PHOTONS/BLOCK_SIZE; ++i) {
    //     photon_vectorized_lab2(heat, heat2);
    // }
    // printf("photon vectorized lab2: %lf\n", (omp_get_wtime()-time));

    // time = omp_get_wtime();
    // for (unsigned int i = 0; i < PHOTONS/BLOCK_SIZE; ++i) {
    //     photon_vectorized_lab2_v2(heat, heat2);
    // }
    // printf("photon vectorized lab2 v2: %lf\n", (omp_get_wtime()-time));

    // time = omp_get_wtime();
    // for (unsigned int i = 0; i < PHOTONS/BLOCK_SIZE; ++i) {
    //     photon_vectorized_mejora_v2(heat, heat2);
    // }
    // printf("photon vectorized mejora v2: %lf\n", (omp_get_wtime()-time));

    // time = omp_get_wtime();
    // photon_vectorized_prueba_v6(heat, heat2, PHOTONS);
    // printf("photon vectorized prueba v6: %lf\n", (omp_get_wtime()-time));

    // time = omp_get_wtime();
    // photon_vectorized_prueba_v7(heat, heat2, PHOTONS);
    // printf("photon vectorized prueba v7: %lf\n", (omp_get_wtime()-time));

    // time = omp_get_wtime();
    // photon_vectorized_prueba_v8(heat, heat2, PHOTONS);
    // printf("photon vectorized prueba v8: %lf\n", (omp_get_wtime()-time));

    time = omp_get_wtime();
    photon_vectorized_prueba_v9(heat, heat2, PHOTONS);
    printf("photon vectorized prueba v9: %lf\n", (omp_get_wtime()-time));

    // time = omp_get_wtime();
    // photon_vectorized_prueba_v10_all_int(heat, heat2, PHOTONS);
    // printf("photon vectorized prueba v10 all int: %lf\n", (omp_get_wtime()-time));

    // time = omp_get_wtime();
    // photon_vectorized_prueba_v11_all_short(heat, heat2, PHOTONS);
    // printf("photon vectorized prueba v11 all short: %lf\n", (omp_get_wtime()-time));

    // time = omp_get_wtime();
    // photon_vectorized_prueba_v12_arrays_for_lanes(heat, heat2, PHOTONS);
    // printf("photon vectorized prueba v12 arrays for lanes: %lf\n", (omp_get_wtime()-time));

    // printf("# Radius\tHeat\n");
    // printf("# [microns]\t[W/cm^3]\tError\n");
    // float t = 4.0f * M_PI * powf(MICRONS_PER_SHELL, 3.0f) * PHOTONS / 1e12;
    // for (unsigned int i = 0; i < SHELLS - 1; ++i) {
    //     printf("%6.0f\t%12.5f\t%12.5f\n", i * (float)MICRONS_PER_SHELL,
    //         heat[i] / t / (i * i + i + 1.0 / 3.0),
    //         sqrt(heat2[i] - heat[i] * heat[i] / PHOTONS) / t / (i * i + i + 1.0f / 3.0f));
    // }
    // printf("# extra\t%12.5f\n", heat[SHELLS - 1] / PHOTONS);

    return (int)heat[0] + (int) heat2[0];
 }