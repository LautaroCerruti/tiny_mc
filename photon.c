#include <math.h>
#include <stdlib.h>
#include <omp.h>

#include "params.h"
#include "xoshiro.h"

#define PI 3.14159265358979323846f

void photon_vectorized(float *__restrict__ heats, float *__restrict__ heats_squared, unsigned int simulationCount) {
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

    int tid = omp_get_thread_num();

    unsigned int hasToSim = BLOCK_SIZE;
    while (hasToSim > 0) {
        float rands[BLOCK_SIZE*4] __attribute__((aligned(64)));
        //next_float_vector_4_times_block(rands);
        next_float_vector_4_times_block_omp(rands, tid);
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
            // gcc ???????????????'
            // ICX se vectoriza el siguiente for
            for (int i = 0; i < BLOCK_SIZE; i++) {
                shell_update[update_count+i] = shell[i];
                heat_update[update_count+i] = h[i];
            }
            update_count+=BLOCK_SIZE;
        } else {
            // gcc no se vectoriza el siguiente for
            // ICX no se vectoriza el siguiente for
            for (int i = 0; i < BLOCK_SIZE; i++) {
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