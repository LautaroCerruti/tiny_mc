#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "params.h"
#include "xoshiro.h"

#define likely(x)       __builtin_expect((x),1)
#define unlikely(x)     __builtin_expect((x),0)

typedef struct {
    float x;
    float y;
    float z;
    float u;
    float v;
    float w;
    float weight; 
} Photon;

void photon(float* heats, float* heats_squared)
{
    const float albedo = MU_S / (MU_S + MU_A);
    const float shells_per_mfp = 1e4 / MICRONS_PER_SHELL / (MU_A + MU_S);

    /* launch */
    Photon p = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f};

    //delete later
    // long long randcounter = 0;
    // long long iterations = 0;

    for (;;) {
        // randcounter++;
        // iterations++;
        float i1, i2;
        next_two_floats(&i1, &i2);
        float t = -logf(i1); /* move */
        p.x += t * p.u;
        p.y += t * p.v;
        p.z += t * p.w;

        unsigned int shell = sqrtf(p.x * p.x + p.y * p.y + p.z * p.z) * shells_per_mfp; /* absorb */
        if (shell > SHELLS - 1) {
            shell = SHELLS - 1;
        }
        heats[shell] += (1.0f - albedo) * p.weight;
        heats_squared[shell] += (1.0f - albedo) * (1.0f - albedo) * p.weight * p.weight; /* add up squares */
        p.weight *= albedo;

        /* New direction, rejection method */
        float xi1, xi2;
        do {
            float x1, x2;
            next_two_floats(&x1, &x2);
            //randcounter += 2;
            xi1 = 2.0f * x1 - 1.0f;
            xi2 = 2.0f * x2 - 1.0f;
            t = xi1 * xi1 + xi2 * xi2;
        } while (1.0f < t);
        p.u = 2.0f * t - 1.0f;
        p.v = xi1 * sqrtf((1.0f - p.u * p.u) / t);
        p.w = xi2 * sqrtf((1.0f - p.u * p.u) / t);

        if (unlikely(p.weight < 0.001f)) { /* roulette */
            //randcounter++;
            if (likely(i2 > 0.1f))
                break;
            p.weight *= 10.0f;
        }
    }
    //printf("randcounter: %lld, iterations: %lld \n", randcounter, iterations);
}
