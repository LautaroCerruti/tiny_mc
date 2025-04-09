#include <math.h>
#include <stdlib.h>

#include "params.h"
#include "xoshiro.h"

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

    float aux = (1.0f / (1U << 24));
    for (;;) {
        uint64_t r1 = next();
        float f1 = (r1 >> 40) * aux;
        float t = -logf(f1); /* move */
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
            uint64_t r2 = next();
            float xf1 = (r2 >> 40) * aux;
            float xf2 = ((r2 >> 8) & 0xFFFFFF) * aux;
            xi1 = 2.0f * xf1 - 1.0f;
            xi2 = 2.0f * xf2 - 1.0f;
            t = xi1 * xi1 + xi2 * xi2;
        } while (1.0f < t);
        u = 2.0f * t - 1.0f;
        v = xi1 * sqrtf((1.0f - u * u) / t);
        w = xi2 * sqrtf((1.0f - u * u) / t);

        if (weight < 0.001f) { /* roulette */
            float f2 = ((r1 >> 8) & 0xFFFFFF) * aux;
            if (f2 > 0.1f)
                break;
            weight /= 0.1f;
        }
    }
}
