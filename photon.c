#include <math.h>
#include <stdlib.h>

#include "params.h"
#include "xoshiro.h"

#define PI 3.14159265358979323846f

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