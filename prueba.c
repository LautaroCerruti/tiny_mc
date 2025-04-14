#include <math.h>
#include <stdlib.h>
#include <stdio.h> // printf()
#include <omp.h> // omp_get_wtime()

#include "params.h"
#include "xoshiro.h"

static float heat[SHELLS];
static float heat2[SHELLS];

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

void photon_vectorized(float* heats, float* heats_squared) {
    // Declaramos arrays en la pila usando VLA
    float x[BLOCK_SIZE] = {0.0f};
    float y[BLOCK_SIZE] = {0.0f};
    float z[BLOCK_SIZE] = {0.0f};
    float u[BLOCK_SIZE] = {0.0f};
    float v[BLOCK_SIZE] = {0.0f};
    float w[BLOCK_SIZE];
    float weight[BLOCK_SIZE];
    int active[BLOCK_SIZE];

    unsigned int shell[BLOCK_SIZE];
    // Inicialización de estados: se asume que todos los fotones parten en el origen con dirección (0,0,1)
    // Se vectoriza el siguiente for
    for (int i = 0; i < BLOCK_SIZE; i++) {
        w[i] = 1.0f;
        weight[i] = 1.0f;
        active[i] = 1;   // 1 indica fotón activo
    }

    const float albedo = MU_S / (MU_S + MU_A);
    const float shells_per_mfp = 1e4f / MICRONS_PER_SHELL / (MU_A + MU_S);

    int photons_remaining = BLOCK_SIZE;
    while (photons_remaining > 0) {
        photons_remaining = 0;
        float t[BLOCK_SIZE];
        // Paso 1: Movimiento y absorción
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if (active[i]) {
                t[i] = -logf(next_float());  // Longitud de paso, usando el generador actual
            }
        }
        // se vectoriza el siguiente for
        for (int i = 0; i < BLOCK_SIZE; i++) {
            x[i] += t[i] * u[i];
            y[i] += t[i] * v[i];
            z[i] += t[i] * w[i];
            shell[i] = (unsigned int)(sqrtf(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]) * shells_per_mfp);
            if (shell[i] >= SHELLS)
                shell[i] = SHELLS - 1;
        }

        for (int i = 0; i < BLOCK_SIZE; i++) {
            if (active[i]) {
                heats[shell[i]] += (1.0f - albedo) * weight[i];
                heats_squared[shell[i]] += (1.0f - albedo) * (1.0f - albedo) * weight[i] * weight[i];
            }
        }

        // Se vectoriza el siguiente for
        for (int i = 0; i < BLOCK_SIZE; i++) {
            weight[i] *= albedo;
        }

        float t_val[BLOCK_SIZE], xi1[BLOCK_SIZE], xi2[BLOCK_SIZE];
        // Paso 2: Cambio de dirección usando el método de rechazo
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if (active[i]) {
                // Bucle de rechazo: se repite hasta obtener un valor de t_val en (0,1]
                do {
                    xi1[i] = 2.0f * next_float() - 1.0f;
                    xi2[i] = 2.0f * next_float() - 1.0f;
                    t_val[i] = xi1[i] * xi1[i] + xi2[i] * xi2[i];
                } while (t_val[i] > 1.0f || t_val[i] == 0.0f);
            }
        }

        // Se vectoriza el siguiente for
        for (int i = 0; i < BLOCK_SIZE; i++) {
            u[i] = 2.0f * t_val[i] - 1.0f;
            float sqrt_val = sqrtf((1.0f - u[i] * u[i]) / t_val[i]);
            v[i] = xi1[i] * sqrt_val;
            w[i] = xi2[i] * sqrt_val;
        }

        float roulette[BLOCK_SIZE];
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if (active[i])
                roulette[i] = next_float();
        }

        // Se vectoriza el siguiente for
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if (active[i]) {
                if (weight[i] < 0.001f) {
                    if (roulette[i] > 0.1f)
                        active[i] = 0;  // Se desactiva el fotón
                    else {
                        weight[i] /= 0.1f;
                    }
                }
            }
            if(active[i]) 
                photons_remaining++;
        }
    }
}

 /***
  * Main matter
  ***/
 int main()
 {
    seed((uint64_t) SEED);
    double t = omp_get_wtime();
    for (unsigned int i = 0; i < PHOTONS; ++i) {
        photon(heat, heat2);
    }
	printf("phonot: %lf\n", (omp_get_wtime()-t));
    t = omp_get_wtime();
     // simulation
    for (unsigned int i = 0; i < PHOTONS/BLOCK_SIZE; ++i) {
        photon_vectorized(heat, heat2);
    }
    printf("phonot vectorized: %lf\n", (omp_get_wtime()-t));
    return (int)heat[0];
 }