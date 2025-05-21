#include "xoshiro.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "params.h"

// Función auxiliar: rotación a la izquierda.
static inline uint64_t rotl(const uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}

// Estado interno del generador.
static uint64_t s[MAX_THREADS][4][BLOCK_SIZE] __attribute__((aligned(64)));

// Función interna para generar el siguiente número aleatorio (entero de 64 bits).
void next_vector(uint64_t *array, int n) {
    // Procesamos bloques de XOSHIRO256_UNROLL números.
    for (int b = 0; b < n; b += BLOCK_SIZE) {
        // Se calcula el resultado para cada lane: suma de s[0] y s[3].
        for (int i = 0; i < BLOCK_SIZE; i++) {
            array[b + i] = s[0][0][i] + s[0][3][i];
        }

        // Se almacena temporalmente el valor t = s[1] << 17 para cada lane.
        uint64_t t[BLOCK_SIZE];
        for (int i = 0; i < BLOCK_SIZE; i++) {
            t[i] = s[0][1][i] << 17;
        }

        // Actualización del estado, siguiendo el mismo orden que en la versión escalar:
        for (int i = 0; i < BLOCK_SIZE; i++) {
            s[0][2][i] ^= s[0][0][i];
        }
        for (int i = 0; i < BLOCK_SIZE; i++) {
            s[0][3][i] ^= s[0][1][i];
        }
        for (int i = 0; i < BLOCK_SIZE; i++) {
            s[0][1][i] ^= s[0][2][i];
        }
        for (int i = 0; i < BLOCK_SIZE; i++) {
            s[0][0][i] ^= s[0][3][i];
        }
        for (int i = 0; i < BLOCK_SIZE; i++) {
            s[0][2][i] ^= t[i];
        }
        for (int i = 0; i < BLOCK_SIZE; i++) {
            s[0][3][i] = rotl(s[0][3][i], 45);
        }
    }
}

void next_float_vector_4_times_block(float *array1, int tid) {
    uint64_t temp[BLOCK_SIZE*2] __attribute__((aligned(64)));
    for (int b = 0; b < BLOCK_SIZE*2; b += BLOCK_SIZE) {
        for (int i = 0; i < BLOCK_SIZE; i++) {
            temp[i] = s[tid][0][i] + s[tid][3][i];
        }
        
        uint64_t t[BLOCK_SIZE];
        for (int i = 0; i < BLOCK_SIZE; i++) {
            t[i] = s[tid][1][i] << 17;
        }
        
        for (int i = 0; i < BLOCK_SIZE; i++) {
            s[tid][2][i] ^= s[tid][0][i];
        }
        for (int i = 0; i < BLOCK_SIZE; i++) {
            s[tid][3][i] ^= s[tid][1][i];
        }
        for (int i = 0; i < BLOCK_SIZE; i++) {
            s[tid][1][i] ^= s[tid][2][i];
        }
        for (int i = 0; i < BLOCK_SIZE; i++) {
            s[tid][0][i] ^= s[tid][3][i];
        }
        for (int i = 0; i < BLOCK_SIZE; i++) {
            s[tid][2][i] ^= t[i];
        }
        for (int i = 0; i < BLOCK_SIZE; i++) {
            s[tid][3][i] = rotl(s[tid][3][i], 45);
        }
        
        const float scale = 1.0f / (1U << 24);
		for (int i = 0; i < BLOCK_SIZE; i++) {
            array1[b + i] = (temp[i] >> 40) * scale;
            array1[b + i + BLOCK_SIZE*2] = ((temp[i] >> 8) & 0xFFFFFFULL) * scale;
        }
    }
}

float next_float(void) {
    // Se calcula el resultado para la lane 0.
    uint64_t result = s[0][0][0] + s[0][3][0];
    uint64_t t = s[0][1][0] << 17;
    
    s[0][2][0] ^= s[0][0][0];
    s[0][3][0] ^= s[0][1][0];
    s[0][1][0] ^= s[0][2][0];
    s[0][0][0] ^= s[0][3][0];
    
    s[0][2][0] ^= t;
    s[0][3][0] = rotl(s[0][3][0], 45);
    
    // Se extraen 24 bits del resultado y se escala para obtener un float en [0, 1).
    return (result >> 40) * (1.0f / (1U << 24));
}

// ------------------------
// Código de splitmix64 para semilla
// ------------------------

static inline uint64_t splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

void seed_vector(uint64_t seed_val, int nthreads) {
    // Inicializa el generador xoshiro256+ usando splitmix64.
    uint64_t states[nthreads][BLOCK_SIZE];
    // Se inicializan los estados para cada hilo.
    int offset = 0;
    for (int i = 0; i < nthreads; i++) {
        for (int j = 0; j < BLOCK_SIZE; j++) {
            states[i][j] = seed_val + offset;  // Semilla distinta por lane
            s[i][0][j] = splitmix64_next(&states[i][j]);
            s[i][1][j] = splitmix64_next(&states[i][j]);
            s[i][2][j] = splitmix64_next(&states[i][j]);
            s[i][3][j] = splitmix64_next(&states[i][j]);
            offset++;
        }
    }
}
