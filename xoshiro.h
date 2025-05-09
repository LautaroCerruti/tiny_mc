#ifndef XOSHIRO_H
#define XOSHIRO_H

#include <stdint.h>
// Xoshiro256+ utilizado para generar numeros de coma flotante en el rango [0,1).

// Inicializa el generador con la semilla proporcionada.
void seed_vector(uint64_t seed_val);
void seed_vector_omp(uint64_t seed_val, int nthreads);

// Devuelve el siguiente número aleatorio en punto flotante en el rango [0,1).
float next_float(void);

void next_float_vector_4_times_block(float *array1);
void next_float_vector_4_times_block_omp(float *array1, int tid);

#endif // XOSHIRO_H
