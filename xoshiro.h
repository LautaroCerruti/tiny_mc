#ifndef XOSHIRO_H
#define XOSHIRO_H

#include <stdint.h>
// Xoshiro256+ utilizado para generar numeros de coma flotante en el rango [0,1).

// Inicializa el generador con la semilla proporcionada.
void seed(uint64_t seed_val);

// Devuelve el siguiente número aleatorio en punto flotante en el rango [0,1).
float next_float(void);

void next_two_floats(float *f1, float *f2);

// Funciones de salto para uso en computación paralela (si son necesarias).
void jump(void);
void long_jump(void);

#endif // XOSHIRO_H
