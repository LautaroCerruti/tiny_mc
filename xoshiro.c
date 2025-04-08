#include "xoshiro.h"
#include <stdint.h>

// Función auxiliar: rotación a la izquierda.
static inline uint64_t rotl(const uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}

// Estado interno del generador.
static uint64_t s[4];

// Función interna para generar el siguiente número aleatorio (entero de 64 bits).
uint64_t next(void) {
	const uint64_t result = s[0] + s[3];

	const uint64_t t = s[1] << 17;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];

	s[2] ^= t;

	s[3] = rotl(s[3], 45);

	return result;
}

// Función para saltar en la secuencia (equivalente a 2^128 llamadas a next()).
void jump(void) {
	static const uint64_t JUMP[] = { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c };

	uint64_t s0 = 0;
	uint64_t s1 = 0;
	uint64_t s2 = 0;
	uint64_t s3 = 0;
	for(unsigned long i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
		for(int b = 0; b < 64; b++) {
			if (JUMP[i] & UINT64_C(1) << b) {
				s0 ^= s[0];
				s1 ^= s[1];
				s2 ^= s[2];
				s3 ^= s[3];
			}
			next();	
		}
		
	s[0] = s0;
	s[1] = s1;
	s[2] = s2;
	s[3] = s3;
}

// Función para un salto largo en la secuencia (equivalente a 2^192 llamadas a next()).
void long_jump(void) {
	static const uint64_t LONG_JUMP[] = { 0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241, 0x39109bb02acbe635 };

	uint64_t s0 = 0;
	uint64_t s1 = 0;
	uint64_t s2 = 0;
	uint64_t s3 = 0;
	for(unsigned long i = 0; i < sizeof LONG_JUMP / sizeof *LONG_JUMP; i++)
		for(int b = 0; b < 64; b++) {
			if (LONG_JUMP[i] & UINT64_C(1) << b) {
				s0 ^= s[0];
				s1 ^= s[1];
				s2 ^= s[2];
				s3 ^= s[3];
			}
			next();	
		}
		
	s[0] = s0;
	s[1] = s1;
	s[2] = s2;
	s[3] = s3;
}

// ------------------------
// Código de splitmix64 para semilla
// ------------------------

static uint64_t splitmix64_state;

static uint64_t splitmix64_next(void) {
    uint64_t z = (splitmix64_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Inicializa el generador xoshiro256+ usando splitmix64.
void seed(uint64_t seed_val) {
    splitmix64_state = seed_val;
    s[0] = splitmix64_next();
    s[1] = splitmix64_next();
    s[2] = splitmix64_next();
    s[3] = splitmix64_next();
}

// Devuelve un número aleatorio en punto flotante en el rango [0,1).
float next_float() {
    // Extraemos 24 bits de la salida de 64 bits:
    // (64 - 24 = 40, desplazamos 40 bits a la derecha)
    return (next() >> 40) * (1.0f / (1U << 24));
}

void next_two_floats(float *f1, float *f2) {
    uint64_t r = next();
    *f1 = (r >> 40) * (1.0f / (1U << 24));          // Primeros 24 bits de la mitad alta
    *f2 = ((r >> 8) & 0xFFFFFF) * (1.0f / (1U << 24)); // Primeros 24 bits de la mitad baja
}