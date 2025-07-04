#pragma once

#include <time.h> // time

#ifndef SHELLS
#define SHELLS 101 // discretization level
#endif

#ifndef PHOTONS
#define PHOTONS (1L<<32) // number of photons
#endif

#ifndef MU_A
#define MU_A 2.0f // Absorption Coefficient in 1/cm !!non-zero!!
#endif

#ifndef MU_S
#define MU_S 20.0f // Reduced Scattering Coefficient in 1/cm
#endif

#ifndef MICRONS_PER_SHELL
#define MICRONS_PER_SHELL 50 // Thickness of spherical shells in microns
#endif

#ifndef SEED
#define SEED (time(NULL)) // random seed
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 8
#endif

#ifndef MAGIC_N
#define MAGIC_N 16
#endif

#ifndef MAX_THREADS
#define MAX_THREADS 12
#endif

#ifndef PHOTONS_BLOCK
#define PHOTONS_BLOCK (PHOTONS / 4096)
#endif

#ifndef GPU_BLOCKS
#define GPU_BLOCKS 128
#endif

#ifndef GPU_THREADS
#define GPU_THREADS 512
#endif
