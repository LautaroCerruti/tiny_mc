/* Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)
 * 1 W Point Source Heating in Infinite Isotropic Scattering Medium
 * http://omlc.ogi.edu/software/mc/tiny_mc.c
 *
 * Adaptado para CP2014, Nicolas Wolovick
 */

#define _GNU_SOURCE
#define _XOPEN_SOURCE 500 // M_PI

#include "params.h"
#include "photon.h"
#include "wtime.h"
#include "xoshiro.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <string.h>
#include <errno.h>
#include <omp.h>
 

char t1[] = "Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)";
char t2[] = "1 W Point Source Heating in Infinite Isotropic Scattering Medium";
char t3[] = "CPU version, adapted for PEAGPGPU by Gustavo Castellano"
            " and Nicolas Wolovick";

// global state, heat and heat square in each shell
static float heat[SHELLS] __attribute__((aligned(64)));
static float heat2[SHELLS] __attribute__((aligned(64)));
 
int write_stat_file(const char *filename, double elapsed) {
    FILE *csvFile = fopen(filename, "r");
    if (csvFile == NULL) {
        csvFile = fopen(filename, "w");
        if (csvFile == NULL) {
            fprintf(stderr, "Error opening file %s\n", filename);
            return 1;
        }
        fprintf(csvFile, "photons,time,pus\n");
    } else {
        fclose(csvFile);
        csvFile = fopen(filename, "a");
    }
    fprintf(csvFile, "%i, %lf, %lf\n", PHOTONS, elapsed, PHOTONS / (elapsed * 1e6));
    fclose(csvFile);
    return 0;
}
 
/***
 * Main matter
 ***/
int main(int argc, char *argv[])
{
    int  nthreads = omp_get_max_threads();
    seed_vector((uint64_t) SEED, nthreads);
    // Variables para la línea de comandos
    const char *output_filename = "resultados.csv";
    int verbose = 1;  // 2: imprimir todo, 1: imprimir result tiempo, 0: modo quiet

    int opt;
    // Se reconocen las opciones -o para archivo y -q para modo silencioso
    while ((opt = getopt(argc, argv, "o:q")) != -1) {
        switch(opt) {
            case 'o':
                output_filename = optarg;
                break;
            case 'q':
                verbose = 0;
                break;
            default:
                fprintf(stderr, "Uso: %s [-o archivo_salida] [-q]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    // Impresión de cabecera si verbose está activado
    if (verbose==2) {
        printf("# %s\n# %s\n# %s\n", t1, t2, t3);
        printf("# Scattering = %8.3f/cm\n", MU_S);
        printf("# Absorption = %8.3f/cm\n", MU_A);
        printf("# Photons    = %8d\n#\n", PHOTONS);
    }

    double start = wtime();
    #pragma omp parallel for schedule(dynamic,1) reduction(+: heat[0:SHELLS], heat2[0:SHELLS])
    for (unsigned int b = 0; b < PHOTONS_BLOCK_COUNT; ++b) {
        photon_vectorized(heat, heat2, PHOTONS / PHOTONS_BLOCK_COUNT);
    }
    double end = wtime();

    assert(start <= end);
    double elapsed = end - start;
 
    if (verbose) {
        printf("# %lf seconds\n", elapsed);
        printf("# %lf photons per microseconds\n", PHOTONS / (elapsed * 1e6));
    }

    write_stat_file(output_filename, elapsed);

    if (verbose==2) {
        printf("# Radius\tHeat\n");
        printf("# [microns]\t[W/cm^3]\tError\n");
        float t = 4.0f * M_PI * powf(MICRONS_PER_SHELL, 3.0f) * PHOTONS / 1e12;
        for (unsigned int i = 0; i < SHELLS - 1; ++i) {
            printf("%6.0f\t%12.5f\t%12.5f\n", i * (float)MICRONS_PER_SHELL,
                heat[i] / t / (i * i + i + 1.0 / 3.0),
                sqrt(heat2[i] - heat[i] * heat[i] / PHOTONS) / t / (i * i + i + 1.0f / 3.0f));
        }
        printf("# extra\t%12.5f\n", heat[SHELLS - 1] / PHOTONS);
    }

    return 0;
}
 