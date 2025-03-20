/* Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)"
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

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>
#include <string.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <errno.h>

char t1[] = "Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)";
char t2[] = "1 W Point Source Heating in Infinite Isotropic Scattering Medium";
char t3[] = "CPU version, adapted for PEAGPGPU by Gustavo Castellano"
            " and Nicolas Wolovick";


// global state, heat and heat square in each shell
static float heat[SHELLS];
static float heat2[SHELLS];

int write_stat_file(double elapsed, long long count) {
    const char *filename = "resultados.csv";
    FILE *csvFile;

    csvFile = fopen(filename, "r");
    if (csvFile == NULL) {
        csvFile = fopen(filename, "w");

        if (csvFile == NULL) {
            fprintf(stderr, "Error opening file\n");
            return 1;
        }
        fprintf(csvFile, "photons,time,pns,ins\n");
    } else {
        fclose(csvFile);
        csvFile = fopen(filename, "a");
    }

    fprintf(csvFile, "%i, %lf, %lf, %lld\n", PHOTONS, elapsed, PHOTONS / (elapsed * 1e9), count);
    fclose(csvFile);

    return 0;
}

// Función helper para invocar la syscall perf_event_open
static long
perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

void start_perf(int* fd) {
    struct perf_event_attr pe;

    // Inicializamos la estructura de atributos a cero
    memset(&pe, 0, sizeof(struct perf_event_attr));
    pe.type = PERF_TYPE_HARDWARE;
    pe.size = sizeof(struct perf_event_attr);
    pe.config = PERF_COUNT_HW_INSTRUCTIONS;  // Contar instrucciones
    pe.disabled = 1;
    pe.exclude_kernel = 0;  // Excluir código del kernel
    pe.exclude_hv = 0;      // Excluir hipervisor

    // Abrir el contador para el proceso actual (pid = 0) en todos los CPUs (cpu = -1)
    *fd = perf_event_open(&pe, 0, -1, -1, 0);
    if (*fd == -1) {
        fprintf(stderr, "Error al abrir perf_event: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }

    // Reiniciar y habilitar el contador
    ioctl(*fd, PERF_EVENT_IOC_RESET, 0);
    ioctl(*fd, PERF_EVENT_IOC_ENABLE, 0);
}

/***
 * Main matter
 ***/

int main(void)
{
    // heading
    printf("# %s\n# %s\n# %s\n", t1, t2, t3);
    printf("# Scattering = %8.3f/cm\n", MU_S);
    printf("# Absorption = %8.3f/cm\n", MU_A);
    printf("# Photons    = %8d\n#\n", PHOTONS);

    // configure RNG
    srand(SEED);

    long long count;
    int fd;
    start_perf(&fd);

    // start timer
    double start = wtime();
    // simulation
    for (unsigned int i = 0; i < PHOTONS; ++i) {
        photon(heat, heat2);
    }
    // stop timer
    double end = wtime();
    assert(start <= end);
    double elapsed = end - start;

    // Deshabilitar el contador
    ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);

    printf("# %lf seconds\n", elapsed);
    printf("# %lf photons per nanosecond\n", PHOTONS  / (elapsed*1e9));

    // Leer el contador
    if (read(fd, &count, sizeof(long long)) == -1) {
        count = -1;
    }
    printf("Instrucciones ejecutadas: %lld\n", count);
    close(fd);

    write_stat_file(elapsed, count);

    printf("# Radius\tHeat\n");
    printf("# [microns]\t[W/cm^3]\tError\n");
    float t = 4.0f * M_PI * powf(MICRONS_PER_SHELL, 3.0f) * PHOTONS / 1e12;
    for (unsigned int i = 0; i < SHELLS - 1; ++i) {
        printf("%6.0f\t%12.5f\t%12.5f\n", i * (float)MICRONS_PER_SHELL,
               heat[i] / t / (i * i + i + 1.0 / 3.0),
               sqrt(heat2[i] - heat[i] * heat[i] / PHOTONS) / t / (i * i + i + 1.0f / 3.0f));
    }
    printf("# extra\t%12.5f\n", heat[SHELLS - 1] / PHOTONS);

    return 0;
}
