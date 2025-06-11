#ifndef PHOTON_CUDA_H
#define PHOTON_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

void launch_simulation(float *h_heats,
                       float *h_heats_sq,
                       double *elapsed_time);

#ifdef __cplusplus
}
#endif

#endif