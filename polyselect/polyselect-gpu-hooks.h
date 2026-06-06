#ifndef CADO_POLYSELECT_GPU_HOOKS_H
#define CADO_POLYSELECT_GPU_HOOKS_H

/* GPU polynomial-selection hook (v3.2.0-modern, Track C2).
 *
 * polyselect stage-1 spends ~30-40% of its time finding the roots of x^d == a
 * (mod p) for the small primes p (utils/roots_mod.cpp's roots_mod_uint64, called
 * per prime in polyselect_proots_compute_subtask). That is embarrassingly
 * parallel over primes. The GPU backend (polyselect-gpu.cu, built only under
 * -DENABLE_GPU=ON) installs the function pointer below; the CUDA-free
 * polyselect_proots.cpp calls it through the pointer to compute a whole batch of
 * primes' roots at once. When no GPU backend is loaded the pointer stays null and
 * the per-prime CPU path runs unchanged -- zero behavioural change.
 *
 * The pointer is defined in polyselect-gpu-hooks.cpp (compiled into
 * polyselect_common, the dependency-free leaf both sides link), so there is no
 * circular static-link dependency (the matmul-gpu-hooks lesson). The whole GPU
 * path is additionally gated at the call site on the CADO_GPU_POLYSELECT env. */

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/* For each i in [0,n): solve x^d == a[i] (mod p[i]) and write the (distinct)
 * roots into roots[i*d .. i*d + nr[i]) with nr[i] <= d. p[i]==0 marks a prime to
 * skip (nr[i]=0). Returns 1 if the batch was handled on the device, 0 to force
 * the caller back to the per-prime CPU path. The root *set* matches
 * roots_mod_uint64 (validated bit-exact vs direct evaluation). */
extern int (*cado_gpu_polyselect_roots)(const uint64_t * a, const uint32_t * p,
                                        unsigned int n, int d,
                                        uint64_t * roots, unsigned int * nr);

/* Install the hook (a no-op in a non-GPU build, where it is defined as a stub).
 * Called once from polyselect's main when CADO_GPU_POLYSELECT is set. */
extern void cado_gpu_polyselect_init(void);

#ifdef __cplusplus
}
#endif

#endif /* CADO_POLYSELECT_GPU_HOOKS_H */
