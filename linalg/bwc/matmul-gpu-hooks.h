#ifndef CADO_MATMUL_GPU_HOOKS_H
#define CADO_MATMUL_GPU_HOOKS_H

/* Cross-TU hooks for keeping BWC vectors device-resident across the comm
 * (v3.1.0-modern, Track 2.2 — comm-on-device, the full-vector-residency win).
 *
 * The BWC comm (mmt_vec_allreduce, in matmul_top_comm.cpp, part of bwc_base)
 * lives in a CUDA-free translation unit. The GPU matmul backend (matmul-gpu.cu)
 * installs the function pointers below in its constructor; bwc_base calls them
 * through the pointers, so the comm can run the GF(2) reduce/broadcast directly
 * on the device-resident sibling vectors instead of bouncing through host memory
 * every iteration. When no GPU backend is loaded the pointers stay null and the
 * comm takes its ordinary host path — zero behavioural change. The whole device
 * path is additionally gated at the call site on CADO_GPU_VECRESIDENT and on the
 * single-node case (no MPI); see matmul_top_comm.cpp. */

#include <cstddef>

/* Single-node GF(2) allreduce on device: given the host pointers of the T
 * sibling vectors (each `bytes` long), reduce them with XOR and broadcast the
 * result back over all T, entirely on the device-resident copies. Returns 1 if
 * it handled the comm on device (all T buffers were present & current in the
 * device registry), 0 if the caller must fall back to the host path. On success
 * the device copies are left current and the host copies are marked stale (to be
 * materialised on demand by gpu_sync_to_host). */
extern int (*cado_gpu_comm_reduce_bcast)(void * const * host_ptrs,
                                         unsigned int T, size_t bytes);

/* Materialise a device-resident vector back to its host buffer if the device
 * copy is newer (a no-op otherwise). Called at host-read sites (inner products,
 * checkpoints, twist) so a resident vector is correct on the host when actually
 * read. Returns 1 if a copy happened, 0 otherwise. */
extern int (*cado_gpu_sync_to_host)(void const * host_ptr);

#endif /* CADO_MATMUL_GPU_HOOKS_H */
