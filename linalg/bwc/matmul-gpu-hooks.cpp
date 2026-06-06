/* Definitions of the cado_gpu_* comm-on-device hook pointers (Track 2.2).
 *
 * These are DECLARED in matmul-gpu-hooks.h and USED from two sides that must not
 * depend on each other at link time:
 *   - bwc_base (matmul_top_comm.cpp) *calls* through them, and
 *   - the GPU matmul backend (matmul-gpu.cu, libmatmul_*_gpu) *installs* them.
 * If the definitions lived in bwc_base, linking the GPU backend into a target
 * that does not otherwise pull all of bwc_base (e.g. bench_matcache) created a
 * circular static-link dependency and left cado_gpu_* unresolved. Defining them
 * here, in matmul_common — the dependency-free leaf that every matmul backend
 * already links — breaks the cycle: both sides resolve the symbols from a common
 * leaf, with no GPU/CUDA dependency in CPU-only builds. Null pointers mean "no
 * GPU backend loaded" => the ordinary host comm path, exactly as before. */

#include "matmul-gpu-hooks.h"

int (*cado_gpu_comm_reduce_bcast)(void * const *, unsigned int, size_t) = nullptr;
int (*cado_gpu_sync_to_host)(void const *) = nullptr;
int (*cado_gpu_dev_xor_block)(void *, void * const *, unsigned int, size_t, size_t, size_t) = nullptr;
int (*cado_gpu_dev_copy_block)(void *, size_t, void const *, size_t, size_t, size_t, size_t) = nullptr;
int (*cado_gpu_dev_upload)(void const *, size_t) = nullptr;
int (*cado_gpu_dev_download)(void *, size_t) = nullptr;
int (*cado_gpu_dev_sync)(void) = nullptr;
int (*cado_gpu_dev_ensure)(void const *, size_t) = nullptr;
int cado_gpu_residency_active = 0;
int cado_gpu_residency_available = 0;
int (*cado_gpu_dev_mark_resident)(void const *, size_t) = nullptr;
int (*cado_gpu_x_dotprod)(void *, uint32_t const *, unsigned int, unsigned int,
                          unsigned int, void const *, size_t, unsigned int,
                          unsigned int, unsigned int, int) = nullptr;
int (*cado_gpu_addmul_tiny)(void *, void const *, void const *, unsigned int,
                            unsigned int, unsigned int, size_t, size_t, size_t) = nullptr;
