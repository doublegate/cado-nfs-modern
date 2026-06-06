/* Definition of the GPU polyselect hook pointer (v3.2.0-modern, Track C2).
 *
 * The pointer is DECLARED in polyselect-gpu-hooks.h and used from two sides that
 * must not depend on each other at link time:
 *   - polyselect_proots.cpp (in polyselect_middle) *calls* through it, and
 *   - the GPU backend (polyselect-gpu.cu) *installs* it from cado_gpu_polyselect_init().
 * If the definition lived next to the caller, linking the CUDA backend created the
 * same circular static-link dependency we hit with the matmul GPU hooks. Defining
 * it here, in polyselect_common -- the dependency-free leaf every polyselect binary
 * already links -- breaks the cycle: both sides resolve the symbol from a common
 * leaf, with no CUDA dependency in CPU-only builds. A null pointer means "no GPU
 * backend installed" => the per-prime CPU root-finding path runs, exactly as before.
 *
 * cado_gpu_polyselect_init() is deliberately NOT defined here: it is defined either
 * in polyselect-gpu.cu (GPU build, installs the pointer) or in
 * polyselect-gpu-stub.cpp (non-GPU build, a no-op), exactly one of which is linked
 * into each binary -- so there is no duplicate-symbol clash. */

#include "polyselect-gpu-hooks.h"

int (*cado_gpu_polyselect_roots)(const uint64_t *, const uint32_t *,
                                 unsigned int, int,
                                 uint64_t *, unsigned int *) = nullptr;

int (*cado_gpu_polyselect_collisions)(
        const uint32_t *, const uint8_t *, const int64_t *,
        unsigned int, unsigned int, int64_t,
        const uint64_t **, const uint32_t **,
        const uint32_t **, unsigned int *) = nullptr;
