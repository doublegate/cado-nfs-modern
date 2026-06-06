/* No-op GPU polyselect init for CPU-only builds (v3.2.0-modern, Track C2).
 *
 * Compiled into the polyselect binary when the CUDA backend is NOT enabled
 * (HAVE_GPU_ECM off). It leaves cado_gpu_polyselect_roots null, so the caller in
 * polyselect_proots.cpp falls through to the per-prime CPU root-finding path with
 * zero behavioural change. The GPU build links polyselect-gpu.cu instead, whose
 * cado_gpu_polyselect_init() installs the real device implementation. */

#include "polyselect-gpu-hooks.h"

void cado_gpu_polyselect_init(void)
{
    /* no GPU backend in this build */
}
