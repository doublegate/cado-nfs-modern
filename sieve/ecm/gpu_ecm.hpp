#ifndef CADO_GPU_ECM_HPP
#define CADO_GPU_ECM_HPP

/* Batched ECM cofactorization on a CUDA GPU, for use as an optional backend of
 * facul_all() (see sieve/ecm/facul.cpp). The GPU handles the common small-
 * cofactor case (odd modulus < 2^62, one machine word); larger cofactors stay
 * on the CPU path. Validated kernels: bench/gpu-ecm*.cu (stage 1 + stage-2
 * BSGS), bit-exact vs CPU. See docs/gpu-cofactorization.md.
 *
 * This header is plain C++ (no CUDA types) so facul.cpp can include it whether
 * or not the build has CUDA; gpu_ecm::available() returns false when built
 * without CUDA or when no device is present.
 */

#include <cstdint>
#include <vector>

namespace gpu_ecm {

/* True iff this binary was built with CUDA and a usable device is present. */
bool available();

/* For each modulus, try `ncurves` ECM curves (stage 1 to B1, stage-2 BSGS to
 * B2). On return, factor[i] is a nontrivial factor of moduli[i], or 0 if none
 * was found / the modulus was skipped. A modulus is processed only if it is
 * odd and < 2^62; otherwise factor[i] = 0 (leave it to the CPU path).
 *
 * `moduli` and `factor` have the same length; `factor` is resized by the call.
 */
void factor_batch(std::vector<uint64_t> const & moduli,
                  int ncurves,
                  unsigned long B1,
                  unsigned long B2,
                  std::vector<uint64_t> & factor);

} // namespace gpu_ecm

#endif /* CADO_GPU_ECM_HPP */
