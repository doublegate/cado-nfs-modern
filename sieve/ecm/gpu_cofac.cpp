/* cxx_mpz <-> uint64 survivor-batch bridge for the GPU ECM backend; see
 * gpu_cofac.hpp. Normal C++ TU (uses GMP/cxx_mpz); calls the validated
 * gpu_ecm::factor_batch (gpu_ecm.cu / gpu_ecm_stub.cpp). */

#include "gpu_cofac.hpp"
#include "gpu_ecm.hpp"
#include <cstdint>
#include <gmp.h>
#include "gmp_auxx.hpp"   /* mpz_set_uint64 / mpz_get_uint64 */

namespace gpu_ecm {

/* GPU ECM here handles a single odd word in [3, 2^62). Wider cofactors and
 * primes are left to the CPU path. (A 128-bit path — validated in
 * bench/gpu-mont128.cu — is the documented extension for larger mfb.) */
static bool eligible(cxx_mpz const & c, uint64_t & out)
{
    if (mpz_sizeinbase(c.x, 2) > 61) return false;     /* must fit < 2^62 */
    if (mpz_cmp_ui(c.x, 2) <= 0) return false;         /* > 2 */
    if (mpz_even_p(c.x)) return false;                 /* odd (Montgomery) */
    out = mpz_get_uint64(c.x);
    return true;
}

std::vector<cxx_mpz> cofac_batch(std::vector<cxx_mpz> const & cofactors,
                                 int ncurves, unsigned long B1, unsigned long B2)
{
    size_t const M = cofactors.size();
    std::vector<cxx_mpz> result(M);
    for (auto & r : result) mpz_set_ui(r.x, 1);        /* default: no factor */

    if (M == 0 || !available()) return result;

    /* gather the eligible single-word cofactors, remembering their positions */
    std::vector<uint64_t> mod;
    std::vector<size_t>   idx;
    mod.reserve(M); idx.reserve(M);
    for (size_t i = 0; i < M; i++) {
        uint64_t w;
        if (eligible(cofactors[i], w)) { mod.push_back(w); idx.push_back(i); }
    }
    if (mod.empty()) return result;

    /* one GPU launch over the whole batch */
    std::vector<uint64_t> fac;
    factor_batch(mod, ncurves, B1, B2, fac);

    /* scatter found factors back to their cofactor positions */
    for (size_t k = 0; k < idx.size(); k++)
        if (fac[k] > 1) mpz_set_uint64(result[idx[k]].x, fac[k]);

    return result;
}

} // namespace gpu_ecm
