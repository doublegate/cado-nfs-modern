"""
GPU pre-NFS factoring stage for cado-nfs.py (v3.1.0-modern, Track 2.1).

Runs the misc/gpu_prefactor `gpu-prefactor` binary (built only with
-DENABLE_GPU=ON) to strip small/medium factors from N on the GPU before NFS.
Returns the prime factors stripped and the remaining cofactor; cado-nfs.py then
either reports a complete factorization (skipping NFS) or continues NFS on the
reduced cofactor. Pure-Python, no third-party deps.
"""

import os
import random
import subprocess


def is_probable_prime(n, rounds=40):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for _ in range(rounds):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_binary(pathdict):
    """Locate the gpu-prefactor binary in the build/install tree, or None."""
    for base in (pathdict.get("lib"), pathdict.get("bin")):
        if not base:
            continue
        cand = os.path.join(base, "misc", "gpu-prefactor")
        if os.path.exists(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def run(binary, N, b1, curves, b2, logger):
    """Run the GPU pre-factoring stage. Returns (sorted prime factors stripped,
    remaining cofactor). On any error, returns ([], N) so the caller falls back
    to a normal NFS run."""
    if b2 <= 0:
        b2 = 100 * b1
    cmd = [binary, str(N), str(b1), str(curves), str(b2)]
    logger.info("GPU pre-factoring: %s", " ".join(cmd))
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, timeout=3600)
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("gpu-prefactor did not run (%s); skipping", e)
        return [], N
    for line in res.stdout.splitlines():
        line = line.rstrip()
        if line.startswith("#") or line.startswith("factors stripped:") \
                or line.startswith("remaining cofactor:"):
            logger.info("  %s", line)
    primes, cof = [], N
    for line in res.stdout.splitlines():
        if line.startswith("factors stripped:"):
            for tok in line.split(":", 1)[1].split():
                try:
                    f = int(tok)
                except ValueError:
                    continue
                # only trust prime divisors (ECM gcds are normally prime, but be safe)
                if f > 1 and N % f == 0 and is_probable_prime(f):
                    while cof % f == 0:
                        cof //= f
                    if f not in primes:
                        primes.append(f)
    return sorted(primes), cof
