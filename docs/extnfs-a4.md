# exTNFS / Tower-NFS feasibility for the DLP side (Roadmap A4)

Per the roadmap, A4 is a **research-grade feasibility study, documentation only —
no speculative math committed**. It assesses what it would take to add (extended)
Tower NFS to CADO's discrete-logarithm side, and gives an honest verdict. It is
also the proper home for the **A1** finding (the "3-D lattice sieving" paper,
arXiv:2001.10860, is TNFS — not factorization).

## What CADO already does for DLP (codebase facts)

CADO computes discrete logarithms via **classic NFS**:

- **GF(p)** — fully supported (`./cado-nfs.py -dlp -ell <ℓ> target=<t> <p>`;
  `parameters/dlp/params.p30…p155`).
- **GF(p²)** — integrated (`-gfpext 2`, `parameters/dlp/params.p2dd20|p2dd30`),
  using the Conjugation / Joux–Lercier / TwoQuadratics polynomial-selection
  methods (`parameters/dlp/{Joux-Lercier,TwoQuadratics}`).
- **GF(p^k), small k** — works "mutatis mutandis" (the two polynomials must share
  an irreducible degree-k factor over GF(p)), **but polynomial selection for k>2
  is not included — the user must supply the polynomials** (README.dlp).

Crucially, the **siever is 2-D**: a relation is a pair `(a,b)` and the q-lattice
has exactly two basis vectors (survivors map `(i,j)→(a,b)`). This is the same
siever the factorization side uses; classic NFS-DLP reduces GF(p^k) to a 2-D sieve
via a polynomial *pair* sharing a degree-k factor.

## What TNFS / exTNFS is, and why it's different

**TNFS** (Barbulescu–Gaudry–Kleinjung, ASIACRYPT 2015) and **exTNFS**
(Kim–Barbulescu, CRYPTO 2016; Kim–Jeong, PKC 2017) are a *different framework* for
GF(p^k) DLP in the **medium-characteristic** regime with **composite k**. Instead
of a number field over ℚ, they work over a **number-field tower**: relations live
in a ring `R = (Z[ι]/h(ι))[x]/f(x)` — i.e. elements are **higher-degree** in the
auxiliary variable `ι`, `a(ι) + b(ι)·x` with `a,b ∈ Z[ι]/h` of degree up to
`deg h − 1`. That extra degree is exactly why TNFS sieving is
**higher-dimensional** (3-D and up — the A1 paper): you enumerate `(a_0,a_1,…)`
over the tower, not a 2-D `(a,b)`.

The payoff is asymptotic: exTNFS drops the medium-prime complexity from
`L_Q(1/3, (96/9)^{1/3})` to `L_Q(1/3, (48/9)^{1/3})` (≈ 2.15 → 1.71 with multiple
number fields), `Q = p^n`, when `n = ηκ` with `gcd(η,κ)=1`; later generalized to
arbitrary composite `n`. This directly drove the keysize updates for
pairing-based cryptography (the `n = 6, 12` cases).

## What CADO would need (component gap analysis)

Adding (ex)TNFS is **not** a localized change — it touches every NFS stage:

1. **Tower polynomial selection.** Choose `h(ι)` (deg η, defining `Z[ι]/h`) and a
   polynomial *pair* over that tower with a common degree-κ factor mod p. The
   existing JL/Conjugation/Sarkar–Singh methods must be lifted to the tower.
   *New code; the determining step for the whole algorithm's cost.*
2. **Higher-dimensional sieving.** The siever must enumerate `a(ι)+b(ι)x` over the
   tower — `deg h ≥ 2` ⇒ ≥ 3-D enumeration (A1's `3`-D and `d`-D lattice
   enumeration). CADO's siever is hard-wired 2-D (the q-lattice, the `(i,j)→(a,b)`
   map, the bucket geometry). *A new siever, the largest single piece.*
3. **Tower ideal / relation handling.** Norms, factor bases, and the relation
   format are over `R`, not `Z[α]`; the filtering and Schirokauer-map / virtual-log
   machinery generalize to the tower. *Pervasive changes in `sieve/`, `filter/`,
   `sqrt/`/`reconstructlog`.*
4. **Linear algebra & individual log.** BWC over GF(ℓ) is reusable (the matrix is
   field-agnostic), but the descent / individual-logarithm step is tower-specific.

## Honest feasibility verdict

- **(ex)TNFS is the asymptotically-best known method** for medium-characteristic
  GF(p^k) with composite k, and is the right tool for the pairing-relevant
  `n = 6, 12` fields. CADO's classic NFS-DLP remains appropriate for GF(p),
  GF(p²), and prime-k fields.
- Implementing it in CADO is a **research-grade, multi-component effort** — most of
  all a **new higher-dimensional siever** (A1) plus **tower polynomial selection**
  and **tower-ideal relation handling**. It is *not* a near-term fork item, and it
  is well outside the fork's actual leverage (the GPU fixed-width tracks A2/C1/C2/C3
  and orchestration, where the measured wins are).
- Per the roadmap ethos, this avenue is **documented, not committed**: no
  speculative tower math is added to the tree. A1 is recorded here (its 3-D
  enumeration is the TNFS siever, not a factorization win); any future pursuit of
  GF(p^6)/GF(p^12) DLP would start from §"component gap analysis" above.

## Sources

- Barbulescu, Gaudry, Kleinjung. *The Tower Number Field Sieve.* ASIACRYPT 2015
  (eprint 2015/505).
- Kim, Barbulescu. *Extended Tower Number Field Sieve: A New Complexity for the
  Medium Prime Case.* CRYPTO 2016 (eprint 2015/1027).
- Kim, Jeong. *exTNFS with Application to Finite Fields of Arbitrary Composite
  Extension Degree.* PKC 2017 (eprint 2016/526).
- Higher-dimensional (3-D/d-D) lattice sieving for TNFS: arXiv:2001.10860 (the A1
  paper).
- CADO-NFS `README.dlp`, `parameters/dlp/`, `scripts/cadofactor/toplevel.py`
  (`-gfpext`).
