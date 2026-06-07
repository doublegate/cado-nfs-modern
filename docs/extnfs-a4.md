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

## A6 — feasibility skeleton (v3.3.0-modern): concrete interfaces

A4 gave the gap analysis; **A6 sketches the *interfaces*** a future implementation
would add, to make the size of each piece concrete. This is a **documented design,
not committed code** — no tower math enters the tree. Each block shows how the
existing 2-D CADO structure generalizes; file references are the real anchors.

### 1. Tower field + tower polynomial selection

The classic side has a single polynomial pair `(f0, f1) ∈ Z[x]`. The tower replaces
`Z` by `Z[ι]/h(ι)` and the pair becomes bivariate in `(ι, x)`:

```text
struct tower_field {                 // generalizes the (p, f0, f1) of polyselect
    cxx_mpz       p;                 // characteristic
    cxx_mpz_poly  h;                 // deg η, irreducible mod p — defines Z[ι]/h
    int           eta;               // = deg h  (eta=1 ⇒ classic NFS-DLP)
    int           kappa;             // target extension: n = eta * kappa, gcd=1
    // the two sides, each an element of (Z[ι]/h)[x]:
    cxx_mpz_poly  f0[ /*eta*/ ];     // f_s(x) = sum_j (coeff in Z[ι]/h) x^j
    cxx_mpz_poly  f1[ /*eta*/ ];     // sharing a common irreducible deg-κ factor mod p
};

// tower_polyselect(): choose h, then lift JL / Conjugation / Sarkar–Singh to the
//   tower so f0,f1 share a degree-κ factor over GF(p^eta). Determines total cost.
//   Anchor to generalize: polyselect/, parameters/dlp/{Joux-Lercier,TwoQuadratics}.
int tower_polyselect(tower_field & T, cxx_mpz const & p, int n);
```

### 2. The ≥3-D special-q siever shape

The classic q-lattice has **two** basis vectors and survivors map `(i,j) → (a,b)`.
On the tower a relation is `a(ι) + b(ι)·x` with `a,b ∈ Z[ι]/h` of degree `< η`, so
the enumeration vector has `2η` integer coordinates ⇒ **(2η)-D** (η=1 ⇒ the
existing 2-D siever; η=2 ⇒ 4-D, etc. — A1's d-D enumeration):

```text
struct tower_qlattice {              // generalizes sieve/ q-lattice (2 -> 2η basis)
    tower_ideal   special_q;         // q is now a tower ideal (below)
    int64_t       basis[2*ETA][2*ETA]; // reduced lattice basis, dim 2η
};

// survivor coordinates -> tower relation: (i_0..i_{η-1}, j_0..j_{η-1}) -> (a(ι),b(ι))
void tower_ij_to_ab(tower_relation & r,
                    int64_t const i_vec[ETA], int64_t const j_vec[ETA],
                    tower_qlattice const & L);

// the inner sieve: enumerate the (2η)-D box, accumulate per-ideal log-norms.
//   Anchor + the hard part: a new siever; sieve/las.cpp + bucket geometry are 2-D.
//   The bucket region, the (i,j) line sieve, and SIMD/GPU layout all assume 2-D.
```

The bucket-sieve geometry (`sieve/`), the `fill_in_buckets` scatter, and the
AVX2/AVX-512 modular-inverse work (B1/B4) are all 2-D-shaped; none survive the
dimensional lift unchanged — this is why A4 calls the siever "the largest single
piece."

### 3. Tower-ideal + relation bookkeeping

```text
struct tower_ideal {                 // generalizes a (p, r) prime ideal
    cxx_mpz       norm_p;            // rational prime below
    cxx_mpz_poly  root_mod;          // the ideal's residue data over Z[ι]/h
    int           side;
};

struct tower_relation {              // generalizes sieve/ las relation (a,b)+factors
    cxx_mpz_poly  a, b;              // a(ι), b(ι) ∈ Z[ι]/h  (deg < η)
    std::vector<tower_ideal> factors[2];
};

// norm over the tower: Res_x( f_s(x), a(ι)+b(ι)x ) then Res_ι( ·, h(ι) ) — a
// nested resultant, not the single Z-resultant of classic NFS.
void tower_norm(cxx_mpz & N, tower_relation const & r, tower_field const & T, int side);
```

Filtering (`filter/`) and the virtual-log / Schirokauer-map reconstruction
(`reconstructlog`, `sqrt/`) consume the relation/ideal format, so they generalize
pervasively; **BWC linear algebra over GF(ℓ) is reusable as-is** (field-agnostic),
which is why A4 scopes it out of the tower-specific work.

### Honest verdict (unchanged)

The skeleton confirms A4's sizing: `tower_polyselect` and a `(2η)-D` siever are the
determining pieces, both research-grade and well outside the fork's measured-win
tracks. A6 commits **only** these interface sketches — no tower arithmetic, no
siever — so the avenue is concretely documented for a future effort without adding
unvalidated math to the tree.

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
