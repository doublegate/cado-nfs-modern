<!--
Reminder: this fork modernizes upstream CADO-NFS 3.0.0 and adds
implementation-level performance/orchestration work. NFS algorithm/math changes
should go upstream: https://gitlab.inria.fr/cado-nfs/cado-nfs
-->

## Summary

<!-- What does this change and why? Which toolchain/OS/version does it target? -->

## Type of change

- [ ] Toolchain / compiler / OS compatibility fix
- [ ] Python stdlib deprecation/removal fix
- [ ] Build system / packaging
- [ ] Performance (build flags / SIMD / GPU cofactorization)
- [ ] Rust orchestration (`rust/`)
- [ ] CI
- [ ] Documentation

## Verification

- [ ] `make` builds cleanly (after `bash scripts/setup-venv.sh`)
- [ ] `make check ARGS="-R '<subsystem>'"` passes for the affected area
- [ ] `cado-nfs.venv/bin/python3 ./cado-nfs.py <N> -t 4` still factors end-to-end (if runtime was touched)

Environment tested on (OS, compiler, CMake/Python/GMP/hwloc/OpenSSL versions):

```
```

## Checklist

- [ ] Minimal, surgical change; no bulk reformatting of upstream source
- [ ] Compatibility shims are version-guarded where practical
- [ ] `CHANGELOG.md` updated
