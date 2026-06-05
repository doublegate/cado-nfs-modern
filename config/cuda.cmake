# Optional CUDA GPU ECM cofactorization backend (sieve/ecm/gpu_ecm.cu).
#
# Enable with -DENABLE_GPU=ON (e.g. via CMAKE_EXTRA_ARGS in local.sh). When off,
# or when no CUDA compiler is found, a stub (gpu_ecm_stub.cpp) provides
# gpu_ecm::available()==false so the rest of the build is unaffected and the CPU
# cofactorization path is used unconditionally.

option(ENABLE_GPU "Build the CUDA GPU ECM cofactorization backend" OFF)

set(HAVE_GPU_ECM 0)

if(ENABLE_GPU)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)     # provides the CUDA::cudart import target
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            # Default to Ampere (RTX 3090 = sm_86). Override with
            # -DCMAKE_CUDA_ARCHITECTURES=<cc> for other GPUs.
            set(CMAKE_CUDA_ARCHITECTURES 86)
        endif()
        set(HAVE_GPU_ECM 1)
        message(STATUS "GPU ECM backend ENABLED (CUDA ${CMAKE_CUDA_COMPILER_VERSION}, sm_${CMAKE_CUDA_ARCHITECTURES})")
    else()
        message(WARNING "ENABLE_GPU=ON but no CUDA compiler was found; GPU ECM backend disabled")
    endif()
endif()
