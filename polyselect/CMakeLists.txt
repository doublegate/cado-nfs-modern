# defining a library avoids compiling the same oject file once for each
# binary. The choice of setting it as a shared library is completely
# arbitrary here. A static would do as well.

add_library(polyselect_common
    ${CADO_SHARED}
    auxiliary.cpp
    murphyE.cpp
    area.cpp
    size_optimization.cpp
    polyselect_alpha.cpp
    polyselect_norms.cpp
    polyselect-gpu-hooks.cpp
    )
target_link_libraries (polyselect_common utils)

# GPU polynomial-selection root-finding backend (v3.2.0-modern, Track C2). Built
# only with the CUDA backend enabled (-DENABLE_GPU=ON, which sets HAVE_GPU_ECM and
# enables the CUDA language in config/cuda.cmake); otherwise a no-op stub leaves
# the hook null so the per-prime CPU path runs. Exactly one of the two is linked
# into the polyselect binary, so cado_gpu_polyselect_init has a single definition.
if(HAVE_GPU_ECM)
    if(NOT DEFINED CADO_GPU_ARCH)
        set(CADO_GPU_ARCH 86)
    endif()
    set(polyselect_gpu_sources polyselect-gpu.cu)
    set_source_files_properties(polyselect-gpu.cu PROPERTIES LANGUAGE CUDA)
else()
    set(polyselect_gpu_sources polyselect-gpu-stub.cpp)
endif()

# It _seems_ that several polyselect-like libraries share the same
# general working principle, and that these objects can be treated as a
# bundle. (we want them to be compiled only once, for efficiency)
add_library(polyselect_middle
    OBJECT
    polyselect_arith.cpp
    polyselect_collisions.cpp
    polyselect_data_series.c
    polyselect_special_q.c
    polyselect_thread.cpp
    polyselect_thread_team.cpp
    polyselect_thread_league.cpp
    polyselect_main_data.cpp
    polyselect_main_queue.cpp
    polyselect_match.cpp
    polyselect_poly_header.c
    polyselect_primes_table.c
    polyselect_priority_queue.c
    polyselect_proots.cpp
    polyselect_qroots.cpp
    polyselect_shash.cpp
    polyselect_stats.c
)
mark_targets_as_openmp(polyselect_middle)

add_executable (polyselect
    polyselect.cpp
    ${polyselect_gpu_sources}
    $<TARGET_OBJECTS:polyselect_middle>
    )
target_link_libraries (polyselect polyselect_common ${pthread_libs})
mark_targets_as_openmp(polyselect)
if(HAVE_GPU_ECM)
    set_target_properties(polyselect PROPERTIES
        CUDA_ARCHITECTURES "${CADO_GPU_ARCH}" CUDA_STANDARD 20)
    target_compile_options(polyselect PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
    target_link_libraries(polyselect CUDA::cudart)
endif()

add_executable (polyselect_ropt
    ropt_main.cpp
    best_polynomials_queue.cpp
    ropt_single_sublattice_priority_queue.cpp
    ropt_sublattice_priority_queue.cpp
    ropt_sublattice_crt.cpp
    ropt.cpp
    ropt_arith.cpp
    ropt_io.cpp
    ropt_linear.cpp
    ropt_param.cpp
    ropt_quadratic.cpp
    ropt_stage1.cpp
    ropt_stage2.cpp
    ropt_str.cpp
    ropt_tree.c
    ropt_sublattice_priority_queue.cpp
)
target_link_libraries (polyselect_ropt polyselect_common ${pthread_libs})
mark_targets_as_openmp(polyselect_ropt)


add_executable (polyselect_gfpn polyselect_gfpn.cpp gfpkdlpolyselect.cpp)
target_link_libraries (polyselect_gfpn  polyselect_common utils)

add_executable (polyselect3 polyselect3.cpp)
mark_targets_as_openmp(polyselect3)
target_link_libraries (polyselect3 polyselect_common utils)

if(ENABLE_SHARED)
    install(TARGETS polyselect_common DESTINATION ${LIBSUFFIX}/polyselect)
endif()
install(TARGETS polyselect RUNTIME DESTINATION ${LIBSUFFIX}/polyselect)
install(TARGETS polyselect_ropt RUNTIME DESTINATION ${LIBSUFFIX}/polyselect)
install(TARGETS polyselect3 RUNTIME DESTINATION ${LIBSUFFIX}/polyselect)

# dlpolyselect
add_executable (dlpolyselect dlpolyselect.cpp)
mark_targets_as_openmp(dlpolyselect)
target_link_libraries (dlpolyselect polyselect_common utils)
install(TARGETS dlpolyselect RUNTIME DESTINATION ${LIBSUFFIX}/polyselect)


include(CMakeLists-nodist.txt OPTIONAL)

# skewness
add_executable (skewness skewness.cpp)
target_link_libraries (skewness polyselect_common utils)
install(TARGETS skewness RUNTIME DESTINATION ${LIBSUFFIX}/polyselect)

# score
add_executable (score score.cpp)
target_link_libraries (score polyselect_common utils)
install(TARGETS score RUNTIME DESTINATION ${LIBSUFFIX}/polyselect)
