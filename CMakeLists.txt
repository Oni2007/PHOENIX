cmake_minimum_required(VERSION 3.24)
project(phoenix LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(PHOENIX_BUILD_TESTS "Build the C++ test suite" ON)
option(PHOENIX_BUILD_BINDINGS "Build the phoenix_core Python module" ON)

# ── compute plane ─────────────────────────────────────────────────────────────
add_library(phoenix_cpp STATIC
  cpp/src/core/provenance.cpp
  cpp/src/core/status.cpp
  cpp/src/core/registry.cpp
  cpp/src/schema/request.cpp
  cpp/src/timing/calibration.cpp
  cpp/src/bench/sample_buffer.cpp
  cpp/src/bench/executor.cpp
  cpp/src/correctness/comparator.cpp
  cpp/backends/cpu/gemm_kernels.cpp
  cpp/backends/cpu/gemm_blas.cpp
  cpp/backends/cpu/cpu_backend.cpp
  cpp/backends/synthetic/synthetic_backend.cpp
)
target_include_directories(phoenix_cpp PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)
target_compile_definitions(phoenix_cpp PUBLIC PHOENIX_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
target_compile_options(phoenix_cpp PRIVATE -Wall -Wextra -Wpedantic)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(phoenix_cpp PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(phoenix_cpp PUBLIC PHOENIX_HAVE_OPENMP)
  message(STATUS "PHOENIX: OpenMP found - gemm_omp will be threaded")
else()
  # Recorded, not silently ignored: gemm_omp still runs, single-threaded, and the
  # environment manifest reports openmp=false so results are not misread.
  message(WARNING "PHOENIX: OpenMP NOT found - gemm_omp will run single-threaded")
endif()

# Independent vendor-BLAS cross-check (ADR-036). Optional: this development host
# has no root and therefore no system BLAS. CI installs libopenblas-dev and
# exercises the real path; here the correctness test visibly SKIPs rather than
# silently vanishing.
find_package(BLAS)
if(BLAS_FOUND)
  target_link_libraries(phoenix_cpp PUBLIC ${BLAS_LIBRARIES})
  target_compile_definitions(phoenix_cpp PUBLIC PHOENIX_HAVE_BLAS)
  message(STATUS "PHOENIX: BLAS found (${BLAS_LIBRARIES}) - vendor cross-check ACTIVE")
else()
  message(WARNING "PHOENIX: BLAS NOT found - vendor cross-check test will SKIP, not fail")
endif()

# ── python bindings ───────────────────────────────────────────────────────────
if(PHOENIX_BUILD_BINDINGS)
  find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(phoenix_core STABLE_ABI NB_STATIC bindings/phoenix_core.cpp)
  target_link_libraries(phoenix_core PRIVATE phoenix_cpp)
endif()

# ── tests ─────────────────────────────────────────────────────────────────────
if(PHOENIX_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
    GIT_SHALLOW TRUE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_executable(phoenix_tests
    cpp/tests/test_provenance.cpp
    cpp/tests/test_comparator.cpp
    cpp/tests/test_request_schema.cpp
    cpp/tests/test_gemm_correctness.cpp
    cpp/tests/test_executor.cpp
  )
  target_link_libraries(phoenix_tests PRIVATE phoenix_cpp GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(phoenix_tests)
endif()
