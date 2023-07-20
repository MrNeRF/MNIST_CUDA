cmake_minimum_required (VERSION 3.22)
# Determine the project root directory
get_filename_component(PROJ_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)

# Define the absolute path to libtorch
set(LIBTORCH_DIR "${PROJ_ROOT_DIR}/external/libtorch")

# Set CMAKE_PREFIX_PATH
set(CMAKE_PREFIX_PATH ${LIBTORCH_DIR})

# Add the _GLIBCXX_USE_CXX11_ABI=0 flag
# This is needed to avoid linker errors with libtorch
# https://stackoverflow.com/questions/62693218/how-to-solve-gtest-and-libtorch-linkage-conflict
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

project(mnist_cuda LANGUAGES CUDA CXX)

include(ProcessorCount)
ProcessorCount(total_cores)
if(NOT DEFINED total_cores OR total_cores EQUAL 0)
    set(total_cores 1)
    set(used_cores 1)
elseif(total_cores GREATER 1)
    math(EXPR used_cores "${total_cores} - 2") # use total_cores-2 if total_cores > 1
endif()
set(ENV{MAKEFLAGS} "-j${used_cores}")
message(STATUS "Building with ${used_cores} out of ${total_cores} available cores")

set(HEADERS
    includes/error_checking.cuh
    includes/load_mnist.cuh
    includes/mlp.cuh
    includes/stb_image_write.h
    includes/linear_layer.cuh
    includes/neural_network.cuh
    includes/layer.cuh
    includes/activation.cuh
    includes/loss.cuh
)

set(SOURCES
    src/main.cu    
    src/load_mnist.cu
    src/linear_layer.cu
    src/activation.cu
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/includes)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES 89
    CUDA_STANDARD 17 
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17 
    CXX_STANDARD_REQUIRED ON
)

# Torch setup
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(Torch REQUIRED)

#target compile options
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(${PROJECT_NAME} PRIVATE -g -G -Xcompiler -Werror)
  set_target_properties(${PROJECT_NAME} PROPERTIES
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_SEPARABLE_COMPILATION ON
  )
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(${PROJECT_NAME} PRIVATE -O3)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(TBB REQUIRED)

#Check CUDA version
if(CUDAToolkit_VERSION VERSION_LESS "12.0")
    message(FATAL_ERROR "This project requires CUDA 12.0 or higher")
endif()

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        ${TORCH_LIBRARIES}
        CUDA::cudart
        CUDA::curand
        CUDA::cublas
)

# Add Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.10.0
)

FetchContent_MakeAvailable(googletest)

# Enable CTest
enable_testing()

# Add test executable
add_executable(cuda_kernel_tests tests/cuda_kernel_tests.cu)

target_include_directories(cuda_kernel_tests PRIVATE ${PROJECT_SOURCE_DIR}/includes ${TORCH_INCLUDE_DIRS})
set_target_properties(cuda_kernel_tests PROPERTIES
    CUDA_ARCHITECTURES 89
    CUDA_STANDARD 17 
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17 
    CXX_STANDARD_REQUIRED ON
)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(cuda_kernel_tests PRIVATE -g -G -Xcompiler -Werror)
  set_target_properties(cuda_kernel_tests PROPERTIES
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_SEPARABLE_COMPILATION ON
  )
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(cuda_kernel_tests PRIVATE -O3)
endif()

target_include_directories(cuda_kernel_tests PRIVATE ${PROJECT_SOURCE_DIR}/includes)

target_link_libraries(cuda_kernel_tests
    PRIVATE
        CUDA::cudart
        CUDA::curand
        CUDA::cublas
        gtest 
        gtest_main
        ${TORCH_LIBRARIES}
)

# Register the test with CTest
add_test(NAME CUDA_Kernel_Tests COMMAND cuda_kernel_tests)

set_tests_properties(CUDA_Kernel_Tests
    PROPERTIES
    REPEAT 10
)

# Automatically run tests after each build and show output
#add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND ctest --output-on-failure)
