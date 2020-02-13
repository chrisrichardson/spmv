# Require CMake 3.5
cmake_minimum_required(VERSION 3.5)

# Choose Release or Debug
set(CMAKE_BUILD_TYPE Release)

# Use C++11
set(CMAKE_CXX_STANDARD 17)

# Require C++11
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_NAME eigen_mult)
project(${PROJECT_NAME} CXX)

# Set CMake behavior
cmake_policy(SET CMP0004 NEW)
# cmake_policy(SET CMP0076 NEW)

set(CMAKE_CXX_FLAGS "-Werror=all")

find_package(MPI REQUIRED)

set(BLA_VENDOR Intel10_64lp_seq)
find_package(BLAS)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR} ${MPI_INCLUDE_PATH})

# Executable
add_executable(${PROJECT_NAME} main.cpp read_petsc.cpp CreateA.cpp L2GMap.cpp)


if(BLAS_FOUND)
    target_compile_definitions(${PROJECT_NAME} PUBLIC EIGEN_USE_MKL_ALL)
    target_link_libraries(${PROJECT_NAME} ${BLAS_LIBRARIES})
    target_sources(${PROJECT_NAME} PUBLIC Operator/MKL.cpp)
endif()

# Target libraries
target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES})

# in cmake 17 this should be find_package(CUDAToolkit)
find_package(CUDA)
if (CUDA_FOUND)
    target_compile_definitions(${PROJECT_NAME} PUBLIC HAVE_CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${CUDA_cusparse_LIBRARY} ${CUDA_LIBRARIES})
    target_sources(${PROJECT_NAME} PUBLIC Operator/CUDA.cpp)
endif()
