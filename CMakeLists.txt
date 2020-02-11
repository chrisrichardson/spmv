# Require CMake 3.5
cmake_minimum_required(VERSION 3.5)

# Choose Release or Debug
set(CMAKE_BUILD_TYPE Release)

# Use C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(eigen_mult)

# Set CMake behavior
cmake_policy(SET CMP0004 NEW)

set(CMAKE_CXX_FLAGS "-Werror=all")

find_package(Eigen3 REQUIRED)
find_package(MPI REQUIRED)
set(BLA_VENDOR Intel10_64lp)
find_package(BLAS)

add_executable(spmv_eigen main.cpp DistributedVector.cpp read_petsc.cpp CreateA.cpp)
target_include_directories(spmv_eigen PUBLIC ${EIGEN3_INCLUDE_DIR} ${MPI_INCLUDE_PATH})
target_link_libraries(spmv_eigen PUBLIC ${MPI_LIBRARIES})

if(BLAS_FOUND)
    add_executable(spmv_mkl main.cpp DistributedVector.cpp read_petsc.cpp CreateA.cpp)
    target_compile_definitions(spmv_mkl PUBLIC EIGEN_USE_MKL_ALL)
    target_include_directories(spmv_mkl PUBLIC ${EIGEN3_INCLUDE_DIR} ${MPI_INCLUDE_PATH})
    target_link_libraries(spmv_mkl PUBLIC ${MPI_LIBRARIES} ${BLAS_LIBRARIES})
endif()