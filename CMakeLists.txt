# Require CMake 3.5
cmake_minimum_required(VERSION 3.5)

# Choose Release or Debug
set(CMAKE_BUILD_TYPE Release)

# Use C++11
set(CMAKE_CXX_STANDARD 17)

# Require C++11
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_NAME spmv)
project(${PROJECT_NAME})

# Set CMake behavior
cmake_policy(SET CMP0004 NEW)

find_package(MPI REQUIRED)

set(BLA_VENDOR Intel10_64lp)
find_package(BLAS)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR} ${MPI_INCLUDE_PATH})

# Not sure why need to set -O3 and -DNDEBUG as should be set by Release build type
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=all -g -O3 -DNDEBUG")

# Executables
add_executable(cgmain cgmain.cpp)
target_link_libraries(cgmain ${PROJECT_NAME})
add_executable(matvecmain matvecmain.cpp CreateA.cpp)
target_link_libraries(matvecmain ${PROJECT_NAME})
add_executable(restrictmain restrictmain.cpp)
target_link_libraries(restrictmain ${PROJECT_NAME})

add_library(${PROJECT_NAME} SHARED read_petsc.cpp L2GMap.cpp cg.cpp Matrix.cpp)

if(BLAS_FOUND)
    target_compile_definitions(${PROJECT_NAME} PUBLIC EIGEN_USE_MKL_ALL)
    target_link_libraries(${PROJECT_NAME} ${BLAS_LIBRARIES})
endif()

# Target libraries
target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES})
