# Require CMake 3.5
cmake_minimum_required(VERSION 3.5)

# Use C++11
set(CMAKE_CXX_STANDARD 11)

# Require C++11
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_NAME eigen_mult)
project(${PROJECT_NAME})

# Set CMake behavior
cmake_policy(SET CMP0004 NEW)

set(CMAKE_CXX_FLAGS "-g -O2 -Werror=all")

find_package(MPI REQUIRED)

set(BLA_VENDOR Intel10_64lp)
find_package(LAPACK)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR} ${MPI_INCLUDE_PATH})

# Executable
add_executable(${PROJECT_NAME} main.cpp DistributedVector.cpp)

if (LAPACK_FOUND)
  target_compile_definitions(${PROJECT_NAME} PUBLIC HAS_MKL)
endif()

# Target libraries
target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES} ${LAPACK_LIBRARIES})
