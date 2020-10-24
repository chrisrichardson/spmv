# Require CMake 3.14
cmake_minimum_required(VERSION 3.14)

# Choose Release or Debug
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_NAME spmv)
set(PACKAGE_VERSION 1.0)
project(${PROJECT_NAME} VERSION ${PACKAGE_VERSION} LANGUAGES CXX)

# Set CMake behavior
cmake_policy(SET CMP0004 NEW)

set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# Not sure why need to set -O3 and -DNDEBUG as should be set by Release build type
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -DNDEBUG -fsycl -DMKL_ILP64")

add_subdirectory(spmv)
include_directories (${CMAKE_CURRENT_SOURCE_DIR})
# add_subdirectory(demos)
