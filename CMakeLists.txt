# Require CMake 3.5
cmake_minimum_required(VERSION 3.5)

# Choose Release or Debug
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_NAME spmv)
project(${PROJECT_NAME} C CXX)

# Set CMake behavior
cmake_policy(SET CMP0004 NEW)
# cmake_policy(SET CMP0076 NEW)

set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

find_package(MPI REQUIRED)

set(BLA_VENDOR Intel10_64lp)
find_package(BLAS)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR} ${MPI_INCLUDE_PATH})

# Not sure why need to set -O3 and -DNDEBUG as should be set by Release build type
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=all -g -O3 -DNDEBUG -DEIGEN_MAX_ALIGN_BYTES=32")

# Executables
add_executable(cgmain cgmain.cpp)
target_link_libraries(cgmain ${PROJECT_NAME})
add_executable(matvecmain matvecmain.cpp CreateA.cpp)
target_link_libraries(matvecmain ${PROJECT_NAME})
add_executable(restrictmain restrictmain.cpp)
target_link_libraries(restrictmain ${PROJECT_NAME})

add_library(${PROJECT_NAME} SHARED read_petsc.cpp L2GMap.cpp cg.cpp Matrix.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "spmv.h")

if(BLAS_FOUND)
    target_compile_definitions(${PROJECT_NAME} PUBLIC EIGEN_USE_MKL_ALL)
    target_link_libraries(${PROJECT_NAME} ${BLAS_LIBRARIES})
    include_directories("/usr/include/mkl")
endif()

# Target libraries
target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES})

# in cmake 17 this should be find_package(CUDAToolkit)
find_package(CUDA)
if (CUDA_FOUND)
    enable_language(CUDA)
    target_sources(${PROJECT_NAME} PRIVATE shuffle_kernel.cu)
    target_compile_definitions(${PROJECT_NAME} PUBLIC HAVE_CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY} ${CUDA_LIBRARIES})
endif()

install(TARGETS ${PROJECT_NAME} cgmain matvecmain restrictmain
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)
