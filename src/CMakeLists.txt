
add_library(${PROJECT_NAME} SHARED L2GMap.cpp Matrix.cpp read_petsc.cpp cg.cpp)
target_include_directories (${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# Target libraries
target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES})

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "spmv.h")
set_target_properties(${PROJECT_NAME} PROPERTIES PRIVATE_HEADER "L2GMap.h;Matrix.h;read_petsc.h;cg.h")

if(BLAS_FOUND)
    target_compile_definitions(${PROJECT_NAME} PUBLIC EIGEN_USE_MKL_ALL)
    target_link_libraries(${PROJECT_NAME} ${BLAS_LIBRARIES})
    include_directories("/usr/include/mkl")
endif()


install(TARGETS ${PROJECT_NAME}
    EXPORT libspmv
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
    PRIVATE_HEADER DESTINATION include
 )

# First attempt at installing a config
include(CMakePackageConfigHelpers)
write_basic_package_version_file( 
  libspmvConfig.cmake
    VERSION ${PACKAGE_VERSION}
    COMPATIBILITY AnyNewerVersion
    )

install(EXPORT libspmv
  FILE libspmvConfig.cmake
  NAMESPACE spmv::
  DESTINATION lib/cmake/spmv)
