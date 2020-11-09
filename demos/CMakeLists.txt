# Executables

add_executable(vecnorm vecnorm.cpp)
target_link_libraries(vecnorm LINK_PUBLIC ${PROJECT_NAME})

add_executable(matvecmain matvecmain.cpp CreateA.cpp)
target_link_libraries(matvecmain ${PROJECT_NAME})

add_executable(diagonal diagonal.cpp)
target_link_libraries(diagonal ${PROJECT_NAME})

add_executable(cgmain cgmain.cpp)
target_link_libraries(cgmain LINK_PUBLIC ${PROJECT_NAME})

# Copy files matrix and vector files to binary dir
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/petsc_mat_poisson
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/petsc_vec_poisson
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})