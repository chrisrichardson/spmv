
# Executables

add_executable(cgmain cgmain.cpp)
target_link_libraries(cgmain LINK_PUBLIC ${PROJECT_NAME})

add_executable(matvecmain matvecmain.cpp CreateA.cpp)
target_link_libraries(matvecmain ${PROJECT_NAME})

add_executable(restrictmain restrictmain.cpp)
target_link_libraries(restrictmain ${PROJECT_NAME})

add_test(
    NAME cg2
    COMMAND $<TARGET_FILE:cgmain> 2
)
add_test(
    NAME cg2-parallel
    COMMAND mpirun -np 2 $<TARGET_FILE:cgmain> 2
)
add_test(
    NAME cg3
    COMMAND $<TARGET_FILE:cgmain> 3
)

set_tests_properties(cg2 PROPERTIES
    ENV OMP_NUM_THREADS=1
)
set_tests_properties(cg3 PROPERTIES
    ENV OMP_NUM_THREADS=1
)