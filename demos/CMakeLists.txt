
# Executables

add_executable(vecnorm vecnorm.cpp)
target_link_libraries(vecnorm LINK_PUBLIC ${PROJECT_NAME})

add_executable(matvecmain matvecmain.cpp CreateA.cpp)
target_link_libraries(matvecmain ${PROJECT_NAME})

add_executable(diagonal diagonal.cpp CreateA.cpp)
target_link_libraries(diagonal ${PROJECT_NAME})

# add_executable(cgmain cgmain.cpp)
# target_link_libraries(cgmain LINK_PUBLIC ${PROJECT_NAME})

# add_executable(restrictmain restrictmain.cpp)
# target_link_libraries(restrictmain ${PROJECT_NAME})
