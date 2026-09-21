cmake_minimum_required(VERSION 3.20)
project(OAA VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(OAA_BUILD_PYTHON_BINDINGS "Build the pybind11 Python module" ON)

add_library(oaa_core STATIC
    cpp/src/engine.cpp
)

set_target_properties(oaa_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(oaa_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

if(MSVC)
    target_compile_options(oaa_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(oaa_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(oaa_cpp_tests
        tests/cpp/test_engine.cpp
    )
    target_link_libraries(oaa_cpp_tests PRIVATE oaa_core)
    add_test(NAME oaa_cpp_tests COMMAND oaa_cpp_tests)
endif()

if(OAA_BUILD_PYTHON_BINDINGS)
    set(PYBIND11_FINDPYTHON ON)

    find_package(
        Python
        REQUIRED
        COMPONENTS Interpreter Development.Module
    )

    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(oaa_cpp
        bindings/pybind_module.cpp
    )
    target_link_libraries(oaa_cpp PRIVATE oaa_core)
    target_include_directories(oaa_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
    if(MINGW)
        target_link_options(oaa_cpp PRIVATE -static-libgcc -static-libstdc++)
    endif()
    install(TARGETS oaa_cpp DESTINATION .)
endif()
