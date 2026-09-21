cmake_minimum_required(VERSION 3.20)
project(OAA VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(OAA_BUILD_PYTHON_BINDINGS "Build the pybind11 Python module" ON)
option(OAA_BUILD_CUDA "Build the optional CUDA backend" OFF)

if(OAA_BUILD_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(OAA_CUDA_RUNTIME_SOURCE cpp/src/cuda_runtime.cu)
else()
    set(OAA_CUDA_RUNTIME_SOURCE cpp/src/cuda_runtime_stub.cpp)
endif()

add_library(oaa_core STATIC
    cpp/src/engine.cpp
    cpp/src/tensor.cpp
    cpp/src/memory.cpp
    cpp/src/tokenizer.cpp
    cpp/src/model.cpp
    cpp/src/inference.cpp
    ${OAA_CUDA_RUNTIME_SOURCE}
)

set_target_properties(oaa_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

if(OAA_BUILD_CUDA)
    set_target_properties(oaa_core PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )
endif()

target_include_directories(oaa_core
    PUBLIC
        cpp/include
)

if(OAA_BUILD_CUDA)
    target_compile_definitions(oaa_core PUBLIC OAA_HAS_CUDA=1)
    target_link_libraries(oaa_core PUBLIC CUDA::cudart)
else()
    target_compile_definitions(oaa_core PUBLIC OAA_HAS_CUDA=0)
endif()

if(MSVC)
    target_compile_options(oaa_core PRIVATE /W4 /permissive-)
elseif(NOT OAA_BUILD_CUDA)
    target_compile_options(oaa_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)

if(BUILD_TESTING)
    add_executable(oaa_cpp_tests
        tests/cpp/test_engine.cpp
    )

    target_link_libraries(oaa_cpp_tests PRIVATE oaa_core)

    add_test(NAME oaa_cpp_tests COMMAND oaa_cpp_tests)

    add_executable(oaa_runtime_core_tests
        tests/cpp/test_runtime_core.cpp
    )

    target_link_libraries(oaa_runtime_core_tests PRIVATE oaa_core)

    add_test(NAME oaa_runtime_core_tests COMMAND oaa_runtime_core_tests)

    add_executable(oaa_cuda_tests
        tests/cpp/test_cuda_runtime.cpp
    )

    target_link_libraries(oaa_cuda_tests PRIVATE oaa_core)

    add_test(NAME oaa_cuda_tests COMMAND oaa_cuda_tests)
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

    target_include_directories(
        oaa_cpp
        PRIVATE
            cpp/include
    )

    if(MINGW)
        target_link_options(oaa_cpp PRIVATE -static-libgcc -static-libstdc++)
    endif()

    install(TARGETS oaa_cpp DESTINATION .)
endif()
