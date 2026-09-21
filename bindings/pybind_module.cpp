#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "oaa/cuda_runtime.hpp"
#include "oaa/engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(oaa_cpp, m) {
    m.doc() = "OAA C++ runtime Python binding";

    py::class_<oaa::GenerationConfig>(m, "GenerationConfig")
        .def(py::init<>())
        .def_readwrite("max_tokens", &oaa::GenerationConfig::max_tokens)
        .def_readwrite("temperature", &oaa::GenerationConfig::temperature);

    py::class_<oaa::RuntimeStats>(m, "RuntimeStats")
        .def_readonly("model_loaded", &oaa::RuntimeStats::model_loaded)
        .def_readonly("generation_calls", &oaa::RuntimeStats::generation_calls)
        .def_readonly("model_path", &oaa::RuntimeStats::model_path);

    py::class_<oaa::Engine>(m, "Engine")
        .def(py::init<>())
        .def("load_model", &oaa::Engine::load_model, py::arg("path"))
        .def("unload", &oaa::Engine::unload)
        .def(
            "generate",
            &oaa::Engine::generate,
            py::arg("prompt"),
            py::arg("config") = oaa::GenerationConfig{})
        .def(
            "generate_stream",
            &oaa::Engine::generate_stream,
            py::arg("prompt"),
            py::arg("config") = oaa::GenerationConfig{})
        .def("get_stats", &oaa::Engine::get_stats)
        .def("loaded", &oaa::Engine::loaded)
        .def("status", &oaa::Engine::status);

    m.def("cuda_compiled", &oaa::CudaRuntime::compiled);
    m.def("cuda_available", &oaa::CudaRuntime::available);
    m.def(
        "cuda_vector_add",
        &oaa::CudaRuntime::vector_add,
        py::arg("lhs"),
        py::arg("rhs"));
}
