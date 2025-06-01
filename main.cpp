#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "domain/GoogLeNetService.hpp"
#include "adapters/TorchModelLoader.hpp"
#include "adapters/TorchImageProcessor.hpp"
#include "adapters/TorchInference.hpp"

// g++ -shared -fPIC -o googlenet.so main.cpp -I/home/hadoop/Documentos/cpp_programs/pybind/py-googlenet-torch/myenv/lib/python3.12/site-packages/pybind11/include -I/home/hadoop/libtorch/include -I/home/hadoop/libtorch/include/torch/csrc/api/include -I/usr/include/python3.12 -L/home/hadoop/libtorch/lib -ltorch -ltorch_cpu -lc10 -std=c++17 -Wl,-rpath,/home/hadoop/libtorch/lib

namespace py = pybind11;

// Forzar la generación de la vtable
namespace adapters {
    void force_vtable_generation() {
        TorchImageProcessor dummy_processor;
        TorchModelLoader dummy_loader;
        TorchInference dummy_inference;
    }
}

PYBIND11_MODULE(googlenet, m) {
    // Llamar a force_vtable_generation para evitar eliminación de código
    adapters::force_vtable_generation();

    py::class_<domain::GoogLeNetService>(m, "GoogLeNetService")
        .def(py::init([](const std::string& /* dummy */) {
            return std::make_unique<domain::GoogLeNetService>(
                std::make_unique<adapters::TorchModelLoader>(),
                std::make_unique<adapters::TorchImageProcessor>(),
                std::make_unique<adapters::TorchInference>()
            );
        }))
        .def("classify", &domain::GoogLeNetService::classify);
}