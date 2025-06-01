#ifndef TORCH_MODEL_LOADER_HPP
#define TORCH_MODEL_LOADER_HPP

#include "../ports/ModelLoaderPort.hpp"
#include <torch/script.h>
#include <stdexcept>
#include <string>

namespace adapters {

class TorchModelLoader : public ports::ModelLoaderPort {
public:
    torch::jit::script::Module load(const std::string& model_path) override {
        try {
            return torch::jit::load(model_path);
        } catch (const std::exception& e) {
            throw std::runtime_error("Error al cargar el modelo: " + std::string(e.what()));
        }
    }
};

} // namespace adapters

#endif // TORCH_MODEL_LOADER_HPP