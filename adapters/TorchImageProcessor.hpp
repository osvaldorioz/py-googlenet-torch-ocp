#ifndef TORCH_IMAGE_PROCESSOR_HPP
#define TORCH_IMAGE_PROCESSOR_HPP

#include "../ports/ImageProcessorPort.hpp"
#include <torch/torch.h>
#include <stdexcept>
#include <string>

namespace adapters {

class TorchImageProcessor : public ports::ImageProcessorPort {
public:
    torch::Tensor preprocess(const std::string& tensor_path) override {
        try {
            // Leer el archivo como un flujo de bytes
            std::ifstream stream(tensor_path, std::ios::binary);
            if (!stream) {
                throw std::runtime_error("No se pudo abrir el archivo: " + tensor_path);
            }

            // Cargar el objeto serializado como un IValue
            c10::IValue ivalue = torch::pickle_load({std::istreambuf_iterator<char>(stream),
                                                     std::istreambuf_iterator<char>()});

            // Verificar que el IValue sea un tensor
            if (!ivalue.isTensor()) {
                throw std::runtime_error("El archivo no contiene un tensor válido");
            }

            // Convertir el IValue a un tensor
            torch::Tensor tensor = ivalue.toTensor();

            // Verificar la forma del tensor
            if (tensor.sizes() != torch::IntArrayRef({1, 3, 224, 224})) {
                throw std::runtime_error("Tensor tiene forma incorrecta: esperado [1, 3, 224, 224]");
            }

            return tensor;
        } catch (const std::exception& e) {
            throw std::runtime_error("Error al cargar el tensor: " + std::string(e.what()));
        }
    }
};

} // namespace adapters

#endif // TORCH_IMAGE_PROCESSOR_HPP