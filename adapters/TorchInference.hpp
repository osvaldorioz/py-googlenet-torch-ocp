#ifndef TORCH_INFERENCE_HPP
#define TORCH_INFERENCE_HPP

#include "../ports/InferencePort.hpp"
#include <torch/torch.h>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <string>

namespace adapters {

class TorchInference : public ports::InferencePort {
public:
    std::vector<std::pair<std::string, float>> infer(
        torch::jit::script::Module& model, 
        const torch::Tensor& input, 
        const std::vector<std::string>& class_names
    ) override {
        try {
            torch::NoGradGuard no_grad;
            
            // Ejecutar inferencia
            auto output = model.forward({input}).toTensor();
            auto probabilities = torch::softmax(output, 1).squeeze(0);
            
            // Obtener top-5 predicciones
            auto [values, indices] = probabilities.topk(5);
            std::vector<std::pair<std::string, float>> results;
            
            for (int i = 0; i < 5; ++i) {
                int idx = indices[i].item<int>();
                float prob = values[i].item<float>();
                std::string class_name = (idx < class_names.size()) ? class_names[idx] : "Unknown";
                results.emplace_back(class_name, prob);
            }
            
            return results;
        } catch (const std::exception& e) {
            throw std::runtime_error("Error en inferencia: " + std::string(e.what()));
        }
    }
};

} // namespace adapters

#endif // TORCH_INFERENCE_HPP