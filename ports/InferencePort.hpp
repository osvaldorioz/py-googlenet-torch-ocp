#ifndef INFERENCE_PORT_HPP
#define INFERENCE_PORT_HPP

#include <torch/torch.h>
#include <vector>
#include <string>

namespace ports {

class InferencePort {
public:
    virtual std::vector<std::pair<std::string, float>> infer(
        torch::jit::script::Module& model, 
        const torch::Tensor& input, 
        const std::vector<std::string>& class_names) = 0;
    virtual ~InferencePort() = default;
};

} // namespace ports

#endif // INFERENCE_PORT_HPP