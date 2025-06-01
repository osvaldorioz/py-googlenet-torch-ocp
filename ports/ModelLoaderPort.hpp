#ifndef MODEL_LOADER_PORT_HPP
#define MODEL_LOADER_PORT_HPP

#include <torch/script.h>
#include <string>

namespace ports {

class ModelLoaderPort {
public:
    virtual torch::jit::script::Module load(const std::string& model_path) = 0;
    virtual ~ModelLoaderPort() = default;
};

} // namespace ports

#endif // MODEL_LOADER_PORT_HPP