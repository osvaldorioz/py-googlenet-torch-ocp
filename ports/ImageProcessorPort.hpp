#ifndef IMAGE_PROCESSOR_PORT_HPP
#define IMAGE_PROCESSOR_PORT_HPP

#include <torch/torch.h>
#include <string>

namespace ports {

class ImageProcessorPort {
public:
    virtual torch::Tensor preprocess(const std::string& tensor_path) = 0;
    virtual ~ImageProcessorPort() = default;
};

} // namespace ports

#endif // IMAGE_PROCESSOR_PORT_HPP