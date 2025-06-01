#ifndef GOOGLENET_SERVICE_HPP
#define GOOGLENET_SERVICE_HPP

#include "../ports/ModelLoaderPort.hpp"
#include "../ports/ImageProcessorPort.hpp"
#include "../ports/InferencePort.hpp"
#include <string>
#include <vector>
#include <memory>

namespace domain {

class GoogLeNetService {
public:
    GoogLeNetService(
        std::unique_ptr<ports::ModelLoaderPort> model_loader,
        std::unique_ptr<ports::ImageProcessorPort> image_processor,
        std::unique_ptr<ports::InferencePort> inference_engine
    );

    std::vector<std::pair<std::string, float>> classify(
        const std::string& model_path, 
        const std::string& image_path, 
        const std::vector<std::string>& class_names
    );

private:
    std::unique_ptr<ports::ModelLoaderPort> model_loader_;
    std::unique_ptr<ports::ImageProcessorPort> image_processor_;
    std::unique_ptr<ports::InferencePort> inference_engine_;
};

} // namespace domain

#include "GoogLeNetService.tpp"

#endif // GOOGLENET_SERVICE_HPP