#include "GoogLeNetService.hpp"
#include <stdexcept>

namespace domain {

GoogLeNetService::GoogLeNetService(
    std::unique_ptr<ports::ModelLoaderPort> model_loader,
    std::unique_ptr<ports::ImageProcessorPort> image_processor,
    std::unique_ptr<ports::InferencePort> inference_engine
) : model_loader_(std::move(model_loader)),
    image_processor_(std::move(image_processor)),
    inference_engine_(std::move(inference_engine)) {}

std::vector<std::pair<std::string, float>> GoogLeNetService::classify(
    const std::string& model_path, 
    const std::string& image_path, 
    const std::vector<std::string>& class_names
) {
    try {
        // Cargar modelo
        auto model = model_loader_->load(model_path);
        model.eval();
        
        // Preprocesar imagen
        auto input_tensor = image_processor_->preprocess(image_path);
        
        // Ejecutar inferencia
        return inference_engine_->infer(model, input_tensor, class_names);
    } catch (const std::exception& e) {
        throw std::runtime_error("Error en clasificación: " + std::string(e.what()));
    }
}

} // namespace domain