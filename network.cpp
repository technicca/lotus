#include "layer.hpp"
#include <vector>

class Network {
public:
    Network(double learningRate) : learningRate(learningRate) {}
    void addLayer(Layer layer) { layers.push_back(layer); }

    // Forward pass
    std::vector<double> feedForward(const std::vector<double>& inputs) {
        layerOutputs.clear(); // Clear layerOutputs
        std::vector<double> layerInputs = inputs;
        for (Layer& layer : layers) {
            layerInputs = layer.calculateLayerOutput(layerInputs);
            layerOutputs.push_back(layerInputs); // Store the outputs of each layer
        }
        return layerInputs;
    }

    // Mean squared error
    double calculateLoss(const std::vector<double>& outputs, const std::vector<double>& targetOutputs) {
        double totalError = 0.0;
        for (size_t i = 0; i < outputs.size(); i++) {
            double error = targetOutputs[i] - outputs[i];
            totalError += error * error;
        }
        return totalError / outputs.size();
    }

    void backpropagate(const std::vector<double>& targetOutputs) {
    std::vector<double> errors;
    for (size_t i = 0; i < layers.back().size(); i++) {
        double output = layers.back()[i].getOutput();
        double target = targetOutputs[i];
        double derivative = layers.back()[i].getActivationDerivative(output);
        errors.push_back((target - output) * derivative);
    }

    // Update weights and biases for the output layer
    for (size_t j = 0; j < layers.back().size(); j++) {
        auto& weights = layers.back()[j].getWeightsRef();
        for (auto& weight : weights) {
            weight += learningRate * errors[j];
        }
        layers.back()[j].setBias(layers.back()[j].getBias() + learningRate * errors[j]);
    }

    // Propagate the errors back through the network and update weights and biases
    for (int i = layers.size() - 2; i >= 0; i--) {
        std::vector<double> nextLayerErrors;
        for (size_t j = 0; j < layers[i].size(); j++) {
            double error = 0.0;
            for (size_t k = 0; k < layers[i + 1].size(); k++) {
                error += errors[k] * layers[i + 1][k].getWeights()[j];
            }
            // Update weights and biases
            auto& weights = layers[i + 1][j].getWeightsRef();
            for (auto& weight : weights) {
                weight += learningRate * error;
            }
            layers[i + 1][j].setBias(layers[i + 1][j].getBias() + learningRate * error);
            nextLayerErrors.push_back(error);
        }
        errors = nextLayerErrors;
    }
}



private:
    std::vector<Layer> layers;
    double learningRate;
    std::vector<std::vector<double>> layerOutputs;
};
