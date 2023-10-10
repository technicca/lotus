#include "layer.hpp"
#include "network.hpp"
#include <vector>

Network::Network(double learningRate) : learningRate(learningRate) {}

void Network::addLayer(Layer layer) { 
    layers.push_back(layer); 
}

std::vector<double> Network::feedForward(const std::vector<double>& inputs) {
    layerOutputs.clear(); // Clear layerOutputs
    std::vector<double> layerInputs = inputs;
    for (Layer& layer : layers) {
        layerInputs = layer.calculateLayerOutput(layerInputs);
        layerOutputs.push_back(layerInputs); // Store the outputs of each layer
    }
    return layerInputs;
}

double Network::calculateLoss(const std::vector<double>& outputs, const std::vector<double>& targetOutputs) {
    double totalError = 0.0;
    for (size_t i = 0; i < outputs.size(); i++) {
        double error = targetOutputs[i] - outputs[i];
        totalError += error * error;
    }
    return totalError / outputs.size();
}

void Network::backpropagate(const std::vector<double>& targetOutputs) {
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
        for (size_t k = 0; k < weights.size(); k++) {
            weights[k] += learningRate * errors[j] * layerOutputs[layerOutputs.size() - 2][k];
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
            error *= layers[i][j].getActivationDerivative(layers[i][j].getOutput());
            // Update weights and biases
            auto& weights = layers[i][j].getWeightsRef();
            for (size_t k = 0; k < weights.size(); k++) {
                weights[k] += learningRate * error * (i > 0 ? layerOutputs[i - 1][k] : layerOutputs[0][k]);
            }
            layers[i][j].setBias(layers[i][j].getBias() + learningRate * error);
            nextLayerErrors.push_back(error);
        }
        errors = nextLayerErrors;
    }
}
