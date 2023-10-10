#include "layer.hpp"
#include <cmath>
#include <random>
#include <iostream>
#include <algorithm>

Layer::Layer(int size, int inputs, std::shared_ptr<ActivationFunction> activationFunc) {
    initializeNeurons(size, inputs, activationFunc);
}

void Layer::initializeNeurons(int size, int inputs, std::shared_ptr<ActivationFunction> activationFunc) {
    neurons.resize(size);
    for (Neuron &neuron: neurons) {
        std::vector<double> weights(inputs);
        std::generate(weights.begin(), weights.end(), [inputs]() { return ((static_cast<float>(rand()) / RAND_MAX) * 2 - 1) / sqrt(inputs); }); // weights randomly initialized 
        double bias = (static_cast<float>(rand()) / RAND_MAX) * 2 - 1; // bias randomly initialized
        neuron = Neuron(weights, bias, activationFunc);
    }
}


std::vector<double> Layer::calculateLayerOutput(const std::vector<double>& inputs) {
    if (!neurons.empty() && inputs.size() != neurons[0].getWeights().size()){
        throw std::invalid_argument("Size of inputs does not match size of weights");
    }
    std::vector<double> output;
    for(auto& neuron : neurons)
        output.push_back(neuron.calculateOutput(inputs));
    return output;
}
