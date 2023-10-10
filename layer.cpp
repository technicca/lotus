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
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, sqrt(2.0 / (inputs + size)));
    for (Neuron &neuron: neurons) {
        std::vector<double> weights(inputs);
        for (double &weight : weights) {
            weight = distribution(generator);
        }
        double bias = distribution(generator);
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
