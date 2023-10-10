#include "layer.hpp"
#include <cmath>
#include <random>
#include <iostream>
#include <algorithm>

Layer::Layer(int size, int inputs, std::shared_ptr<ActivationFunction> activationFunc, InitializationType initType, double min, double max) {
    if (min > max) {
        throw std::invalid_argument("Min cannot be greater than Max");
    }
    initializeNeurons(size, inputs, activationFunc, initType, min, max);
}

void Layer::initializeNeurons(int size, int inputs, std::shared_ptr<ActivationFunction> activationFunc, InitializationType initType, double min, double max) {
    neurons.resize(size);

    std::default_random_engine generator;
    std::uniform_real_distribution<double> uniform_distribution(min, max);
    std::normal_distribution<double> normal_distribution(0.0, sqrt(2.0 / (inputs + size)));
    
    std::vector<double> weights(inputs);
    double bias;
    for (Neuron &neuron: neurons) {
        switch (initType) {
            case InitializationType::Random:
                std::generate(weights.begin(), weights.end(), [&]() { return uniform_distribution(generator) - 0.5; });
                bias = uniform_distribution(generator) - 0.5;
                break;
            case InitializationType::RandomRange:
                std::generate(weights.begin(), weights.end(), [&]() { return uniform_distribution(generator); });
                bias = uniform_distribution(generator);
                break;
            case InitializationType::Zero:
                std::fill(weights.begin(), weights.end(), 0.0);
                bias = 0.0;
                break;
            case InitializationType::Xavier:
                std::generate(weights.begin(), weights.end(), [&]() { return normal_distribution(generator); });
                bias = normal_distribution(generator);
                break;
        }
        neuron = Neuron(weights, bias, activationFunc);
    }
}

std::vector<double> Layer::calculateLayerOutput(const std::vector<double>& inputs) {
    if (!neurons.empty() && inputs.size() != neurons[0].getWeights().size()){
        throw std::invalid_argument("Size of inputs does not match size of weights");
    }
    std::vector<double> output;
    output.reserve(neurons.size());
    for(auto& neuron : neurons)
        output.push_back(neuron.calculateOutput(inputs));
    return output;
}