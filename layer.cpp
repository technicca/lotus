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
    
    // Each initialization type has its own distribution
    std::uniform_real_distribution<double> uniform_distribution(min, max);
    std::normal_distribution<double> normal_distribution(0.0, sqrt(2.0 / (inputs + size)));
    std::uniform_real_distribution<double> he_uniform_distribution(0.0, sqrt(6.0 / (inputs + size)));
    std::normal_distribution<double> he_normal_distribution(0.0, sqrt(2.0 / (inputs + size)));

    std::vector<double> weights(inputs);
    double bias;
    for (Neuron &neuron: neurons) { 
        switch (initType) {
            // Random initialization: breaks symmetry between neurons but may lead to vanishing/exploding gradients
            case InitializationType::Random:
                if (min == 0.0 && max == 0.0) {
                    std::cout << "Range for Random activation function is not defined. Defaulting to [-0.5, 0.5]" << std::endl;
                    min = -0.5;
                    max = 0.5;
                    std::uniform_real_distribution<double> random_distribution(min, max);
                    std::generate(weights.begin(), weights.end(), [&]() { return random_distribution(generator); });
                    bias = random_distribution(generator);
                } else {
                    std::uniform_real_distribution<double> random_distribution(min, max);
                    std::generate(weights.begin(), weights.end(), [&]() { return random_distribution(generator); });
                    bias = random_distribution(generator);
                }
                break;

            case InitializationType::Zero:
                // Zero initialization: easy to implement but may lead to identical neuron learning
                std::fill(weights.begin(), weights.end(), 0.0);
                bias = 0.0;
                break;
            case InitializationType::Xavier:
                // Xavier initialization: takes into account the number of inputs/outputs, suitable for sigmoid/tanh activations
                if (min != 0.0 || max != 0.0) {
                    std::cout << "Range for Xavier activation function is not defined. Defaulting to normal distribution with mean 0 and standard deviation sqrt(2.0 / (inputs + size))" << std::endl;
                }
                std::generate(weights.begin(), weights.end(), [&]() { return normal_distribution(generator); });
                bias = normal_distribution(generator);
                break;
            case InitializationType::HeUniform:
                if (min != 0.0 || max != 0.0) {
                    std::cout << "Range for He Uniform activation function is not user-defined. Defaulting to uniform distribution between 0 and sqrt(6.0 / (inputs + size))" << std::endl;
                }
                std::generate(weights.begin(), weights.end(), [&]() { return he_uniform_distribution(generator); });
                bias = he_uniform_distribution(generator);
                break;
            case InitializationType::HeNormal:
                if (min != 0.0 || max != 0.0) {
                    std::cout << "Range for He Normal activation function is not user-defined. Defaulting to normal distribution with mean 0 and standard deviation sqrt(2.0 / (inputs + size))" << std::endl;
                }
                std::generate(weights.begin(), weights.end(), [&]() { return he_normal_distribution(generator); });
                bias = he_normal_distribution(generator);
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