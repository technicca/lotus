#include <iostream>
#include <string>
#include <memory>
#include "activation.hpp"
#include "neuron.hpp"

std::shared_ptr<ActivationFunction> make_activation_function(const std::string& type) {
    if (type == "Sigmoid") {
        return std::make_shared<Sigmoid>();
    } else if (type == "Tanh") {
        return std::make_shared<Tanh>();
    } else if (type == "Relu") {
        return std::make_shared<Relu>();
    } else {
        throw std::invalid_argument("Unknown activation function type: " + type);
    }
}

void testNeuron(const std::vector<double>& weights, double bias, const std::vector<double>& inputs, const std::string& activationType) {
    std::cout << "-----------------------------------------\n";
    std::cout << "Testing Neuron with " << activationType << " activation function:\n";
    
    std::shared_ptr<ActivationFunction> activationFunc = make_activation_function(activationType);
    Neuron neuron(weights, bias, activationFunc);

    std::cout << "Initial state:\n";
    std::cout << "Weights: ";
    for (double weight : neuron.getWeights()) {
        std::cout << weight << ' ';
    }
    std::cout << "\nBias: ";
    std::cout << neuron.getBias() << '\n';
    std::cout << "Calculated output: " << neuron.calculateOutput(inputs) << "\n\n";
    
    std::cout << "After modifying weights and bias:\n";
    std::vector<double> newWeights = {0.5, 0.7, 0.3};
    neuron.setWeights(newWeights);
    neuron.setBias(0.4);

    std::cout << "Weights: ";
    for (double weight : neuron.getWeights()) {
        std::cout << weight << ' ';
    }
    std::cout << "\nBias: ";
    std::cout << neuron.getBias() << '\n';
    std::cout << "Calculated output: " << neuron.calculateOutput(inputs) << "\n";
    std::cout << "-----------------------------------------\n";
}

int main() {
    std::vector<double> weights = {0.4, 0.6, 0.2};
    double bias = 0.3;
    std::vector<double> inputs = {1.0, 2.0, 3.0};

    testNeuron(weights, bias, inputs, "Sigmoid");
    testNeuron(weights, bias, inputs, "Tanh");
    testNeuron(weights, bias, inputs, "Relu");

    return 0;
}
