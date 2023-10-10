#include "layer.hpp"
#include "neuron.hpp"
#include "activation.hpp"
#include "network.hpp"
#include <memory>
#include <iostream>
#include <ctime>

int main() {
    Network network(0.5);  // Create a network with a learning rate of 0.5
    network.addLayer(Layer(2, 2, std::make_shared<Sigmoid>()));  // Add a hidden layer with 2 neurons, each having 2 inputs
    network.addLayer(Layer(1, 2, std::make_shared<Sigmoid>()));  // Add an output layer with 1 neuron with 2 inputs

    // The XOR dataset
    std::vector<std::vector<double>> inputs = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    std::vector<std::vector<double>> outputs = {
        {0},
        {1},
        {1},
        {0}
    };

    // Train the network
    for (int epoch = 0; epoch < 10000; epoch++) {
        for (size_t i = 0; i < inputs.size(); i++) {
            network.feedForward(inputs[i]);
            network.backpropagate(outputs[i]);
        }
    }

    // Test the network
    for (const auto& input : inputs) {
        auto result = network.feedForward(input);
        std::cout << "Input: [" << input[0] << ", " << input[1] << "], Output: " << result[0] << std::endl;
    }

    return 0;
}
