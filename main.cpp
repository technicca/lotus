#include "network.hpp"
#include "layer.hpp"
#include "activation.hpp"
#include "utils/track_memory.hpp"
#include "utils/activation_factory.hpp"
#include <iostream>
#include <ctime>

int main() {
    srand(time(0));
    Network network(0.3); // Initialize network with a learning rate

    // Add layers
    network.addLayer(Layer(2, 2, use_tanh(), InitializationType::Xavier, -1.0, 1.0));
    network.addLayer(Layer(2, 2, use_tanh(), InitializationType::Xavier, -1.0, 1.0)); // Hidden layer
    network.addLayer(Layer(1, 2, use_sigmoid(), InitializationType::Xavier, -1.0, 1.0)); // Output layer


    // XOR dataset
    std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> outputs = {{0}, {1}, {1}, {0}};

    // Train the network
    for (int epoch = 0; epoch < 1000; epoch++) { // set the # of epochs
        double totalLoss = 0.0;
        for (size_t i = 0; i < inputs.size(); i++) {
            printMemoryUsage();
            auto output = network.feedForward(inputs[i]);
            totalLoss += network.calculateLoss(output, outputs[i]);
            network.backpropagate(outputs[i]);
        }
        if (epoch % 100 == 0) { // Print the loss every x epochs
            std::cout << "Epoch: " << epoch << " Loss: " << totalLoss / inputs.size() << std::endl;
        }
    }
    // Print weights and biases
    for (const auto& layer : network.getLayers()) {
        for (const auto& neuron : layer) {
            neuron.printWeightsAndBias();
        }
    }

    // Test the network
    for (const auto& input : inputs) {
        auto result = network.feedForward(input);
        std::cout << "Input: " << input[0] << ", " << input[1] << " Output: " << result[0] << std::endl;
    }

    return 0;
}

