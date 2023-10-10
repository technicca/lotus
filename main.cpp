#include "layer.hpp"
#include "neuron.hpp"
#include "activation.hpp"
#include <memory>
#include <iostream>
#include <ctime>

int main() {
    // Create the activation function
    std::shared_ptr<ActivationFunction> sigmoid = std::make_shared<Sigmoid>();

    // Create the layers
    Layer inputLayer(2, 2, sigmoid); // 2 neurons, each with 2 inputs
    Layer hiddenLayer(2, 2, sigmoid); // 2 neurons, each with 2 inputs
    Layer outputLayer(1, 2, sigmoid); // 1 neuron with 2 inputs

    // For XOR problem, the training data would be:
    // Inputs: [0, 0], [0, 1], [1, 0], [1, 1]
    // Outputs: [0], [1], [1], [0]
    srand(time(0));

    // Let's test with input [0, 1]
    std::vector<double> inputs = {0, 1};

    // Feed inputs through the input layer
    std::vector<double> inputLayerOutput = inputLayer.calculateLayerOutput(inputs);

    // Feed output of input layer as input to hidden layer
    std::vector<double> hiddenLayerOutput = hiddenLayer.calculateLayerOutput(inputLayerOutput);

    // Feed output of hidden layer as input to output layer
    std::vector<double> output = outputLayer.calculateLayerOutput(hiddenLayerOutput);

    std::cout << "Output: " << output[0] << std::endl;

    return 0;
}
