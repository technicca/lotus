#include <iostream>
#include "layer.hpp"
#include "activation.hpp"
#include <ctime>


void printVector(const std::vector<double>& vec){
    for(double v : vec){
        std::cout << v << ' ';
    }
    std::cout << '\n';
}

void setLayerWeightsAndBias(Layer& layer, const std::vector<double>& weights, double bias) {
    for (Neuron& neuron : layer) {
        neuron.setWeights(weights);
        neuron.setBias(bias);
    }
}

void testKnownReluLayer(Layer& layer, const std::vector<double>& inputs) {
    std::cout << "Testing Relu Layer with known weights and bias:\n";
    std::cout << "Output: ";
    printVector(layer.calculateLayerOutput(inputs));
    std::cout << '\n';
}

int main() {
    std::srand(std::time(0));
    std::vector<double> inputs = {0.5, 0.3};

    //Generate some additional inputs for more thorough testing
    std::vector<double> largerInputs = {0.5, 0.3, 0.7, 0.2, 0.6};
    std::vector<double> edgeInputs = {0, 10000, -10000};

    // Test Sigmoid activation function
    std::shared_ptr<ActivationFunction> sigmoid = std::make_shared<Sigmoid>();
    Layer sigmoidLayer(2, inputs.size(), sigmoid);
    Layer sigmoidLayerLarger(10, inputs.size(), sigmoid);
    Layer sigmoidLayerMoreInputs(2, largerInputs.size(), sigmoid);
    Layer sigmoidLayerEdge(2, edgeInputs.size(), sigmoid);
    std::cout << "Sigmoid Layer Output: ";
    printVector(sigmoidLayer.calculateLayerOutput(inputs));
    std::cout << "Sigmoid Layer (Larger) Output: ";
    printVector(sigmoidLayerLarger.calculateLayerOutput(inputs));
    std::cout << "Sigmoid Layer (More Inputs) Output: ";
    printVector(sigmoidLayerMoreInputs.calculateLayerOutput(largerInputs));
    std::cout << "Sigmoid Layer (Edge Inputs) Output: ";
    printVector(sigmoidLayerEdge.calculateLayerOutput(edgeInputs));

    // Repeat for Tanh activation function
    std::shared_ptr<ActivationFunction> tanhFunc = std::make_shared<Tanh>();
    Layer tanhLayer(2, inputs.size(), tanhFunc);
    Layer tanhLayerLarger(10, inputs.size(), tanhFunc);
    Layer tanhLayerMoreInputs(2, largerInputs.size(), tanhFunc);
    Layer tanhLayerEdge(2, edgeInputs.size(), tanhFunc);
    std::cout << "Tanh Layer Output: ";
    printVector(tanhLayer.calculateLayerOutput(inputs));
    std::cout << "Tanh Layer (Larger) Output: ";
    printVector(tanhLayerLarger.calculateLayerOutput(inputs));
    std::cout << "Tanh Layer (More Inputs) Output: ";
    printVector(tanhLayerMoreInputs.calculateLayerOutput(largerInputs));
    std::cout << "Tanh Layer (Edge Inputs) Output: ";
    printVector(tanhLayerEdge.calculateLayerOutput(edgeInputs));

    // Repeat for Relu activation function
    std::shared_ptr<ActivationFunction> reluFunc = std::make_shared<Relu>();
    Layer reluLayer(2, inputs.size(), reluFunc);
    Layer reluLayerLarger(10, inputs.size(), reluFunc);
    Layer reluLayerMoreInputs(2, largerInputs.size(), reluFunc);
    Layer reluLayerEdge(2, edgeInputs.size(), reluFunc);
    std::cout << "Relu Layer Output: ";
    printVector(reluLayer.calculateLayerOutput(inputs));
    std::cout << "ReluLayer (Larger) Output: ";
    printVector(reluLayerLarger.calculateLayerOutput(inputs));
    std::cout << "Relu Layer (More Inputs) Output: ";
    printVector(reluLayerMoreInputs.calculateLayerOutput(largerInputs));
    std::cout << "Relu Layer (Edge Inputs) Output: ";
    printVector(reluLayerEdge.calculateLayerOutput(edgeInputs));

    // additional relu tests

    std::vector<double> knownWeights = {0.5, -0.4};
    double knownBias = 0.2;

    setLayerWeightsAndBias(reluLayer, knownWeights, knownBias);

    // Test layers with known weights/bias and inputs
    testKnownReluLayer(reluLayer, {0.1, 0});      // output should be: max(0, 0.5*0.1 + -0.4*0 + 0.2) = max(0, 0.05) = 0.05
    testKnownReluLayer(reluLayer, {-0.1, 3});    // output should be: max(0, 0.5*(-0.1) + -0.4*3 + 0.2) = max(0, -0.05 - 1.2 + 0.2) = max(0, -1.05) = 0
    testKnownReluLayer(reluLayer, {2, -3});      // output should be: max(0, 0.5*2 + -0.4*(-3) + 0.2) = max(0, 1 + 1.2 + 0.2) = max(0, 2.4) = 2.4
    return 0;
}
