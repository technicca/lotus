#include "layer.hpp"
#include <random>

Layer::Layer(int size, int inputs)
    : neurons(size, Neuron(std::vector<float>(inputs), 0)) { // Initialize neurons with random weights and 0 bias
}

void Layer::addNeuron(const Neuron& neuron) {
    neurons.push_back(neuron);
}

std::vector<float> Layer::calculateLayerOutput(const std::vector<float>& inputs) {
    std::vector<float> output;
    for (auto& neuron : neurons)
        output.push_back(neuron.calculateOutput(inputs));
    return output;
}
