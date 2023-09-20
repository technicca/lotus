#include "network.hpp"

Network::Network(const std::vector<int>& topology) {
    for (int i = 0; i < topology.size() - 1; ++i) {
        // Create a new Layer and add it to the layers vector.
        layers.push_back(Layer(topology[i], topology[i + 1]));
    }
}

void Network::feedForward(const std::vector<float>& input) {
    std::vector<float> layerInput = input;
    for (auto& layer : layers) {
        layerInput = layer.calculateLayerOutput(layerInput);
    }
}


void Network::backPropagate(const std::vector<float>& target) {
}

void Network::updateWeights(float eta) {
}

std::vector<float> Network::getResults() const {
}
