#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "layer.hpp"
#include <vector>

class Network {
public:
    Network(double learningRate);
    void addLayer(Layer layer);
    std::vector<double> feedForward(const std::vector<double>& inputs);
    double calculateLoss(const std::vector<double>& outputs, const std::vector<double>& targetOutputs);
    void backpropagate(const std::vector<double>& targetOutputs);

private:
    std::vector<Layer> layers;
    double learningRate;
    std::vector<std::vector<double>> layerOutputs; // Add this line
};

#endif // NETWORK_HPP
