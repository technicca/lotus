#pragma once

#include <vector>
#include "layer.hpp"

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& topology, double learningRate);
    void feedForward(const std::vector<double>& inputValues);
    void backPropagation(const std::vector<double>& targetValues);
    void getResults(std::vector<double>& resultValues) const;

private:
    std::vector<Layer> layers;
    double error;
    double learningRate;
};
