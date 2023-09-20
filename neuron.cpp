#include <cmath>
#include "neuron.hpp"
#include <numeric>

Neuron::Neuron(const std::vector<float>& weights, float bias) : weights(weights), bias(bias) {}

std::vector<float> Neuron::getWeights() const {
    return weights;
}

void Neuron::setWeights(const std::vector<float>& weights) {
    this->weights = weights;
}

float Neuron::getBias() const {
     return bias;
}

void Neuron::setBias(float bias) {
    this->bias = bias;
}

float Neuron::calculateOutput(const std::vector<float>& inputs) const {
    float sum = std::inner_product(inputs.begin(), inputs.end(), weights.begin(), 0.0f);
    return sigmoid(sum + bias);
}

float Neuron::sigmoid(float x) const {
    return 1 / (1 + exp(-x));
}
