#include <cmath>
#include "neuron.hpp"
#include <numeric>

Neuron::Neuron() : bias(0), activationFunc(std::make_shared<Sigmoid>()) {}

Neuron::Neuron(const std::vector<double>& weights, double bias, std::shared_ptr<ActivationFunction> func) : weights(weights), bias(bias), activationFunc(func) {}

std::vector<double> Neuron::getWeights() const {
    return weights;
}

void Neuron::setWeights(const std::vector<double>& weights) {
    this->weights = weights;
}

std::vector<double>& Neuron::getWeightsRef() {
    return weights;
}

void Neuron::updateWeights(const std::vector<double>& newWeights) {
    if (newWeights.size() != weights.size()) {
        throw std::invalid_argument("Size of new weights does not match size of current weights");
    }
    weights = newWeights;
}

double Neuron::getBias() const {
     return bias;
}

void Neuron::setBias(double bias) {
    this->bias = bias;
}

double Neuron::calculateOutput(const std::vector<double>& inputs) {
    if(inputs.size() != weights.size()){
        throw std::invalid_argument("Size of inputs does not match size of weights");
    }
    double sum = std::inner_product(inputs.begin(), inputs.end(), weights.begin(), 0.0);
    output = activationFunc->apply(sum + bias);
    return output;
}


double Neuron::getOutput() const {
    return output;
}
