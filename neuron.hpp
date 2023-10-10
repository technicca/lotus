#ifndef NEURON_HPP
#define NEURON_HPP

#include <vector>
#include <memory>
#include "activation.hpp"

class Neuron {
public:
    Neuron();
    Neuron(const std::vector<double>& weights, double bias, std::shared_ptr<ActivationFunction> func);
    std::vector<double> getWeights() const;
    std::vector<double>& getWeightsRef();
    void updateWeights(const std::vector<double>& newWeights);
    void setWeights(const std::vector<double>& weights);
    double getBias() const;
    void setBias(double bias);
    double calculateOutput(const std::vector<double>& inputs);
    double getOutput() const;

    double getActivationDerivative(double x) const { // activation function derivative to use in network.cpp
        return activationFunc->derivative(x);
    }

    void printWeightsAndBias() const;
private:
    std::vector<double> weights;
    double bias;
    double output;
    std::shared_ptr<ActivationFunction> activationFunc; // memory management
};

#endif // NEURON_HPP
