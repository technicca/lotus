#include <vector>
#include <memory>
#include "activation.hpp"

class Neuron {
public:
    Neuron();
    Neuron(const std::vector<double>& weights, double bias, std::shared_ptr<ActivationFunction> func);
    std::vector<double> getWeights() const;
    void setWeights(const std::vector<double>& weights);
    double getBias() const;
    void setBias(double bias);
    double calculateOutput(const std::vector<double>& inputs) const;

private:
    std::vector<double> weights;
    double bias;
    std::shared_ptr<ActivationFunction> activationFunc; // memory management
};

