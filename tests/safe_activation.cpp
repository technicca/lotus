#include <memory>
#include <string>
#include "activation.hpp"
#include "neuron.hpp"

std::shared_ptr<ActivationFunction> make_activation_function(const std::string& type) {
    if (type == "Sigmoid") {
        return std::make_shared<Sigmoid>();
    } else if (type == "Tanh") {
        return std::make_shared<Tanh>();
    } else if (type == "Relu") {
        return std::make_shared<Relu>();
    } else {
        throw std::invalid_argument("Unknown activation function type: " + type);
    }
}

int main() {
    std::vector<double> weights = {0.4, 0.6, 0.2};
    double bias = 0.3;

    auto sigmoid = make_activation_function("Sigmoid");
    Neuron neuron(weights, bias, sigmoid);

}
