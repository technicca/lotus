#include "activation_factory.hpp"

std::shared_ptr<ActivationFunction> use_tanh() {
    return std::make_shared<Tanh>();
}

std::shared_ptr<ActivationFunction> use_sigmoid() {
    return std::make_shared<Sigmoid>();
}
