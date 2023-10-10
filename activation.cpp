#include "activation.hpp"
#include <cmath>

double Sigmoid::apply(double x) const {
    return 1 / (1 + std::exp(-x));
}

double Sigmoid::derivative(double x) const {
    double sigmoid = apply(x);
    return sigmoid * (1 - sigmoid);
}

double Tanh::apply(double x) const {
    return std::tanh(x);
}

double Tanh::derivative(double x) const {
    double tanh = apply(x);
    return 1 - tanh * tanh;
}

