#include "activation.hpp"
#include <cmath>
#include <algorithm>

double Sigmoid::apply(double x) const {
    return 1 / (1 + std::exp(-x));
}

double Tanh::apply(double x) const {
    return std::tanh(x);
}

double Relu::apply(double x) const {
    return std::max(0.0, x);
}

