#ifndef ACTIVATION_FACTORY_HPP
#define ACTIVATION_FACTORY_HPP

#include <memory>
#include "../activation.hpp"

std::shared_ptr<ActivationFunction> use_tanh();
std::shared_ptr<ActivationFunction> use_sigmoid();

#endif // ACTIVATION_FACTORY_HPP
