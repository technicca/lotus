#pragma once
#include "matrices.hpp"

class ActivationFunction {
public:
    virtual ~ActivationFunction() = default;
    virtual double apply(double x) const = 0;
    virtual double derivative(double x) const = 0;
};


class Sigmoid : public ActivationFunction {
public:
    double apply(double x) const override;
    double derivative(double x) const override;
};

class Tanh : public ActivationFunction {
public:
    double apply(double x) const override;
    double derivative(double x) const override;
};

class Relu : public ActivationFunction {
public:
    double apply(double x) const override;
};


// remember to ensure the memory management of activation functions if they are created dynamically, nor to use them after their lifetimes if they are created on stack or statically.