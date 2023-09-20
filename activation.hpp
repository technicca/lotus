#pragma once
#include "matrices.hpp"

class Activation {
public:
    static Matrix apply_sigmoid(const Matrix& matrix);
    static Matrix apply_sigmoid_derivative(const Matrix& matrix);

private:
    static float sigmoid(float x);
    static float sigmoid_derivative(float x);
};
