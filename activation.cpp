#include "activation.hpp"
#include <cmath>

float Activation::sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

float Activation::sigmoid_derivative(float x) {
    return x * (1 - x);
}

Matrix Activation::apply_sigmoid(const Matrix& matrix) {
    Matrix result(matrix.getNumRows(), matrix.getNumColumns());
    for(int i = 0; i < matrix.getNumRows(); i++)
        for(int j = 0; j < matrix.getNumColumns(); j++)
            result.set(i, j, sigmoid(matrix.get(i, j)));
    return result;
}

Matrix Activation::apply_sigmoid_derivative(const Matrix& matrix) {
    Matrix result(matrix.getNumRows(), matrix.getNumColumns());
    for(int i = 0; i < matrix.getNumRows(); i++)
        for(int j = 0; j < matrix.getNumColumns(); j++)
            result.set(i, j, sigmoid_derivative(matrix.get(i, j)));
    return result;
}
