#ifndef MATRICES_H
#define MATRICES_H

#include <vector>
#include <ostream>


class Matrix {
public:
    Matrix(int rows, int cols); // random matrix 
    Matrix(std::initializer_list<std::initializer_list<float>> list); // user-specified matrix
    Matrix(int rows, int cols, float value); // constant matrix
    Matrix multiply(const Matrix& other) const;
    Matrix add(const Matrix& other) const;
    Matrix subtract(const Matrix& other) const;
    Matrix elementwise_multiply(const Matrix& other) const;
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    int getNumRows() const;
    int getNumColumns() const;
    float get(unsigned row, unsigned col) const;
    void set(unsigned row, unsigned col, float value);

private:
    std::vector<std::vector<float>> data;
    float random_float(float min, float max);
};

#endif
