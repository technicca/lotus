#ifndef MATRICES_H
#define MATRICES_H

#include <vector>

class Matrix {
public:
    Matrix(int rows, int cols); // random matrix
    Matrix(std::initializer_list<std::initializer_list<float>> list); // user-specified matrix in the format of Matrix m = {{1, 2}, {3, 4}};
    Matrix multiply(const Matrix& other);
    Matrix add(const Matrix& other);
    Matrix subtract(const Matrix& other);
    Matrix elementwise_multiply(const Matrix& other);
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
