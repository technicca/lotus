#include "matrices.hpp"
#include <random>
#include <stdexcept>
#include <ostream>

Matrix::Matrix(int rows, int cols) : data(rows, std::vector<float>(cols)) {
    for(int i=0; i<rows; i++)
        for(int j=0; j<cols; j++)
            data[i][j] = random_float(-1, 1);
} // random matrix constructor

Matrix::Matrix(std::initializer_list<std::initializer_list<float>> list)
    : data(list.begin(), list.end()) {} //  user-specified matrix constructor

Matrix::Matrix(int rows, int cols, float value) : data(rows, std::vector<float>(cols, value)) {} //constant matrix constructor

int Matrix::getNumRows() const {
    return data.size();
}

int Matrix::getNumColumns() const {
    // assuming each row has the same number of columns
    return data.empty() ? 0 : data[0].size();
}

float Matrix::random_float(float min, float max) {
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(min, max); // rage 0 - 1
    return dis(e);
}

Matrix Matrix::multiply(const Matrix& other) const {
    if (data[0].size() != other.data.size()) {
        throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
    }
    Matrix result(data.size(), other.data[0].size(), 0); // Initialize result matrix with zeros
    for(std::size_t i=0; i<data.size(); i++)
        for(std::size_t j=0; j<other.data[0].size(); j++)
            for(std::size_t k=0; k<data[0].size(); k++)
                result.data[i][j] += data[i][k] * other.data[k][j];
    return result;
}


Matrix Matrix::add(const Matrix& other) const {
    if (data.size() != other.data.size() || data[0].size() != other.data[0].size()) {
        throw std::invalid_argument("Both matrices must be of the same size");
    }
    Matrix result(data.size(), data[0].size());
    for(std::size_t i=0; i<data.size(); i++)
        for(std::size_t j=0; j<data[0].size(); j++)
            result.data[i][j] = data[i][j] + other.data[i][j];
    return result;
}

Matrix Matrix::subtract(const Matrix& other) const {
    if (data.size() != other.data.size() || data[0].size() != other.data[0].size()) {
        throw std::invalid_argument("Both matrices must be of the same size");
    }
    Matrix result(data.size(), data[0].size());
    for(std::size_t i=0; i<data.size(); i++)
        for(std::size_t j=0; j<data[0].size(); j++)
            result.data[i][j] = data[i][j] - other.data[i][j];
    return result;
}

// Operators to use +, -, * to perform matrix operations
Matrix Matrix::operator+(const Matrix& other) const {
    if (data.size() != other.data.size() || data[0].size() != other.data[0].size()) {
        throw std::invalid_argument("Both matrices must be of the same size");
    }
    Matrix result(data.size(), data[0].size());
    for(std::size_t i=0; i<data.size(); i++)
        for(std::size_t j=0; j<data[0].size(); j++)
            result.data[i][j] = data[i][j] + other.data[i][j];
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (data.size() != other.data.size() || data[0].size() != other.data[0].size()) {
        throw std::invalid_argument("Both matrices must be of the same size");
    }
    Matrix result(data.size(), data[0].size());
    for(std::size_t i=0; i<data.size(); i++)
        for(std::size_t j=0; j<data[0].size(); j++)
            result.data[i][j] = data[i][j] - other.data[i][j];
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (data[0].size() != other.data.size()) {
        throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
    }
    Matrix result(data.size(), other.data[0].size(), 0);  // Initialize result matrix with zeros
    for(std::size_t i=0; i<data.size(); i++)
        for(std::size_t j=0; j<other.data[0].size(); j++)
            for(std::size_t k=0; k<data[0].size(); k++)
                result.data[i][j] += data[i][k] * other.data[k][j];
    return result;
}



Matrix Matrix::elementwise_multiply(const Matrix& other) const {
    if (data.size() != other.data.size() || data[0].size() != other.data[0].size()) {
        throw std::invalid_argument("Both matrices must be of the same size");
    }
    Matrix result(data.size(), data[0].size());
    for(std::size_t i=0; i<data.size(); i++)
        for(std::size_t j=0; j<data[0].size(); j++)
            result.data[i][j] = data[i][j] * other.data[i][j];
    return result;
}

float Matrix::get(unsigned row, unsigned col) const {
    if(row >= data.size() || col >= data[0].size()) {
        throw std::out_of_range("Matrix indices out of range");
    }
    return data[row][col];
}

void Matrix::set(unsigned row, unsigned col, float value) {
    if(row >= data.size() || col >= data[0].size()) {
        throw std::out_of_range("Matrix indices out of range");
    }
    data[row][col] = value;
}
