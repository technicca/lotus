#include "matrices.hpp"
#include <iostream>

void printMatrix(const Matrix& matrix) {
    for (unsigned i = 0; i < matrix.getNumRows(); ++i) {
        for (unsigned j = 0; j < matrix.getNumColumns(); ++j) {
            std::cout << matrix.get(i, j) << ' ';
        }
        std::cout << '\n';
    }
}

void testMatrixOperations() {
    Matrix m1(3, 3, 0); // 3x3 matrix filled with zeros
    if (m1.getNumRows() == 3 && m1.getNumColumns() == 3 && m1.get(0, 0) == 0) {
        std::cout << "Zero matrix constructor test passed\n";
    } else {
        std::cout << "Zero matrix constructor test failed\n";
    }

    Matrix m2 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}; // user-specified matrix
    if (m2.getNumRows() == 3 && m2.getNumColumns() == 3 && m2.get(0, 0) == 1) {
        std::cout << "User-specified matrix constructor test passed\n";
    } else {
        std::cout << "User-specified matrix constructor test failed\n";
    }

    Matrix m3 = m1 + m2; // test addition
    if (m3.get(0, 0) == 1 && m3.get(1, 1) == 5 && m3.get(2, 2) == 9) {
        std::cout << "Matrix addition test passed\n";
    } else {
        std::cout << "Matrix addition test failed\n";
    }

    Matrix m4 = m2 - m1; // test subtraction
    if (m4.get(0, 0) == 1 && m4.get(1, 1) == 5 && m4.get(2, 2) == 9) {
        std::cout << "Matrix subtraction test passed\n";
    } else {
        std::cout << "Matrix subtraction test failed\n";
    }

    Matrix m5 = m2 * m2; // Test multiplication
    if (m5.get(0, 0) == 30 && m5.get(0, 1) == 36 && m5.get(0, 2) == 42 &&
        m5.get(1, 0) == 66 && m5.get(1, 1) == 81 && m5.get(1, 2) == 96 &&
        m5.get(2, 0) == 102 && m5.get(2, 1) == 126 && m5.get(2, 2) == 150) {
        std::cout << "Matrix multiplication test passed\n";
    } else {
        std::cout << "Matrix multiplication test failed\n";
    }


    std::cout << "Matrix multiplication result:\n";
    printMatrix(m5);
    std::cout << "\n";

}

int main() {
    testMatrixOperations();
    return 0;
}
