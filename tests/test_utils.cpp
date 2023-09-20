#include "test_utils.hpp"

std::ostream& operator<<(std::ostream& os, const Matrix& m) {
    for(const auto& row : m.data){
        for(const auto& val : row)
            os << val << "\t";
        os << '\n';
    }
    return os;
}
