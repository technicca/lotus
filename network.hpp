#include <vector>
#include "layer.hpp"

class Network {
public:
    Network(const std::vector<int>& topology);
    void feedForward(const std::vector<float>& input);
    void backPropagate(const std::vector<float>& target);
    void updateWeights(float eta);
    std::vector<float> getResults() const;
private:
    std::vector<Layer> layers;
};
