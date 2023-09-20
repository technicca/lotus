#include <vector>
#include "neuron.hpp"

class Layer {
public:
    Layer(int size, int inputs);
    void addNeuron(const Neuron& neuron);
    std::vector<float> calculateLayerOutput(const std::vector<float>& inputs);

private:
    std::vector<Neuron> neurons; // This layer's neurons
};
