#include <vector>

class Neuron {
public:
    Neuron(const std::vector<float>& weights, float bias);
    std::vector<float> getWeights() const;
    void setWeights(const std::vector<float>& weights);
    float getBias() const;
    void setBias(float bias);
    float sigmoid(float x) const;
    float calculateOutput(const std::vector<float>& inputs) const;

private:
    std::vector<float> weights;
    float bias;
};
