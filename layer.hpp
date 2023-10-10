#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>
#include <memory>
#include "neuron.hpp"
#include "activation.hpp"

class Layer {
public:
    Layer(int size, int inputs, std::shared_ptr<ActivationFunction> activationFunc);
    std::vector<double> calculateLayerOutput(const std::vector<double>& inputs);    
    Neuron& operator[](std::size_t idx) { return neurons[idx]; }
    const Neuron& operator[](std::size_t idx) const { return neurons[idx]; }
    std::size_t size() const { return neurons.size(); }

    std::vector<Neuron>::iterator begin() { return neurons.begin(); }
    std::vector<Neuron>::const_iterator begin() const { return neurons.begin(); }
    std::vector<Neuron>::iterator end() { return neurons.end(); }
    std::vector<Neuron>::const_iterator end() const { return neurons.end(); }

private:
    std::vector<Neuron> neurons; 
    void initializeNeurons(int size, int inputs, std::shared_ptr<ActivationFunction> activationFunc);
};

#endif // LAYER_HPP