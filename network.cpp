#include "network.hpp"
#include <cmath>
#include <iostream>

NeuralNetwork::NeuralNetwork(const std::vector<int>& topology, double learningRate) : learningRate(learningRate) {
    int numLayers = topology.size();
    for (int layerNum = 0; layerNum < numLayers; ++layerNum) {
        std::shared_ptr<ActivationFunction> sigmoid(new Sigmoid());
        layers.emplace_back(topology[layerNum], ((layerNum == numLayers - 1) ? 0 : topology[layerNum + 1]), sigmoid);
    }
}

void NeuralNetwork::feedForward(const std::vector<double>& inputValues) {
    // Assign the input values into the input neurons
    for (size_t i = 0; i < inputValues.size(); ++i) {
        layers[0][i].setOutputValue(inputValues[i]);
    }

    // Forward propagate
    for (size_t layerNum = 1; layerNum < layers.size(); ++layerNum) {
        Layer &prevLayer = layers[layerNum - 1];
        std::vector<double> prevLayerOutputs;
        for (size_t n = 0; n < prevLayer.size(); ++n){
            prevLayerOutputs.push_back(prevLayer[n].getOutputValue());
        }
        for (size_t n = 0; n < layers[layerNum].size(); ++n){
            layers[layerNum][n].feedForward(prevLayerOutputs);
        }
    }  
}

void NeuralNetwork::backPropagation(const std::vector<double>& targetValues) {
    // Calculate overall net error (RMS of output neuron errors)
    Layer &outputLayer = layers.back();

    error = 0.0;
    for (size_t n = 0; n < outputLayer.size() - 1; ++n) {
        double delta = targetValues[n] - outputLayer[n].getOutputValue();
        error += delta * delta;
    }
    error /= outputLayer.size() - 1; // get average error squared
    error = sqrt(error); // RMS

    // Calculate output layer gradients

    for (size_t n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].calcOutputGradients(targetValues[n]);
    }

    // Calculate gradients on hidden layers

    for (size_t layerNum = layers.size() - 2; layerNum > 0; --layerNum) {
        Layer &hiddenLayer = layers[layerNum];
        Layer &nextLayer = layers[layerNum + 1];

        for (size_t n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // For all layers from outputs to first hidden layer,
    // update connection weights

    for (size_t layerNum = layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = layers[layerNum];
        Layer &prevLayer = layers[layerNum - 1];

        for (size_t n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void NeuralNetwork::getResults(std::vector<double>& resultValues) const {
    resultValues.clear();
    for (size_t n = 0; n < layers.back().size() - 1; ++n) {
        resultValues.push_back(layers.back()[n].getOutputValue());
    }
}
