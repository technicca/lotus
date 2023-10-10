set different ranges for different weght
bias inits, set default ranges for init functions
test memory
check iterators in layer


Implement differen weight/bias init for xavier:


The original paper by Glorot and Bengio suggests a variance of 2/(n_in + n_out) where n_in is the number of inputs and n_out is the number of outputs of the neuron 365datascience.com.

Some sources, such as the deeplearning.ai notes, suggest a variance of 1/n_in deeplearning.ai.

Some other sources, such as machinelearningmastery.com, suggest a uniform distribution in the range -(sqrt(6)/sqrt(n_in + n_out)) to sqrt(6)/sqrt(n_in + n_out) machinelearningmastery.com.

print statements are triggered even if the ranges are defined for init methods