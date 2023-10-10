# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++17 -Wall -Wextra

# Source files
SRCS = activation.cpp layer.cpp main.cpp matrices.cpp neuron.cpp network.cpp utils/activation_factory.cpp utils/track_memory.cpp

# Executable name
EXEC = net

all: $(EXEC)

$(EXEC): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(EXEC)
