
CXX=g++
CXXFLAGS=--std=c++11 -O3 -w
LIBS=-lm

.PHONY: all
all: logic mnist
	

.PHONY: logic
logic: samples/logic/main_logic.cpp network/network.cpp network/layer/layer.cpp network/activation/activation.cpp
	$(CXX) $(CXXFLAGS) -o logic $^ $(LIBS)

.PHONY: mnist
mnist: samples/mnist/main_mnist.cpp network/network.cpp network/layer/layer.cpp network/activation/activation.cpp
	$(CXX) $(CXXFLAGS) -o mnist $^ $(LIBS)

.PHONY: clean
clean:
	rm -f *.o
