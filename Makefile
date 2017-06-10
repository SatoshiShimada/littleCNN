
CXX=g++
CXXFLAGS=-std=c++11 -O3
LIBS=-lm

.PHONY: all
all: logic mnist conv
	

.PHONY: logic
logic: network.o fully_connected.o activation.o main_logic.o util.o
	$(CXX) $(CXXFLAGS) -o logic $^ $(LIBS)

.PHONY: mnist
mnist: network.o fully_connected.o activation.o main_mnist.o util.o
	$(CXX) $(CXXFLAGS) -o mnist $^ $(LIBS)

.PHONY: conv
conv: network.o fully_connected.o convolution.o max_pooling.o activation.o main_mnist_conv.o util.o
	$(CXX) $(CXXFLAGS) -o conv $^ $(LIBS)

.PHONY: check
check: network.o fully_connected.o convolution.o max_pooling.o activation.o check.o util.o
	$(CXX) $(CXXFLAGS) -o check $^ $(LIBS)

.PHONY: filter
filter: network.o fully_connected.o convolution.o max_pooling.o activation.o filter.o util.o
	$(CXX) $(CXXFLAGS) -o filter $^ $(LIBS)

filter.o: samples/mnist/filter.cpp
	$(CXX) $(CXXFLAGS) -c $^ $(LIBS)

check.o: samples/mnist/check.cpp
	$(CXX) $(CXXFLAGS) -c $^ $(LIBS)

main_logic.o: samples/logic/main_logic.cpp
	$(CXX) $(CXXFLAGS) -c $^ $(LIBS)

main_mnist.o: samples/mnist/main_mnist.cpp
	$(CXX) $(CXXFLAGS) -c $^ $(LIBS)

main_mnist_conv.o: samples/mnist/main_mnist_conv.cpp
	$(CXX) $(CXXFLAGS) -c $^ $(LIBS)

network.o: network/network.cpp
	$(CXX) $(CXXFLAGS) -c $^ $(LIBS)

fully_connected.o: network/layer/fully_connected.cpp
	$(CXX) $(CXXFLAGS) -c $^ $(LIBS)

convolution.o: network/layer/convolution.cpp
	$(CXX) $(CXXFLAGS) -c $^ $(LIBS)

max_pooling.o: network/layer/max_pooling.cpp
	$(CXX) $(CXXFLAGS) -c $^ $(LIBS)

activation.o: network/activation/activation.cpp
	$(CXX) $(CXXFLAGS) -c $^ $(LIBS)

util.o: network/util.cpp
	$(CXX) $(CXXFLAGS) -c $^ $(LIBS)

.PHONY: clean
clean:
	rm -f *.o
