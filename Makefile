
CXX=g++
CXXFLAGS=--std=c++11 -O3
LIBS=-lm

.PHONY: all
all: logic mnist conv
	

.PHONY: logic
logic: network.o layer.o activation.o main_logic.o
	$(CXX) $(CXXFLAGS) -o logic $^ $(LIBS)

.PHONY: mnist
mnist: network.o layer.o activation.o main_mnist.o
	$(CXX) $(CXXFLAGS) -o mnist $^ $(LIBS)

.PHONY: conv
conv: network.o layer.o activation.o main_mnist_conv.o
	$(CXX) $(CXXFLAGS) -o conv $^ $(LIBS)

.PHONY: check
check: network.o layer.o activation.o check.o
	$(CXX) $(CXXFLAGS) -o check $^ $(LIBS)

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

layer.o: network/layer/layer.cpp
	$(CXX) $(CXXFLAGS) -c $^ $(LIBS)

activation.o: network/activation/activation.cpp
	$(CXX) $(CXXFLAGS) -c $^ $(LIBS)

.PHONY: clean
clean:
	rm -f *.o
