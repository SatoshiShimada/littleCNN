
#include "layer.h"
#include "../util.h"

FullyConnectedLayer::FullyConnectedLayer(
	int inputNum, int outputNum, act_T *act, float learningRate)
{
	this->inputNum = inputNum;
	this->outputNum = outputNum;
	this->activationFunc = act;
	this->lr = learningRate;

	weight    = new float[inputNum * outputNum];
	bias      = new float[outputNum];
	outputs   = new float[outputNum];
	activated = new float[outputNum];
	nextDelta = new float[inputNum];

	randomRange(weight, inputNum * outputNum);
	for(int i = 0; i < outputNum; i++) {
		bias[i] = 0.0;
	}
	//randomRange(bias, outputNum);
}

FullyConnectedLayer::~FullyConnectedLayer()
{
	delete weight;
	delete bias;
	delete outputs;
	delete activated;
	delete nextDelta;
}

float *FullyConnectedLayer::forward(float *inputs)
{
	for(int i = 0; i < outputNum; i++) {
		float value = 0;
		for(int j = 0; j < inputNum; j++) {
			value += weight[i * inputNum + j] * inputs[j];
		}
		outputs[i] = value + bias[i];
		activated[i] = activationFunc->apply(outputs[i]);
	}
	return activated;
}

void FullyConnectedLayer::backward(float *inputs, float *delta)
{
	/* update parameters */
	for(int i = 0; i < outputNum; i++) {
		for(int j = 0; j < inputNum; j++) {
			weight[i * inputNum + j] -= this->lr * delta[i] * inputs[j];
		}
		bias[i] -= this->lr * delta[i];
	}
}

float *FullyConnectedLayer::backward(float *inputs, float *delta, float *prevOut)
{
	/* calculation delta for back propagation */
	for(int i = 0; i < inputNum; i++) {
		float value = 0;
		for(int j = 0; j < outputNum; j++) {
			value += weight[i * outputNum + j] * delta[j] * activationFunc->diff(prevOut[i]);
		}
		nextDelta[i] = value;
	}
	/* update parameters */
	for(int i = 0; i < outputNum; i++) {
		for(int j = 0; j < inputNum; j++) {
			weight[i * inputNum + j] -= this->lr * delta[i] * inputs[j];
		}
		bias[i] -= this->lr * delta[i];
	}
	return nextDelta;
}

float *FullyConnectedLayer::getWeight(void)
{
	return weight;
}

float *FullyConnectedLayer::getBias(void)
{
	return bias;
}

float *FullyConnectedLayer::getOutput(void)
{
	return outputs;
}

float FullyConnectedLayer::apply(float input)
{
	return this->activationFunc->apply(input);
}

float FullyConnectedLayer::diff(float input)
{
	return this->activationFunc->diff(input);
}

ConvolutionLayer::ConvolutionLayer(int inputWidth, int inputHeight, int inputChannels, int filterWidth, int filterHeight, int filterNum, act_T *, float)
{
	this->inputWidth = inputWidth;
	this->inputHeight = inputHeight;
	this->inputChannels = inputChannels;
	this->filterWidth = filterWidth;
	this->filterHeight = filterHeight;
	this->filterNum = filterNum;

	this->weight = new float[(filterNum*inputChannels*filterHeight*filterWidth)];
}

ConvolutionLayer::~ConvolutionLayer()
{
	delete this->weight;
}

float *ConvolutionLayer::forward(float *inputs)
{
	return this->activated;
}

void ConvolutionLayer::backward(float *inputs, float *delta)
{
}

float *ConvolutionLayer::backward(float *inputs, float *delta, float *prevOut)
{
	return this->nextDelta;
}

float *ConvolutionLayer::getWeight(void)
{
	return this->weight;
}

float *ConvolutionLayer::getBias(void)
{
	return this->bias;
}

float *ConvolutionLayer::getOutput(void)
{
	return this->outputs;
}

float ConvolutionLayer::apply(float input)
{
	return this->activationFunc->apply(input);
}

float ConvolutionLayer::diff(float input)
{
	return this->activationFunc->diff(input);
}

