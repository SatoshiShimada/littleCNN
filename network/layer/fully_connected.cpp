
#include <iostream>
#include "fully_connected.h"
#include "../util.h"

FullyConnectedLayer::FullyConnectedLayer(
	int inputNum, int outputNum, act_T *act, float learningRate)
{
	this->inputNum       = inputNum;
	this->outputNum      = outputNum;
	this->activationFunc = act;
	this->lr             = learningRate;
	this->weightSize     = inputNum * outputNum;
	this->biasSize       = outputNum;

	weight    = new float[weightSize];
	bias      = new float[biasSize];
	outputs   = new float[outputNum];
	activated = new float[outputNum];
	nextDelta = new float[inputNum];

	/* initialize the weights with random number */
	randomRange(this->weight, weightSize);
	/* initialize the biases with zero */
	for(int i = 0; i < biasSize; i++) {
		this->bias[i] = 0.0;
	}
	/* initialize the biases with random number */
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
	}
	this->apply(outputs, activated, outputNum);
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
			value += weight[i * outputNum + j] * delta[j];// * activationFunc->diff(prevOut[i]);
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

void FullyConnectedLayer::apply(float *inputs, float *outputs, int num)
{
	this->activationFunc->apply(inputs, outputs, num);
	return;
}

void FullyConnectedLayer::diff(float *inputs, float *outputs, int num)
{
	this->activationFunc->diff(inputs, outputs, num);
	return;
}

int FullyConnectedLayer::getWeightSize(void)
{
	return this->weightSize;
}

int FullyConnectedLayer::getBiasSize(void)
{
	return this->biasSize;
}

void FullyConnectedLayer::saveParameters(const char *filename)
{
	FILE *fp;
	fp = fopen(filename, "wb");
	if(!fp) {
		return;
	}
	fwrite(this->weight, sizeof(float), this->weightSize, fp);
	fwrite(this->bias, sizeof(float), this->biasSize, fp);
	fclose(fp);
}

void FullyConnectedLayer::loadParameters(const char *filename)
{
	FILE *fp;
	fp = fopen(filename, "rb");
	if(!fp) {
		return;
	}
	fread(this->weight, sizeof(float), this->weightSize, fp);
	fread(this->bias, sizeof(float), this->biasSize, fp);
	fclose(fp);
}

