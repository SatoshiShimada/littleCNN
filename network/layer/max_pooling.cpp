
#include <iostream>
#include "max_pooling.h"
#include "../util.h"

MaxPoolingLayer::MaxPoolingLayer(int inputWidth, int inputHeight, int inputChannels, int kernelWidth, int kernelHeight, int stride)
{
	this->inputWidth    = inputWidth;
	this->inputHeight   = inputHeight;
	this->inputChannels = inputChannels;
	this->kernelWidth   = kernelWidth;
	this->kernelHeight  = kernelHeight;
	this->stride        = stride;
	this->outputWidth   = (int)((inputWidth - 1) / stride) + 1;
	this->outputHeight  = (int)((inputHeight - 1) / stride) + 1;
	this->weightSize    = 0;
	this->biasSize      = 0;

	this->weight    = NULL;
	this->bias      = NULL;
	this->outputs   = new float[(outputWidth * outputHeight * inputChannels)];
	this->activated = new float[(outputWidth * outputHeight * inputChannels)];
	this->nextDelta = new float[(inputWidth * inputHeight * inputChannels)];
}

MaxPoolingLayer::MaxPoolingLayer(int inputWidth, int inputHeight, int inputChannels, int kernelWidth, int kernelHeight)
{
	this->inputWidth    = inputWidth;
	this->inputHeight   = inputHeight;
	this->inputChannels = inputChannels;
	this->kernelWidth   = kernelWidth;
	this->kernelHeight  = kernelHeight;
	this->stride        = 2;
	this->outputWidth   = (int)((inputWidth - 1) / stride) + 1;
	this->outputHeight  = (int)((inputHeight - 1) / stride) + 1;
	this->weightSize    = 0;
	this->biasSize      = 0;

	this->weight    = NULL;
	this->bias      = NULL;
	this->outputs   = new float[(outputWidth * outputHeight * inputChannels)];
	this->activated = new float[(outputWidth * outputHeight * inputChannels)];
	this->nextDelta = new float[(inputWidth * inputHeight * inputChannels)];
}

MaxPoolingLayer::~MaxPoolingLayer()
{
	delete outputs;
	delete activated;
	delete nextDelta;
}

float *MaxPoolingLayer::forward(float *inputs)
{
	for(int c = 0; c < inputChannels; c++) {
		for(int i = 0; i < outputHeight; i++) {
			for(int j = 0; j < outputWidth; j++) {
				float maxValue = inputs[c * (inputHeight * inputWidth) + (i * kernelHeight) * inputWidth + (j * kernelWidth)];
				for(int s = -(kernelHeight / 2); s <= (kernelHeight / 2); s++) {
					for(int t = -(kernelWidth / 2); t <= (kernelWidth / 2); t++) {
						float value;
						if(((i * stride + s) < 0) ||
						   ((i * stride + s) >= inputHeight) ||
						   ((j * stride + t) < 0) ||
						   ((j * stride + t) >= inputWidth)) {
							value = 0.0;
						} else {
							//value = inputs[c * (inputHeight * inputWidth) + (i * kernelHeight + s) * inputWidth + (j * kernelWidth + t)];
							value = inputs[c * (inputHeight * inputWidth) + (i * stride + s) * inputWidth + (j * stride + t)];
						}
						if(maxValue < value)
							maxValue = value;
					}
				}
				outputs[c * (outputHeight * outputWidth) + i * outputWidth + j] = maxValue;
			}
		}
	}
	this->apply(outputs, activated, inputChannels * outputHeight * outputWidth);
	return this->activated;
}

void MaxPoolingLayer::backward(float *inputs, float *delta)
{
	return;
}

float *MaxPoolingLayer::backward(float *inputs, float *delta, float *prevOut)
{
	for(int c = 0; c < inputChannels; c++) {
		for(int i = 0; i < outputHeight; i++) {
			for(int j = 0; j < outputWidth; j++) {
				for(int s = 0; s < kernelHeight; s++) {
					for(int t = 0; t < kernelWidth; t++) {
						if(inputs[c * (outputHeight * outputWidth) + (i * kernelHeight + s) * outputWidth + (j * kernelWidth + t)] == outputs[c * (inputHeight * inputWidth) + i * inputWidth + j])
							nextDelta[c * (outputHeight * outputWidth) + (i * kernelHeight + s) * outputWidth + (j * kernelWidth + t)] = delta[c * (outputHeight * outputWidth) + i * outputWidth + j];
					}
				}
			}
		}
	}
	return this->nextDelta;
}

float *MaxPoolingLayer::getOutput(void)
{
	return this->outputs;
}

void MaxPoolingLayer::apply(float *inputs, float *outputs, int num)
{
	//return this->activationFunc->apply(inputs, outputs, num);
	return;
}

void MaxPoolingLayer::diff(float *inputs, float *outputs, int num)
{
	//return this->activationFunc->diff(inputs, outputs, num);
	return;
}

float *MaxPoolingLayer::getWeight(void)
{
	return this->weight;
}

float *MaxPoolingLayer::getBias(void)
{
	return this->bias;
}

int MaxPoolingLayer::getWeightSize(void)
{
	return this->weightSize;
}

int MaxPoolingLayer::getBiasSize(void)
{
	return this->biasSize;
}

