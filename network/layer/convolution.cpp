
#include <iostream>
#include "convolution.h"
#include "../util.h"

ConvolutionLayer::ConvolutionLayer(int inputWidth, int inputHeight, int inputChannels, int filterWidth, int filterHeight, int filterNum, act_T *act, float learningRate)
{
	this->inputWidth     = inputWidth;
	this->inputHeight    = inputHeight;
	this->inputChannels  = inputChannels;
	this->filterWidth    = filterWidth;
	this->filterHeight   = filterHeight;
	this->filterNum      = filterNum;
	this->outputWidth    = inputWidth - 2 * (int)(filterWidth / 2);
	this->outputHeight   = inputHeight - 2 * (int)(filterHeight / 2);
	this->activationFunc = act;
	this->lr             = learningRate;
	this->weightSize     = filterNum * inputChannels * filterHeight * filterWidth;
	this->biasSize       = filterNum;

	this->weight      = new float[weightSize];
	this->bias        = new float[biasSize];
	this->outputs     = new float[(filterNum * outputHeight * outputWidth)];
	this->activated   = new float[(filterNum * outputHeight * outputWidth)];
	this->nextDelta   = new float[(filterNum * inputHeight * inputWidth)];
	this->deltaWeight = new float[(filterNum * inputChannels * filterHeight * filterWidth)];
	this->deltaBias   = new float[filterNum];

	/* initialize the weights with random number */
	randomRange(this->weight, weightSize);
	/* initialize the biases with zero */
	for(int i = 0; i < biasSize; i++) {
		this->bias[i] = 0.0;
	}
}

ConvolutionLayer::~ConvolutionLayer()
{
	delete this->weight;
	delete this->bias;
	delete this->outputs;
	delete this->activated;
	delete this->nextDelta;
	delete this->deltaWeight;
	delete this->deltaBias;
}

float *ConvolutionLayer::forward(float *inputs, int padding_size)
{
	float *ret;
	float *padding_input = new float[inputChannels * inputHeight * inputWidth];
	int i;
	int padHeight = inputHeight - padding_size * 2;
	int padWidth = inputWidth - padding_size * 2;
	for(int c = 0; c < inputChannels; c++) {
		for(i = 0; i < padding_size; i++) {
			for(int j = 0; j < inputWidth; j++) {
				padding_input[c * (inputHeight * inputWidth) + i * (inputWidth) + j] = 0.0;
			}
		}
		for(; i < padHeight + padding_size; i++) {
			for(int n = 0; n < padding_size; n++) {
				padding_input[c * (inputHeight * inputWidth) + i * inputWidth + n] = 0.0;
			}
			for(int j = 0; j < padWidth; j++) {
				padding_input[c * (inputHeight * inputWidth) + i * inputWidth + (j + padding_size)] = inputs[c * (padHeight * padWidth) + (i - padding_size) * padWidth + j];
			}
			for(int n = padding_size + padWidth; n < inputWidth; n++) {
				padding_input[c * (inputHeight * inputWidth) + i * inputWidth + n] = 0.0;
			}
		}
		for(; i < inputHeight; i++) {
			for(int j = 0; j < inputWidth; j++) {
				padding_input[c * (inputHeight * inputWidth) + i * (inputWidth) + j] = 0.0;
			}
		}
	}

	ret = this->forward(padding_input);
	delete padding_input;
	return ret;
}

float *ConvolutionLayer::forward(float *inputs)
{
	for(int k = 0; k < filterNum; k++) {
		for(int i = 0; i < outputHeight; i++) {
			for(int j = 0; j < outputWidth; j++) {
				outputs[k * (outputHeight * outputWidth) + i * outputWidth + j] = 0.0;
				for(int c = 0; c < inputChannels; c++) {
					for(int s = 0; s < filterHeight; s++) {
						for(int t = 0; t < filterWidth; t++) {
							outputs[k * (outputHeight * outputWidth) + i * outputWidth + j] +=
								weight[k * (inputChannels * filterHeight * filterWidth) + c * (filterHeight * filterWidth) + s * filterWidth + t] *
								inputs[c * (inputHeight * inputWidth) + (i+s) * inputWidth + (j+t)];
						}
					}
				}
				outputs[k * (outputHeight * outputWidth) + i * outputWidth + j] += bias[k];
			}
		}
	}
	this->apply(output, activated, filterNum * outputHeight * outputWidth);

	return this->activated;
}

void ConvolutionLayer::backward(float *inputs, float *delta)
{
	/* calculate gradient */
	for(int i = 0; i < weightSize; i++) {
		deltaWeight[i] = 0.0;
	}
	for(int k = 0; k < filterNum; k++) {
		deltaBias[k] = 0.0;
		for(int i = 0; i < outputHeight; i++) {
			for(int j = 0; j < outputWidth; j++) {
				int index = k * (outputHeight * outputWidth) + i * outputWidth + j;
				float d = delta[index] * this->diff(outputs[index]);
				deltaBias[k] += d;
				for(int c = 0; c < inputChannels; c++) {
					for(int s = 0; s < filterHeight; s++) {
						for(int t = 0; t < filterWidth; t++) {
							int index1 =
								k * (inputChannels * filterHeight * filterWidth) + c * (filterHeight * filterWidth) + s * filterWidth + t;
							int index2 =
								c * (inputHeight * inputWidth) + (i+s) * inputWidth + (j+t);
							deltaWeight[index1] += d * inputs[index2];
						}
					}
				}
			}
		}
	}

	/* update parameters */
	for(int k = 0; k < filterNum; k++) {
		bias[k] -= lr * deltaBias[k];
		for(int c = 0; c < inputChannels; c++) {
			for(int s = 0; s < filterHeight; s++) {
				for(int t = 0; t < filterWidth; t++) {
					int index = k * (inputChannels * filterHeight * filterWidth) + c * (filterHeight * filterWidth) + s * filterWidth + t;
					weight[index] -= lr * deltaWeight[index];
				}
			}
		}
	}
}

float *ConvolutionLayer::backward(float *inputs, float *delta, float *prevOut)
{
	/* calculate delta for previous layer */
	for(int c = 0; c < inputChannels; c++) {
		for(int i = 0; i < inputHeight; i++) {
			for(int j = 0; j < inputWidth; j++) {
				int indexDelta = c * (inputHeight * inputWidth) + i * inputWidth + j;
				nextDelta[indexDelta] = 0.0;
				for(int k = 0; k < filterNum; k++) {
					for(int s = 0; s < filterHeight; s++) {
						for(int t = 0; t < filterWidth; t++) {
							int index1 = i - filterHeight - s;
							int index2 = j - filterWidth  - t;
							if(index1 < 0 || index2 < 0)
								continue;
							int index = k * (outputHeight * outputWidth) + index1 * outputWidth + index2;
							nextDelta[indexDelta] +=
								delta[index] * this->diff(outputs[index]) *
								weight[k * (inputChannels * filterHeight * filterWidth) + c * (filterHeight * filterWidth) + s * filterWidth + t];
						}
					}
				}
			}
		}
	}

	/* calculate gradient */
	for(int i = 0; i < weightSize; i++) {
		deltaWeight[i] = 0.0;
	}
	for(int k = 0; k < filterNum; k++) {
		deltaBias[k] = 0.0;
		for(int i = 0; i < outputHeight; i++) {
			for(int j = 0; j < outputWidth; j++) {
				int index = k * (outputHeight * outputWidth) + i * outputWidth + j;
				float d = delta[index] * this->diff(outputs[index]);
				deltaBias[k] += d;
				for(int c = 0; c < inputChannels; c++) {
					for(int s = 0; s < filterHeight; s++) {
						for(int t = 0; t < filterWidth; t++) {
							int index1 =
								k * (inputChannels * filterHeight * filterWidth) + c * (filterHeight * filterWidth) + s * filterWidth + t;
							int index2 =
								c * (inputHeight * inputWidth) + (i+s) * inputHeight + (j+t);
							deltaWeight[index1] += d * inputs[index2];
						}
					}
				}
			}
		}
	}

	/* update parameters */
	for(int k = 0; k < filterNum; k++) {
		bias[k] -= lr * deltaBias[k];
		for(int c = 0; c < inputChannels; c++) {
			for(int s = 0; s < filterHeight; s++) {
				for(int t = 0; t < filterWidth; t++) {
					int index = k * (inputChannels * filterHeight * filterWidth) + c * (filterHeight * filterWidth) + s * filterWidth + t;
					weight[index] -= lr * deltaWeight[index];
				}
			}
		}
	}

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

int ConvolutionLayer::getWeightSize(void)
{
	return this->weightSize;
}

int ConvolutionLayer::getBiasSize(void)
{
	return this->biasSize;
}

void ConvolutionLayer::saveParameters(const char *filename)
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

void ConvolutionLayer::loadParameters(const char *filename)
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

