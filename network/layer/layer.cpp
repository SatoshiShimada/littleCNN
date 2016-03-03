
#include <iostream>
#include "layer.h"
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

int FullyConnectedLayer::getWeightSize(void)
{
	return this->weightSize;
}

int FullyConnectedLayer::getBiasSize(void)
{
	return this->biasSize;
}

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
	for(int k = 0; k < filterNum; k++) {
		for(int i = 0; i < outputHeight; i++) {
			for(int j = 0; j < outputWidth; j++) {
				int index = k * (outputHeight * outputWidth) + i * outputWidth + j;
				activated[index] = this->apply(outputs[index]);
			}
		}
	}

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
				activated[c * (outputHeight * outputWidth) + i * outputWidth + j] = this->apply(maxValue);
			}
		}
	}
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

float MaxPoolingLayer::apply(float input)
{
	//return this->activationFunc->apply(input);
	return input;
}

float MaxPoolingLayer::diff(float input)
{
	//return this->activationFunc->diff(input);
	return 1.0;
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

