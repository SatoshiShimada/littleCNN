
#ifndef __MAX_POOLING_H__
#define __MAX_POOLING_H__

#include "../activation/activation.h"
#include "layer.h"

class MaxPoolingLayer : public Layer {
private:
	int inputWidth;
	int inputHeight;
	int inputChannels;
	int kernelWidth;
	int kernelHeight;
	int outputWidth;
	int outputHeight;
	int stride;
	int weightSize;
	int biasSize;
	float *weight;
	float *bias;
	float *outputs;
	float *activated;
	float *nextDelta;
	act_T *activationFunc;
public:
	MaxPoolingLayer(int, int, int, int, int, int);
	MaxPoolingLayer(int, int, int, int, int);
	~MaxPoolingLayer();
	float *forward(float *);
	float *backward(float *, float *, float *);
	void   backward(float *, float *);
	float *getOutput(void);
	void   apply(float *, float *, int);
	void   diff(float *, float *, int);
	float *getWeight(void);
	float *getBias(void);
	int    getWeightSize(void);
	int    getBiasSize(void);
};

#endif // __MAX_POOLING_H__

