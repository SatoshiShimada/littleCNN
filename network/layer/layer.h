
#ifndef __LAYER_H__
#define __LAYER_H__

#include "../activation/activation.h"

class Layer {
private:
	int weightSize;
	int biasSize;
	act_T *activationFunc;
public:
	int inputNum;
	int outputNum;
	virtual float *forward(float *) = 0;
	virtual float *backward(float *, float *, float *) = 0;
	virtual void   backward(float *, float *) = 0;
	virtual float *getOutput(void) = 0;
	virtual void   apply(float *, float *, int) = 0;
	virtual void   diff(float *, float *, int) = 0;
	virtual float *getWeight(void) = 0;
	virtual float *getBias(void) = 0;
	virtual int    getWeightSize(void) = 0;
	virtual int    getBiasSize(void) = 0;
};

class LearningLayer {
private:
	int weightSize;
	int biasSize;
	float *weight;
	float *bias;
public:
	virtual float *getWeight(void) = 0;
	virtual float *getBias(void) = 0;
};

#endif // __LAYER_H__

