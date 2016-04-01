
#ifndef __FULLY_CONNECTED_H__
#define __FULLY_CONNECTED_H__

#include "../activation/activation.h"
#include "layer.h"

class FullyConnectedLayer : public Layer, public LearningLayer {
private:
	int weightSize;
	int biasSize;
	float *weight;
	float *bias;
	float *outputs;
	float *activated;
	float *nextDelta;
	float lr;
	act_T *activationFunc;
public:
	FullyConnectedLayer(int, int, act_T *, float);
	~FullyConnectedLayer();
	float *forward(float *);
	float *backward(float *, float *, float *);
	void   backward(float *, float *);
	float *getWeight(void);
	float *getBias(void);
	float *getOutput(void);
	float  apply(float);
	float  diff(float);
	int    getWeightSize(void);
	int    getBiasSize(void);
};

#endif // __FULLY_CONNECTED_H__

