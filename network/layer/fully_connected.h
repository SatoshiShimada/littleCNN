
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
	void   apply(float *, float *, int);
	void   diff(float *, float *, int);
	int    getWeightSize(void);
	int    getBiasSize(void);
	void   saveParameters(const char *filename);
	void   loadParameters(const char *filename);
};

#endif // __FULLY_CONNECTED_H__

