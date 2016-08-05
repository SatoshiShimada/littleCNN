
#ifndef __CONVOLUTION_H__
#define __CONVOLUTION_H__

#include "../activation/activation.h"
#include "layer.h"

class ConvolutionLayer : public Layer, public LearningLayer {
private:
	int inputWidth;
	int inputHeight;
	int inputChannels;
	int filterWidth;
	int filterHeight;
	int filterNum;
	int outputWidth;
	int outputHeight;
	int weightSize;
	int biasSize;
	float *weight;
	float *bias;
	float *outputs;
	float *activated;
	float *nextDelta;
	float *deltaWeight;
	float *deltaBias;
	float lr;
	act_T *activationFunc;
public:
	ConvolutionLayer(int, int, int, int, int, int, act_T *, float);
	~ConvolutionLayer();
	float *forward(float *);
	float *forward(float *, int);
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

#endif // __CONVOLUTION_H__

