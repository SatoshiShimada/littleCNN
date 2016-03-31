
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
	virtual float  apply(float) = 0;
	virtual float  diff(float) = 0;
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
	float  apply(float);
	float  diff(float);
	int    getWeightSize(void);
	int    getBiasSize(void);
};

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
	float  apply(float);
	float  diff(float);
	float *getWeight(void);
	float *getBias(void);
	int    getWeightSize(void);
	int    getBiasSize(void);
};

