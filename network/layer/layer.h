
#include "../activation/activation.h"

class Layer {
public:
	int inputNum;
	int outputNum;
	virtual float *forward(float *) = 0;
	virtual float *backward(float *, float *, float *) = 0;
	virtual void   backward(float *, float *) = 0;
	virtual float *getOutput(void) = 0;
	virtual float apply(float) = 0;
	virtual float diff(float) = 0;
	virtual float *getWeight(void) = 0;
	virtual float *getBias(void) = 0;
	act_T *activationFunc;
};

class LearningLayer {
private:
	float *weight;
	float *bias;
public:
	virtual float *getWeight(void) = 0;
	virtual float *getBias(void) = 0;
};

class FullyConnectedLayer : public Layer, public LearningLayer {
private:
	float *weight;
	float *bias;
	float *outputs;
	float *activated;
	float *nextDelta;
	float lr;
public:
	FullyConnectedLayer(int, int, act_T *, float);
	~FullyConnectedLayer();
	float *forward(float *);
	float *backward(float *, float *, float *);
	void   backward(float *, float *);
	float *getWeight(void);
	float *getBias(void);
	float *getOutput(void);
	float apply(float);
	float diff(float);
	act_T *activationFunc;
};

class ConvolutionLayer : public Layer, public LearningLayer {
private:
	float *weight;
	float *bias;
	float *outputs;
	float *activated;
	float *nextDelta;
	float lr;
	int inputWidth;
	int inputHeight;
	int inputChannels;
	int filterWidth;
	int filterHeight;
	int filterNum;
public:
	ConvolutionLayer(int, int, int, int, int, int, act_T *, float);
	~ConvolutionLayer();
	float *forward(float *);
	float *backward(float *, float *, float *);
	void   backward(float *, float *);
	float *getWeight(void);
	float *getBias(void);
	float *getOutput(void);
	float apply(float);
	float diff(float);
	act_T *activationFunc;
};

