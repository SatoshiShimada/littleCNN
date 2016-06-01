
#include <math.h>
#include "activation.h"

void logistic_apply(float *inputs, float *result, int num)
{
	for(int i = 0; i < num; i++) {
		result[i] = (1.0 / (1.0 + expf(-inputs[i])));
	}
	return;
}

void logistic_diff(float *inputs, float *result, int num)
{
	for(int i = 0; i < num; i++) {
		result[i] = (1.0 / (1.0 + expf(-inputs[i]))) * (1.0 - (1.0 / (1.0 + expf(-inputs[i]))));
	}
	return;
}

void relu_apply(float *inputs, float *result, int num)
{
	for(int i = 0; i < num; i++) {
		result[i] = (inputs[i] > 0.0) ? (inputs[i]) : (0.0);
	}
	return;
}

void relu_diff(float *inputs, float *result, int num)
{
	for(int i = 0; i < num; i++) {
		result[i] = (inputs[i] > 0.0) ? (1.0) : (0.0);
	}
	return;
}

void softmax_apply(float *inputs, float *result, int num)
{
	float buf = 0.0;
	for(int i = 0; i < num; i++) {
		buf += expf(inputs[i]);
	}
	if(buf = 0.0) return;
	for(int i = 0; i < num; i++) {
		result[i] = expf(inputs[i]) / buf;
	}
	return;
}

void softmax_diff(float *inputs, float *result, int num)
{
	// TODO
	return;
}

