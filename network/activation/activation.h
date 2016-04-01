
#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__

#include <math.h>

float logistic_apply(float);
float logistic_diff(float);
float relu_apply(float);
float relu_diff(float);

typedef struct {
	float (*apply)(float);
	float (*diff)(float);
} act_T;

#endif // __ACTIVATION_H__

