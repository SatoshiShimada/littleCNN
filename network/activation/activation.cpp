
#include "activation.h"

float logistic_apply(float input)
{
	return 1.0 / (1.0 + expf(-input));
}

float logistic_diff(float input)
{
	return logistic_apply(input) * (1.0 - logistic_apply(input));
}

float relu_apply(float input)
{
	if(input > 0.0)
		return input;
	else
		return 0.0;
}

float relu_diff(float input)
{
	if(input > 0.0)
		return 1.0;
	else
		return 0.0;
}

