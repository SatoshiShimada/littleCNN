
#include "activation.h"

float logistic_apply(float input)
{
	return 1.0 / (1.0 + expf(-input));
}

float logistic_diff(float input)
{
	return logistic_apply(input) * (1.0 - logistic_apply(input));
}

