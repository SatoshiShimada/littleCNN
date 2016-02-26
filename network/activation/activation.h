
#include <math.h>

float logistic_apply(float);
float logistic_diff(float);

typedef struct {
	float (*apply)(float);
	float (*diff)(float);
} act_T;

