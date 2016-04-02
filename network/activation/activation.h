
#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__

void logistic_apply(float *, float *, int);
void logistic_diff(float *, float *, int);
void relu_apply(float *, float *, int);
void relu_diff(float *, float *, int);
void softmax_apply(float *, float *, int);
void softmax_diff(float *, float *, int);

typedef struct {
	void (*apply)(float *, float *, int);
	void (*diff)(float *, float *, int);
} act_T;

#endif // __ACTIVATION_H__

