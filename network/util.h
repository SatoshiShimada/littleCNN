
#ifndef __UTIL_H__
#define __UTIL_H__

bool randomRange(float *values, size_t size);

bool loadTrainingData(float **data, char *filename, int trainingDataNum, int countPerData);
bool loadTrainingLabel(float **data, char *filename, int labelDataNum, int countPerData);

#endif // __UTIL_H__
