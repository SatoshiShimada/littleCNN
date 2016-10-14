
#ifndef __UTIL_H__
#define __UTIL_H__

bool randomRange(float *values, size_t size);

bool loadTrainingData(float **data, const char *filename, int trainingDataNum, int countPerData, int ratio=1);
bool loadTrainingLabel(float **data, const char *filename, int labelDataNum, int countPerData);

#endif // __UTIL_H__
