
#include <cstdio>
#include <iostream>
#include <random>
#include "util.h"

bool randomRange(float *values, size_t size)
{
	std::random_device rand;
	std::mt19937 mt(rand());
	std::uniform_real_distribution<> randReal(0.0, 2.0);
	for(size_t i = 0; i < size; i++) {
		values[i] = (float)randReal(mt) - 1.0;
	}
	return true;
}

bool loadTrainingData(float **data, char *filename, int trainingDataNum, int countPerData)
{
	int value;
	int ret;
	FILE *fp;
	
	fp = fopen(filename, "r");
	if(!fp) {
		std::cerr << "Error: couldn't open dataset file" << std::endl;
		std::cerr << filename << std::endl;
		return false;
	}
	for(int i = 0; i < trainingDataNum; i++) {
		for(int j = 0; j < countPerData; j++) {
			ret = fscanf(fp, " %d", &value);
			if(ret != 1) {
				std::cerr << "Error: couldn't load training dataset" << std::endl;
				std::cerr << filename << std::endl;
				return false;
			}
			data[i][j] = (float)value;
		}
	}
	fclose(fp);
	return true;
}

bool loadTrainingLabel(float **data, char *filename, int labelDataNum, int countPerData)
{
	int value;
	int ret;
	FILE *fp;
	
	fp = fopen(filename, "r");
	if(!fp) {
		std::cerr << "Error: couldn't open dataset file" << std::endl;
		std::cerr << filename << std::endl;
		return false;
	}
	for(int i = 0; i < labelDataNum; i++) {
		ret = fscanf(fp, " %d", &value);
		if(ret != 1) {
			std::cerr << "Error: couldn't load training dataset" << std::endl;
			std::cerr << filename << std::endl;
			return false;
		}
		for(int j = 0; j < countPerData; j++) {
			data[i][j] = ((value == j) ? 1.0 : 0.0);
		}
	}
	fclose(fp);
	return true;
}
