
#include <iostream>

#include "../../network/network.h"

int main(int argc, char *argv[])
{
	/* create Data */
	int ret;
	FILE *fp;
	int value;
	const int trainingDataNum = 60000;
	float *trainingData[trainingDataNum];
	float *labelData[trainingDataNum];
	fp = fopen("dataset/mnist/train-images.txt", "r");
	if(!fp) {
		std::cerr << "Error: couldn't open dataset file" << std::endl;
		return 0;
	}
	for(int i = 0; i < trainingDataNum; i++) {
		trainingData[i] = new float[784];
		for(int j = 0; j < 784; j++) {
			ret = fscanf(fp, " %d", &value);
			if(ret != 1) {
				std::cerr << "Error: couldn't load training dataset" << std::endl;
				return -1;
			}
			*(trainingData[i] + j) = value / 255.0;
		}
	}
	fclose(fp);
	fp = fopen("dataset/mnist/train-labels.txt", "r");
	if(!fp) {
		std::cerr << "Error: couldn't open dataset file" << std::endl;
		return 0;
	}
	int label;
	for(int i = 0; i < trainingDataNum; i++) {
		ret = fscanf(fp, " %d", &label);
		if(ret != 1) {
			std::cerr << "Error: couldn't load training dataset" << std::endl;
			return -1;
		}
		labelData[i] = new float[10];
		for(int j = 0; j < 10; j++) {
			*(labelData[i] + j) = ((j == label) ? 1.0 : 0.0);
		}
	}
	fclose(fp);
	const int testDataNum = 10000;
	float *testData[testDataNum];
	float *testLabelData[testDataNum];
	fp = fopen("dataset/mnist/test-images.txt", "r");
	if(!fp) return 0;
	for(int i = 0; i < testDataNum; i++) {
		testData[i] = new float[784];
		for(int j = 0; j < 784; j++) {
			ret = fscanf(fp, " %d", &value);
			if(ret != 1) {
				std::cerr << "Error: couldn't load training dataset" << std::endl;
				return -1;
			}
			*(testData[i] + j) = value / 255.0;
		}
	}
	fclose(fp);
	fp = fopen("dataset/mnist/test-labels.txt", "r");
	if(!fp) return 0;
	for(int i = 0; i < testDataNum; i++) {
		ret = fscanf(fp, " %d", &label);
		if(ret != 1) {
			std::cerr << "Error: couldn't load training dataset" << std::endl;
			return -1;
		}
		testLabelData[i] = new float[10];
		for(int j = 0; j < 10; j++) {
			*(testLabelData[i] + j) = ((j == label) ? 1.0 : 0.0);
		}
	}
	fclose(fp);
	std::cout << "Dataset loaded" << std::endl;

	/* parameters */
	int epoch = 50;
	float lr = 0.01;

	/* Create Network */
	Network *net;
	net = new Network();

	FullyConnectedLayer *full1, *full2;
	act_T *act1t, *act2t;

	act1t = new act_T;
	act1t->apply = logistic_apply;
	act1t->diff  = logistic_diff;
	act2t = new act_T;
	act2t->apply = logistic_apply;
	act2t->diff  = logistic_diff;
	full1 = new FullyConnectedLayer(784, 100, act1t, lr);
	full2 = new FullyConnectedLayer(100, 10, act2t, lr);

	net->appendLayer(full1);
	net->appendLayer(full2);

	net->setTest(testData, testLabelData, testDataNum);
	//net->loadParameters((char *)"parameters/mnist/200.param");
	net->train(trainingData, labelData, trainingDataNum, epoch);
	//net->saveParameters((char *)"parameters/mnist/250.param");
	//net->test(testData, testLabelData, testDataNum);

	delete net;
	delete act1t;
	delete act2t;
	delete full1;
	delete full2;

	return 0;
}
