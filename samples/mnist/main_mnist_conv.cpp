
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
	/* load training data */
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
	/* load test data */
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
	int epoch = 10;
	float lr = 0.02;

	/* Create Network */
	Network *net;
	net = new Network();

	ConvolutionLayer *conv1, *conv2;
	MaxPoolingLayer *pool1, *pool2;
	FullyConnectedLayer *full1, *full2;
	act_T *act1, *act2, *act3, *act4;

	act1 = new act_T;
	act1->apply = logistic_apply;
	act1->diff  = logistic_diff;
	act2 = new act_T;
	act2->apply = logistic_apply;
	act2->diff  = logistic_diff;
	act3 = new act_T;
	act3->apply = logistic_apply;
	act3->diff  = logistic_diff;
	act4 = new act_T;
	act4->apply = logistic_apply;
	act4->diff  = logistic_diff;
	conv1 = new ConvolutionLayer(28, 28, 1, 4, 4, 6, act1, lr);
	pool1 = new MaxPoolingLayer(24, 24, 6, 4, 4);
	conv2 = new ConvolutionLayer(12, 12, 6, 4, 4, 14, act2, lr);
	pool2 = new MaxPoolingLayer(8, 8, 14, 4, 4);
	full1 = new FullyConnectedLayer(224, 50, act2, lr);
	full2 = new FullyConnectedLayer(50, 10, act3, lr);

	net->appendLayer(conv1);
	net->appendLayer(pool1);
	net->appendLayer(conv2);
	net->appendLayer(pool2);
	net->appendLayer(full1);
	net->appendLayer(full2);

	net->setTest(testData, testLabelData, testDataNum, 1);
	//net->loadParameters((char *)"parameters/mnist/conv_6layer_10.param");
	net->train(trainingData, labelData, trainingDataNum, epoch);
	net->saveParameters((char *)"parameters/mnist/conv_6layer_20.param");
	//net->test(testData, testLabelData, testDataNum);

	delete net;
	delete act1;
	delete act2;
	delete act3;
	delete act4;
	delete conv1;
	delete conv2;
	delete pool1;
	delete pool2;
	delete full1;
	delete full2;

	return 0;
}

