
/*
 * Sample code for mnist dataset
 * Using convolutinal layer and max-pooling layer
 */

#include <iostream>

#include "../../network/network.h"
#include "../../network/util.h"

int main(int argc, char *argv[])
{
	bool re;

	/* Load training data and label */
	const int trainingDataNum = 60000;
	const int dataNum = 784; /* 28 x 28 = 784 pixel */
	const int outDim = 10; /* 10 types of digits */
	float *trainingData[trainingDataNum];
	float *labelData[trainingDataNum];

	for(int i = 0; i < trainingDataNum; i++)
		trainingData[i] = new float[dataNum];
	re = loadTrainingData(trainingData, "dataset/mnist/train-images.txt", trainingDataNum, dataNum, 255);
	if(re == false) return 0;

	for(int i = 0; i < trainingDataNum; i++)
		labelData[i] = new float[outDim];
	re = loadTrainingLabel(labelData, "dataset/mnist/train-labels.txt", trainingDataNum, outDim);
	if(re == false) return 0;

   	/* Load test data and label */
	const int testDataNum = 10000;
	float *testData[testDataNum];
	float *testLabelData[testDataNum];

	for(int i = 0; i < testDataNum; i++)
		testData[i] = new float[dataNum];
	re = loadTrainingData(testData, "dataset/mnist/test-images.txt", testDataNum, dataNum, 255);
	if(re == false) return 0;

	for(int i = 0; i < testDataNum; i++)
		testLabelData[i] = new float[outDim];
	re = loadTrainingLabel(testLabelData, "dataset/mnist/test-labels.txt", testDataNum, outDim);
	if(re == false) return 0;
	std::cout << "Dataset loaded" << std::endl;

	/* Parameters */
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
	full2 = new FullyConnectedLayer(50, outDim, act3, lr);

	net->appendLayer(conv1);
	net->appendLayer(pool1);
	net->appendLayer(conv2);
	net->appendLayer(pool2);
	net->appendLayer(full1);
	net->appendLayer(full2);

	net->setTest(testData, testLabelData, testDataNum, 1);
	//net->loadParameters("parameters/mnist/conv_6layer_10.param");
	net->train(trainingData, labelData, trainingDataNum, epoch);
	net->saveParameters("parameters/mnist/conv_6layer_20.param");
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

	for(int i = 0; i < trainingDataNum; i++)
		delete trainingData[i];
	for(int i = 0; i < trainingDataNum; i++)
		delete labelData[i];

	return 0;
}

