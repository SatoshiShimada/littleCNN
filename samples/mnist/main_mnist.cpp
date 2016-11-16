
#include <iostream>

#include "../../network/network.h"
#include "../../network/util.h"

int main(int argc, char *argv[])
{
	bool re;

	/* Load training data */
	const int trainingDataNum = 60000;
	const int dataDim = 784; /* 28 x 28 = 784 pixel */
	const int outDim = 10; /* 10 type of digits */
	float *trainingData[trainingDataNum];
	float *labelData[trainingDataNum];

	for(int i = 0; i < trainingDataNum; i++)
		trainingData[i] = new float[dataDim];
	re = loadTrainingData(trainingData, "dataset/mnist/train-images.txt", trainingDataNum, dataDim, 255);
	if(re == false) return 0;

	for(int i = 0; i < trainingDataNum; i++)
		labelData[i] = new float[outDim];
	re = loadTrainingLabel(labelData, "dataset/mnist/train-labels.txt", trainingDataNum, outDim);
	if(re == false) return 0;

	/* Load test data */
	const int testDataNum = 10000;
	float *testData[testDataNum];
	float *testLabelData[testDataNum];

	for(int i = 0; i < testDataNum; i++)
		testData[i] = new float[dataDim];
	re = loadTrainingData(testData, "dataset/mnist/test-images.txt", testDataNum, dataDim, 255);
	if(re == false) return 0;

	for(int i = 0; i < testDataNum; i++)
		testLabelData[i] = new float[outDim];
	re = loadTrainingLabel(testLabelData, "dataset/mnist/test-labels.txt", testDataNum, outDim);
	if(re == false) return 0;
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
	full1 = new FullyConnectedLayer(dataDim, 100, act1t, lr);
	full2 = new FullyConnectedLayer(100, outDim, act2t, lr);

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

	for(int i = 0; i < trainingDataNum; i++)
		delete trainingData[i];
	for(int i = 0; i < trainingDataNum; i++)
		delete labelData[i];

	return 0;
}
