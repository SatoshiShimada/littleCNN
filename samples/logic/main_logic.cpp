
#include <iostream>

#include "../../network/network.h"
#include "../../network/util.h"

int main(int argc, char *argv[])
{
	/* create Data */
	bool re;
	const int trainingDataNum = 4;
	const int dataDim = 2;
	float *trainingData[trainingDataNum];
	float *labelData[trainingDataNum];
	
	for(int i = 0; i < trainingDataNum; i++) {
		trainingData[i] = new float(dataDim);
	}
	re = loadTrainingData(trainingData, "dataset/logic/train-exor.txt", trainingDataNum, dataDim);
	if(re == false) return 0;

	for(int i = 0; i < trainingDataNum; i++) {
		labelData[i] = new float(dataDim);
	}
	re = loadTrainingLabel(labelData, "dataset/logic/train-exor-label.txt", trainingDataNum, dataDim);
	if(re == false) return 0;

	for(int i = 0; i < 4; i++) {
		std::cout << trainingData[i][0] << trainingData[i][1] << " | ";
		std::cout << labelData[i][0] << labelData[i][1] << std::endl;
	}

	/* parameters */
	int epoch = 20000;
	float lr = 0.1;

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
	full1 = new FullyConnectedLayer(2, 3, act1t, lr);
	full2 = new FullyConnectedLayer(3, 2, act2t, lr);

	net->appendLayer(full1);
	net->appendLayer(full2);

	net->setTest(trainingData, labelData, trainingDataNum, 1000);
	net->train(trainingData, labelData, trainingDataNum, epoch);
	net->saveParameters((char *)"parameters/logic/exor_20000.param");

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
