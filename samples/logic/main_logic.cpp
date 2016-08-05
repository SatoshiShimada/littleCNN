
#include <iostream>

#include "../../network/network.h"
#include "../../network/util.h"

int main(int argc, char *argv[])
{
	/* create Data */
	int ret;
	FILE *fp;
	const int trainingDataNum = 4;
	float *trainingData[trainingDataNum];
	float *labelData[trainingDataNum];
	float d[4][2];
	float l[4][2];
	int value;
	
	bool re;
	trainingData[0] = d[0];
	trainingData[1] = d[1];
	trainingData[2] = d[2];
	trainingData[3] = d[3];
	re = loadTrainingData(trainingData, "dataset/logic/train-exor.txt", trainingDataNum, 2);
	if(re == false) return 0;

	labelData[0] = l[0];
	labelData[1] = l[1];
	labelData[2] = l[2];
	labelData[3] = l[3];
	re = loadTrainingLabel(labelData, "dataset/logic/train-exor-label.txt", trainingDataNum, 2);
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

	return 0;
}
