
#include <stdio.h>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <vector>

#include "network.h"

Network::Network()
{
	testFlag = false;
	accuracyFlag = false;
}

void Network::appendLayer(Layer *layer)
{
	this->layers.push_back(layer);
}

void Network::train(float **trainingData, float **labelData, int trainingDataCount, int epoch)
{
	float **z = new float *[layers.size()+1];
	//float *label = new float[layers[layers.size()-1]->outputNum];
	float *label;
	float **delta;
	float *output;
	int layerNum = layers.size();

	for(int ep = 0; ep < epoch; ep++) {
		for(int i = 0; i < trainingDataCount; i++) {
			/* feed forward */
			z[0] = trainingData[i];
			label = labelData[i];
			for(int n = 0; n < layerNum; n++) {
				z[n+1] = layers[n]->forward(z[n]);
			}
			/* calculate error for output layer */
			delta = new float *[layerNum];
			for(int j = 0; j < layerNum; j++) {
				delta[j] = new float[layers[layerNum-1]->outputNum];
			}
			output = layers[layerNum-1]->getOutput();
			for(int n = 0; n < layers[layerNum-1]->outputNum; n++) {
				delta[layerNum-1][n] = (z[layerNum][n] - label[n]) * layers[layerNum-1]->diff(output[n]);
			}
			/* back propagation */
			for(int n = 0; n < layerNum; n++) {
				if(layerNum-n-2 < 0)
					layers[layerNum-n-1]->backward(z[layerNum-n-1], delta[layerNum-n-1]);
				else
					delta[layerNum-n-2] = layers[layerNum-n-1]->backward(z[layerNum-n-1], delta[layerNum-n-1], layers[layerNum-n-2]->getOutput());
			}
		}
		/* test */
		if((ep + 1) % 1 == 0) {
			int acc = 0;
			if(accuracyFlag) {
				for(int i = 0; i < trainingDataCount; i++) {
					/* feed forward */
					z[0] = trainingData[i];
					for(int n = 0; n < layerNum; n++) {
						z[n+1] = layers[n]->forward(z[n]);
					}
					std::vector<float> result(z[layerNum], z[layerNum]+10);
					std::vector<float>::iterator maxIt = std::max_element(result.begin(), result.end());
					int maxIndex = std::distance(result.begin(), maxIt);
					std::vector<float> label(labelData[i], labelData[i] + 10);
					std::vector<float>::iterator maxItLabel = std::max_element(label.begin(), label.end());
					int labelIndex = std::distance(label.begin(), maxItLabel);
					if(maxIndex == labelIndex)
						acc++;
				}
				printf("accuracy [%d / %d]\n", acc, trainingDataCount);
				printf("\t%f\n", acc * 100.0 / trainingDataCount);
			}
			if(testFlag) {
				acc = 0;
				for(int i = 0; i < testDataNum; i++) {
					/* feed forward */
					z[0] = testData[i];
					for(int n = 0; n < layerNum; n++) {
						z[n+1] = layers[n]->forward(z[n]);
					}
					std::vector<float> result(z[layerNum], z[layerNum]+10);
					std::vector<float>::iterator maxIt = std::max_element(result.begin(), result.end());
					int maxIndex = std::distance(result.begin(), maxIt);
					std::vector<float> label(testDataLabel[i], testDataLabel[i] + 10);
					std::vector<float>::iterator maxItLabel = std::max_element(label.begin(), label.end());
					int labelIndex = std::distance(label.begin(), maxItLabel);
					if(maxIndex == labelIndex)
						acc++;
				}
				printf("test [%d / %d]\n", acc, testDataNum);
				printf("\t%f\n", acc * 100.0 / testDataNum);
			}
		}
	}
	//delete z;
	//delete label;
}

void Network::test(float **testData, float **testDataLabel, int testDataNum)
{
	float **z = new float *[layers.size()+1];
	int layerNum = layers.size();
	int acc = 0;

	for(int i = 0; i < testDataNum; i++) {
		/* feed forward */
		z[0] = testData[i];
		for(int n = 0; n < layerNum; n++) {
			z[n+1] = layers[n]->forward(z[n]);
		}
		std::vector<float> result(z[layerNum], z[layerNum]+10);
		std::vector<float>::iterator maxIt = std::max_element(result.begin(), result.end());
		int maxIndex = std::distance(result.begin(), maxIt);
		std::vector<float> label(testDataLabel[i], testDataLabel[i] + 10);
		std::vector<float>::iterator maxItLabel = std::max_element(label.begin(), label.end());
		int labelIndex = std::distance(label.begin(), maxItLabel);
		if(maxIndex == labelIndex)
			acc++;
	}
	printf("accuracy [%d / %d]\n", acc, testDataNum);
	printf("\t%f\n", acc * 100.0 / testDataNum);
}

void Network::setTest(float **testData, float **testDataLabel, int testDataNum)
{
	this->testFlag = true;
	this->testData = testData;
	this->testDataLabel = testDataLabel;
	this->testDataNum = testDataNum;
}

void Network::saveParameters(char *filename)
{
	int layerNum = layers.size();
	FILE *fp;

	fp = fopen(filename, "w");
	if(!fp) return;

	for(int i = 0; i < layerNum; i++) {
		int in = layers[i]->inputNum;
		int out = layers[i]->outputNum;
		float *w = layers[i]->getWeight();
		float *b = layers[i]->getBias();
		fprintf(fp, "%d,%d\n", in, out);
		/* save weights */
		int flag = 0;
		for(int cntOut = 0; cntOut < out; cntOut++) {
			for(int cntIn = 0; cntIn < in; cntIn++) {
				if(flag)
					fprintf(fp, ",");
				fprintf(fp, "%f", w[cntOut * in + cntIn]);
				flag = 1;
			}
			fprintf(fp, "\n");
			flag = 0;
		}
		/* save biases */
		for(int cntOut = 0; cntOut < out; cntOut++) {
			fprintf(fp, "%f", b[cntOut]);
			fprintf(fp, "\n");
		}
	}
	fclose(fp);
}

void Network::loadParameters(char *filename)
{
	int layerNum = layers.size();
	FILE *fp;
	int readIn, readOut;

	fp = fopen(filename, "r");
	if(!fp) return;

	for(int i = 0; i < layerNum; i++) {
		int in = layers[i]->inputNum;
		int out = layers[i]->outputNum;
		float *w = layers[i]->getWeight();
		float *b = layers[i]->getBias();
		fscanf(fp, "%d,%d\n", &readIn, &readOut);
		if(readIn != in || readOut != out)
			return;
		/* save weights */
		int flag = 0;
		char buf;
		for(int cntOut = 0; cntOut < out; cntOut++) {
			for(int cntIn = 0; cntIn < in; cntIn++) {
				if(flag) {
					fscanf(fp, "%c", &buf);
				}
				fscanf(fp, "%f", (w + cntOut * in + cntIn));
				flag = 1;
			}
			fscanf(fp, "%c", &buf);
			flag = 0;
		}
		/* save biases */
		for(int cntOut = 0; cntOut < out; cntOut++) {
			fscanf(fp, "%f", (b + cntOut));
			fscanf(fp, "%c", &buf);
		}
	}
	fclose(fp);
}

