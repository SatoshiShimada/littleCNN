
#include <cstdio>
#include <cstdlib>
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
	float *label;
	float **delta;
	float *output;
	int layerNum = layers.size();

	for(int ep = 0; ep < epoch; ep++) {
		for(int i = 0; i < trainingDataCount; i++) {
			/* caluate accuracy by test data */
			if(i % 10000 == 0) {
				std::cerr << "images: [" << i << " / " << trainingDataCount << "]" << std::endl;
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
					std::vector<float> testlabel(testDataLabel[i], testDataLabel[i] + 10);
					std::vector<float>::iterator maxItLabel = std::max_element(testlabel.begin(), testlabel.end());
					int labelIndex = std::distance(testlabel.begin(), maxItLabel);
					if(maxIndex == labelIndex)
						acc++;
				}
				printf("Test [Epoch: %d] [%d / %d]\n", ep, acc, testDataNum);
				printf("\t%f%%\n", acc * 100.0 / testDataNum);
			}
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
				delta[layerNum-1][n] = (z[layerNum][n] - label[n]) ;//* layers[layerNum-1]->diff(output[n]);
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
		if((ep + 1) % this->testInterval == 0) {
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
					std::vector<float> testlabel(labelData[i], labelData[i] + 10);
					std::vector<float>::iterator maxItLabel = std::max_element(testlabel.begin(), testlabel.end());
					int labelIndex = std::distance(testlabel.begin(), maxItLabel);
					if(maxIndex == labelIndex)
						acc++;
				}
				printf("Accuracy [Epoch: %d] [%d / %d]\n", ep, acc, trainingDataCount);
				printf("\t%f%%\n", acc * 100.0 / trainingDataCount);
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
					std::vector<float> testlabel(testDataLabel[i], testDataLabel[i] + 10);
					std::vector<float>::iterator maxItLabel = std::max_element(testlabel.begin(), testlabel.end());
					int labelIndex = std::distance(testlabel.begin(), maxItLabel);
					if(maxIndex == labelIndex)
						acc++;
				}
				printf("Test [Epoch: %d] [%d / %d]\n", ep, acc, testDataNum);
				printf("\t%f%%\n", acc * 100.0 / testDataNum);
			}
		}
	}
	delete z;
	for(int j = 0; j < layerNum; j++)
		delete delta[j];
	delete delta;
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
	printf("\t%f%%\n", acc * 100.0 / testDataNum);

	delete z;
}

void Network::setTest(float **testData, float **testDataLabel, int testDataNum, int interval)
{
	this->testFlag = true;
	this->testData = testData;
	this->testDataLabel = testDataLabel;
	this->testDataNum = testDataNum;
	this->testInterval = interval;
}

void Network::setTest(float **testData, float **testDataLabel, int testDataNum)
{
	this->testFlag = true;
	this->testData = testData;
	this->testDataLabel = testDataLabel;
	this->testDataNum = testDataNum;
	this->testInterval = 1;
}

void Network::saveParameters(char *filename)
{
	int layerNum = layers.size();
	FILE *fp;

	fp = fopen(filename, "w");
	if(!fp) return;

	for(int i = 0; i < layerNum; i++) {
		bool flag = false;
		int weightCnt = layers[i]->getWeightSize();
		int biasCnt   = layers[i]->getBiasSize();
		float *w = layers[i]->getWeight();
		float *b = layers[i]->getBias();

		if(weightCnt == 0 && biasCnt == 0)
			continue;
		/* save weights */
		for(int cnt = 0; cnt < weightCnt; cnt++) {
			if(flag)
				fprintf(fp, ",");
			fprintf(fp, "%f", w[cnt]);
			flag = true;
		}
		fprintf(fp, "\n");
		/* save biases */
		flag = false;
		for(int cnt = 0; cnt < biasCnt; cnt++) {
			if(flag)
				fprintf(fp, ",");
			fprintf(fp, "%f", b[cnt]);
			flag = true;
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
	std::cerr << "saved parameters at [" << filename << "]" << std::endl;
}

void Network::loadParameters(char *filename)
{
	int layerNum = layers.size();
	int ret;
	FILE *fp;

	fp = fopen(filename, "r");
	if(!fp) return;

	layerNum = 1;
	for(int i = 0; i < layerNum; i++) {
		bool flag = false;
		int weightCnt = layers[i]->getWeightSize();
		int biasCnt   = layers[i]->getBiasSize();
		float *w = layers[i]->getWeight();
		float *b = layers[i]->getBias();

		if(weightCnt == 0 && biasCnt == 0)
			continue;
		/* load weights */
		char buf;
		for(int cnt = 0; cnt < weightCnt; cnt++) {
			if(flag) {
				ret = fscanf(fp, "%c", &buf);
				if(ret != 1) {
					std::cerr << "file read error" << std::endl;
					return;
				}
			}
			ret = fscanf(fp, "%f", (w + cnt));
			if(ret != 1) {
				std::cerr << "file read error" << std::endl;
				return;
			}
			flag = true;
		}
		/* load biases */
		flag = false;
		for(int cnt = 0; cnt < biasCnt; cnt++) {
			if(flag) {
				ret = fscanf(fp, "%c", &buf);
				if(ret != 1) {
					std::cerr << "file read error" << std::endl;
					return;
				}
			}
			ret = fscanf(fp, "%f", (b + cnt));
			if(ret != 1) {
				std::cerr << "file read error" << std::endl;
				return;
			}
			flag = true;
		}
		ret = fscanf(fp, "%c", &buf); /* skip new line */
		if(ret != 1) {
			std::cerr << "file read error" << std::endl;
			return;
		}
	}
	fclose(fp);
}

void Network::visualize(float **testData, int layerIndex, int filterNum, int inputChannels, int imageHeight, int imageWidth)
{
	float **z = new float *[layers.size()+1];
	int layerNum = layers.size();
	char filename[100];
	FILE *fout;
	int num = 0; // data index

	/* feed forward */
	z[0] = testData[num];
	for(int n = 0; n < layerNum; n++) {
		z[n+1] = layers[n]->forward(z[n]);
	}

	float *data = z[layerIndex+1];
	float minValue = data[0];
	int count = filterNum * inputChannels * imageHeight * imageWidth;
	for(int i = 0; i < count; i++) {
		if(data[i] < minValue)
			minValue = data[i];
	}
	float maxValue = data[0] - minValue;
	for(int i = 0; i < count; i++) {
		data[i] -= minValue;
		if(data[i] > maxValue)
			maxValue = data[i];
	}
	float ratio = 255.0 / maxValue;
	for(int i = 0, k = 0; k < filterNum; k++) {
		for(int c = 0; c < inputChannels; c++) {
			sprintf(filename, "visual-%d-%d-%d.ppm", layerIndex, k, c);
			fout = fopen(filename, "w");
			if(!fout) {
				std::cerr << "Error: file open" << std::endl;
				return;
			}
			fprintf(fout, "P2\n# visualized image of hidden layer\n");
			fprintf(fout, "%d %d\n255\n", imageWidth, imageHeight);
			for(int s = 0; s < imageHeight; s++) {
				for(int t = 0; t < imageWidth; t++) {
					fprintf(fout, "%d ", (int)(ratio * data[i++]));
				}
				fprintf(fout, "\n");
			}
			fclose(fout);
		}
	}
	delete z;
}

