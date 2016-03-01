
#include <iostream>
#include <cstdio>
#include <cstdlib>

int main(int argc, char *argv[])
{
	int filterNum, inputNum, filterHeight, filterWidth;
	int ret;
	int idx = 0;
	float value, maxValue;
	float *weights;
	FILE *fin, *fout;
	char buf;

	if(argc != 6) {
		std::cerr << "Usage:" << std::endl;
		std::cerr << argv[0] << " [filename] [filterNum] [inputChannels] [filterHeight] [filterWidth]" << std::endl;
		return 0;
	}

	filterNum = atoi(argv[2]);
	inputNum  = atoi(argv[3]);
	filterHeight = atoi(argv[4]);
	filterWidth  = atoi(argv[5]);

	fin = fopen(argv[1], "r");
	if(!fin) {
		std::cerr << "Error: file open" << std::endl;
		return -1;
	}

	maxValue = 0.0;
	weights = new float[(filterNum * inputNum * filterHeight * filterWidth)];
	for(int i = 0; i < filterNum; i++) {
		for(int j = 0; j < inputNum; j++) {
			for(int k = 0; k < filterHeight; k++) {
				for(int l = 0; l < filterWidth; l++) {
					ret = fscanf(fin, " %f", &value);
					if(ret != 1) {
						std::cerr << "Error: couldn't read file" << std::endl;
						return -1;
					}
					ret = fscanf(fin, "%c", &buf);
					weights[idx++] = value;
					if(abs(value) > maxValue)
						maxValue = abs(value);
				}
			}
		}
	}
	fclose(fin);
	idx = 0;
	float ratio = 255.0 / maxValue;
	char *filename = new char[100];
	for(int i = 0; i < filterNum; i++) {
		for(int j = 0; j < inputNum; j++) {
			sprintf(filename, "filter-%d-%d.ppm", i, j);
			fout = fopen(filename, "w");
			if(!fout) {
				std::cerr << "Error: file open" << std::endl;
				return -1;
			}
			fprintf(fout, "P3\n# Filter of Convolutional Neural Network\n");
			fprintf(fout, "%d %d\n255\n", filterWidth, filterHeight);
			for(int k = 0; k < filterHeight; k++) {
				for(int l = 0; l < filterWidth; l++) {
					value = weights[idx++];
					if(value > 0.0) {
						fprintf(fout, "%d 0 0 ", (int)(value * ratio));
					} else {
						fprintf(fout, "0 0 %d ", abs((int)(value * ratio)));
					}
				}
				fprintf(fout, "\n");
			}
			fclose(fout);
		}
	}
	delete filename;
	delete weights;

	return 0;
}

