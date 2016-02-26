
#include "layer/layer.h"
#include <vector>

class Network {
public:
	Network();
	void appendLayer(Layer *layer);
	void train(float **, float **, int, int);
	void test(float **, float **, int);
	void setTest(float **, float **, int);
	void saveParameters(char *);
	void loadParameters(char *);
private:
	std::vector<Layer *> layers;
	bool testFlag;
	bool accuracyFlag;
	int testDataNum;
	float **testData;
	float **testDataLabel;
};

