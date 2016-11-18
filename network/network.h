
#include "layer/layer.h"
#include "layer/fully_connected.h"
#include "layer/convolution.h"
#include "layer/max_pooling.h"
#include <vector>

class Network {
public:
	Network();
	void appendLayer(Layer *layer);
	void train(float **, float **, int, int);
	void test(float **, float **, int);
	void setTest(float **, float **, int);
	void setTest(float **, float **, int, int);
	void saveParameters(const char *);
	void loadParameters(const char *);
	void visualize(float **, int, int, int, int, int);
private:
	std::vector<Layer *> layers;
	bool testFlag;
	bool accuracyFlag;
	int testDataNum;
	float **testData;
	float **testDataLabel;
	int testInterval;
};

