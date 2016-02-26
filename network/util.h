
#include <random>

bool randomRange(float *values, size_t size)
{
	std::random_device rand;
	std::mt19937 mt(rand());
	std::uniform_real_distribution<> randReal(0.0, 2.0);
	for(size_t i = 0; i < size; i++) {
		values[i] = (float)randReal(mt) - 1.0;
	}
	return true;
}

