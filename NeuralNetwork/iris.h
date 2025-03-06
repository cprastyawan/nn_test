#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define IRIS_TARGET_SETOSA		0
#define IRIS_TARGET_VERSICOLOR	1
#define IRIS_TARGET_VIRGINICA	2

typedef struct {
	uint8_t target;
	float data[4];
} iris_data_t;

void iris_init();
void iris_shuffle();
void iris_run();