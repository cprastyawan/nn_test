#include "iris.h"

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include "neuralnetwork.h"
#include "activation_function.h"
#include "iris/iris_data.h"
#include "iris/iris_target.h"

#include "iris/layer_1.h"
#include "iris/layer_2.h"
#include "iris/layer_3.h"

#include "iris/bias_1.h"
#include "iris/bias_2.h"
#include "iris/bias_3.h"

static neuralnetwork_t nn_iris;
static iris_data_t iris[150];

extern uint64_t ns();

static void iris_getDesiredOutput(uint8_t label, floating_point* desiredOutput) {
	for (uint8_t i = 0; i < 3; i++) {
		desiredOutput[i] = label == i ? 1.0f : 0.0f;
	}

	return;
}

static uint8_t iris_correct(uint8_t label, floating_point* actv) {
	floating_point maxActv = *actv;
	uint8_t maxIdx = 0;

	for (uint8_t i = 1; i < 3; i++) {
		if (maxActv < actv[i]) {
			maxActv = actv[i];

			maxIdx = i;
		}
	}

	return label == maxIdx ? 1 : 0;
}

static void iris_data_init() {
	for (uint8_t i = 0; i < 150; i++) {
		iris[i].target = iris_target[i];

		for (uint8_t j = 0; j < 4; j++) {
			iris[i].data[j] = *(float*)(&iris_data[(i * 4) + j]);
		}
	}
}

void iris_init() {
	uint16_t numOfNeurons[] = { 4, 16, 16, 3 };
	activation_t activations[] = {relu, relu, softmax};

	neuralnetwork_init(&nn_iris, 4, numOfNeurons, activations);

	nn_iris.inputLayer->next->neurons.weights = (float*)layer_1;
	nn_iris.inputLayer->next->neurons.bias = (float*)bias_1;

	nn_iris.inputLayer->next->next->neurons.weights = (float*)layer_2;
	nn_iris.inputLayer->next->next->neurons.bias = (float*)bias_2;

	nn_iris.outputLayer->neurons.weights = (float*)layer_3;
	nn_iris.outputLayer->neurons.bias = (float*)bias_3;

	iris_data_init();

	return;
}

void iris_shuffle() {
	for (uint16_t i = 0; i < 150; i++) {
		uint16_t idx_array = rand() % 150;

		iris_data_t tmp = iris[i];
		iris[i] = iris[idx_array];
		iris[idx_array] = tmp;
	}

	return;
}

void iris_run() {
	uint16_t numCorrect = 0;

	for (uint16_t i = 0; i < 150; i++) {
		floating_point desiredOutput[4];
		neuralnetwork_input(&nn_iris, iris[i].data);

		uint64_t t = ns();

		neuralnetwork_feedforward(&nn_iris);
		
		t = ns() - t;

		iris_getDesiredOutput(iris[i].target, desiredOutput);

		uint8_t isCorrect = iris_correct(iris[i].target, desiredOutput);

		floating_point loss = neuralnetwork_calculateLoss(&nn_iris, desiredOutput);

		printf("I-%d\tlabel: %d\tLoss: %f\tCorrect: %s\tTime: %llu\r\n", i, iris[i].target, loss,
			isCorrect == 1 ? "Yes" : "No", t);

		numCorrect += isCorrect;
	}

	floating_point accuracy = (floating_point)numCorrect / 150.0f;

	printf("Total correct : %d, accuracy: %f\r\n", numCorrect, accuracy);

	return;
}