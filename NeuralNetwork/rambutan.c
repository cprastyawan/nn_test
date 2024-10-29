#include "neuralnetwork.h"
#include "activation_function.h"

#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

typedef struct {
	uint8_t label;
	floating_point input[2];
} xor_data_t;

static uint8_t xor_datainit[] = {0, 1, 2, 3};
static xor_data_t xor_dataset[sizeof(xor_datainit)];

static neuralnetwork_t rambutan;

void xor_desiredoutput(uint8_t input, floating_point* output) {
	switch (input) {
	case 0:
		*output = 0.0;
		break;
	case 1:
		*output = 1.0;
		break;
	case 2:
		*output = 1.0;
		break;
	case 3:
		*output = 0.0;
		break;
	}

	return;
}

void xordataset_init(uint8_t input, xor_data_t* data) {
	if (input == 0) {
		data->label = 0;

		data->input[0] = 0.0;
		data->input[1] = 0.0;
	}
	else if (input == 1) {
		data->label = 1;

		data->input[0] = 1.0;
		data->input[1] = 0.0;
	}
	else if (input == 2) {
		data->label = 1;

		data->input[0] = 0.0;
		data->input[1] = 1.0;
	}
	else if (input == 3) {
		data->label = 0;

		data->input[0] = 1.0;
		data->input[1] = 1.0;
	}

	return;
}

void xor_shuffle() {
	for (uint16_t i = 0; i < 4; i++) {
		uint16_t idx_array = rand() % 4;

		xor_data_t tmp = xor_dataset[i];
		xor_dataset[i] = xor_dataset[idx_array];
		xor_dataset[idx_array] = tmp;
	}
}

void rambutan_run() {
	uint16_t numOfNeurons[] = { 2, 32, 32, 1 };

	activation_t activations[3] = { relu, relu, linear};

	neuralnetwork_init(&rambutan, 4, numOfNeurons, activations);

	neuralnetwork_initRandomWeights(&rambutan);

	neuralnetwork_initZeroBias(&rambutan);

	for (uint8_t i = 0; i < sizeof(xor_datainit); i++) {
		xordataset_init(xor_datainit[i], &xor_dataset[i]);
	}

	for (uint32_t i = 0; i < 100000; i++) {
		printf("iteration-%d\t", i);

		floating_point loss = 0.f;

		for (uint16_t j = 0; j < sizeof(xor_datainit); j++) {
			neuralnetwork_input(&rambutan, xor_dataset[j].input);
			neuralnetwork_feedforward(&rambutan);

			//neuralnetwork_print(&rambutan);

			floating_point desiredOutput = (floating_point)xor_dataset[j].label;
			loss += fabsf(desiredOutput - *rambutan.outputLayer->neurons.actv);

			neuralnetwork_backpropagate(&rambutan, &desiredOutput, 0.01f);
			//printf("Input: %.1f ^ %.1f\tOutput:%f\r\n", xor_dataset[j].input[0], xor_dataset[j].input[1], 
			//	*rambutan.outputLayer->neurons.actv);
			//printf("\r\n\r\n");
		}

		loss /= 4.f;

		printf("Loss: %f\r\n", loss);

		if (loss <= 0.001) break;

		xor_shuffle();
	}

	//neuralnetwork_print(&rambutan);

	//neuralnetwork_print(&rambutan);

	return;
}