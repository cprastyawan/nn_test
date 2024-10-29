#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "floating_point.h"

static floating_point** weights_inputlayer_to_hiddenlayer1;
static floating_point* bias_inputlayer_to_hiddenlayer1;

static floating_point** weights_hiddenlayer1_to_hiddenlayer2;
static floating_point* bias_hiddenlayer1_to_hiddenlayer2;

static floating_point** weights_hiddenlayer2_to_outputlayer;
static floating_point* bias_hiddenlayer2_to_outputlayer;

static void get_weights(const char* filename, floating_point*** p, uint16_t numOfNeurons, uint16_t numOfPrevLayerNeurons) {
	FILE* fp = fopen(filename, "rb");

	*p = malloc(sizeof(floating_point*) * numOfNeurons);

	for (uint16_t i = 0; i < numOfNeurons; i++) {
		(*p)[i] = malloc(sizeof(floating_point) * numOfPrevLayerNeurons);

		fread((*p)[i], sizeof(floating_point), numOfPrevLayerNeurons, fp);
	}

	fclose(fp);

	return;
}

static void get_biases(const char* filename, floating_point** p, uint16_t numOfNeurons) {
	FILE* fp = fopen(filename, "rb");

	*p = malloc(sizeof(floating_point) * numOfNeurons);

	fread(*p, sizeof(floating_point), numOfNeurons, fp);

	fclose(fp);

	return;
}

void init_weightsandbiases() {
	get_weights("weights/weights_inputlayer_to_hiddenlayer1.bin", &weights_inputlayer_to_hiddenlayer1, 256, 784);
	get_weights("weights/weights_hiddenlayer1_to_hiddenlayer2.bin", &weights_hiddenlayer1_to_hiddenlayer2, 256, 256);
	get_weights("weights/weights_hiddenlayer2_to_outputlayer.bin", &weights_hiddenlayer2_to_outputlayer, 10, 256);

	get_biases("weights/bias_inputlayer_to_hiddenlayer1.bin", &bias_inputlayer_to_hiddenlayer1, 256);
	get_biases("weights/bias_hiddenlayer1_to_hiddenlayer2.bin", &bias_hiddenlayer1_to_hiddenlayer2, 256);
	get_biases("weights/bias_hiddenlayer2_to_outputlayer.bin", &bias_hiddenlayer2_to_outputlayer, 10);

	return;
}

floating_point** get_weightspointer(uint16_t layer) {
	if (layer == 0) {
		return weights_inputlayer_to_hiddenlayer1;
	}
	else if (layer == 1) {
		return weights_hiddenlayer1_to_hiddenlayer2;
	}
	else if (layer == 2) {
		return weights_hiddenlayer2_to_outputlayer;
	}

	return NULL;
}

floating_point* get_biaspointer(uint16_t layer) {
	if (layer == 0) {
		return bias_inputlayer_to_hiddenlayer1;
	}
	else if (layer == 1) {
		return bias_hiddenlayer1_to_hiddenlayer2;
	}
	else if (layer == 2) {
		return bias_hiddenlayer2_to_outputlayer;
	}

	return NULL;
}