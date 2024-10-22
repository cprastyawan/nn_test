#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

static float** weights_inputlayer_to_hiddenlayer1;
static float* bias_inputlayer_to_hiddenlayer1;

static float** weights_hiddenlayer1_to_hiddenlayer2;
static float* bias_hiddenlayer1_to_hiddenlayer2;

static float** weights_hiddenlayer2_to_outputlayer;
static float* bias_hiddenlayer2_to_outputlayer;

static void get_weights(const char* filename, float*** p, uint16_t numOfNeurons, uint16_t numOfPrevLayerNeurons) {
	FILE* fp = fopen(filename, "rb");

	*p = malloc(sizeof(float*) * numOfNeurons);

	for (uint16_t i = 0; i < numOfNeurons; i++) {
		(*p)[i] = malloc(sizeof(float) * numOfPrevLayerNeurons);

		fread((*p)[i], sizeof(float), numOfPrevLayerNeurons, fp);
	}

	fclose(fp);

	return;
}

static void get_biases(const char* filename, float** p, uint16_t numOfNeurons) {
	FILE* fp = fopen(filename, "rb");

	*p = malloc(sizeof(float) * numOfNeurons);

	fread(*p, sizeof(float), numOfNeurons, fp);

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

float** get_weightspointer(uint16_t layer) {
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

float* get_biaspointer(uint16_t layer) {
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