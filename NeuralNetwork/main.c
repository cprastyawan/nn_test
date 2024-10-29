#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mnist_csv.h"
#include "neuralnetwork.h"

extern void init_weightsandbiases();
extern floating_point** get_weightspointer(uint16_t layer);
extern floating_point* get_biaspointer(uint16_t layer);

extern uint64_t ns();

extern void rambutan_run();

csvparser_t parser;

static const char* filename_train = "dataset/mnist_train.csv";

//floating_point* softmax_param_pz;
//uint16_t* softmax_param_num_neurons;

void mnist_getDesiredOutput(uint8_t label, floating_point* desiredOutput) {
	for (uint8_t i = 0; i < 10; i++) {
		desiredOutput[i] = label == i ? 1.0f : 0.0f;
	}

	return;
}
/*
static void activation_softmax(floating_point input, floating_point* output) {
	double total = 0.0;
	floating_point maxVal = *softmax_param_pz;

	for (uint16_t i = 1; i < *softmax_param_num_neurons; i++) {
		maxVal = maxVal < softmax_param_pz[i] ? softmax_param_pz[i] : maxVal;
	}

	for (uint16_t i = 0; i < *softmax_param_num_neurons; i++) {
		//floating_point tmp = softmax_param_pz[i] > 80 ? 80 : softmax_param_pz[i];

		total += exp(softmax_param_pz[i] - maxVal);
	}

	if (total == 0.0) total = 1.0e-3f;

	*output = (floating_point)((exp(input - maxVal) / (floating_point)total));

	return;
}

static void softmax_init(neuralnetwork_t* nn) {
	softmax_param_pz = nn->outputLayer->neurons.z;
	softmax_param_num_neurons = &nn->outputLayer->neurons.numOfNeurons;

	return;
}

static void activation_relu(floating_point input, floating_point* output) {
	*output = input > 0.f ? input : 0.f;

	return;
}

static void activation_sigmoid(floating_point input, floating_point* output) {
	*output = (1.0f / (1.0f + expf(input)));

	return;
}

static void activation_linear(floating_point input, floating_point* output) {
	*output = input;

	return;
}
*/

static uint8_t mnist_correct(uint8_t label, floating_point* actv) {
	floating_point maxActv = *actv;
	uint8_t maxIdx = 0;

	for (uint8_t i = 1; i < 10; i++) {
		if (maxActv < actv[i]) {
			maxActv = actv[i];

			maxIdx = i;
		}
	}

	return label == maxIdx ? 1 : 0;
}

void ready_run() {
#define NUMOFLAYERS 4
#define NUMOFINPUTLAYERS	(28 * 28)

	init_weightsandbiases();

#define MNIST_TESTSIZE	10000

	csvparser_init(&parser, filename_train);

	char* header = NULL;

	header = csvparser_readLine(&parser);
	mnist_data_t* mnist_out = malloc(sizeof(mnist_data_t) * MNIST_TESTSIZE);
	mnist_parse(&parser, mnist_out, MNIST_TESTSIZE);

	fclose(parser.fp);

	neuralnetwork_t serigala;
	uint16_t numOfNeurons[NUMOFLAYERS] = { NUMOFINPUTLAYERS, 256, 256, 10 };
	activation_t activations[NUMOFLAYERS - 1] = { relu, relu, softmax };

	neuralnetwork_init(&serigala, NUMOFLAYERS, numOfNeurons, activations);

	neuralnetwork_initLayerNeuronsWeight(serigala.inputLayer->next, get_weightspointer(0));
	neuralnetwork_initLayerNeuronsWeight(serigala.inputLayer->next->next, get_weightspointer(1));
	neuralnetwork_initLayerNeuronsWeight(serigala.outputLayer, get_weightspointer(2));

	neuralnetwork_initLayerBias(serigala.inputLayer->next, get_biaspointer(0));
	neuralnetwork_initLayerBias(serigala.inputLayer->next->next, get_biaspointer(1));
	neuralnetwork_initLayerBias(serigala.outputLayer, get_biaspointer(2));

	uint16_t numCorrect = 0;

	for (uint16_t i = 0; i < MNIST_TESTSIZE; i++) {
		floating_point desiredOutput[10];

		neuralnetwork_input(&serigala, mnist_out[i].data);

		uint64_t t = ns();

		neuralnetwork_feedforward(&serigala);

		t = ns() - t;

		mnist_getDesiredOutput(mnist_out[i].label, desiredOutput);

		uint8_t isCorrect = mnist_correct(mnist_out[i].label, serigala.outputLayer->neurons.actv);

		floating_point loss = neuralnetwork_calculateLoss(&serigala, desiredOutput);

		printf("I-%d\tlabel: %d\tLoss: %f\tCorrect: %s\tTime: %llu\r\n", i, mnist_out[i].label, loss,
			isCorrect == 1 ? "Yes" : "No", t);

		numCorrect += isCorrect;
	}

	floating_point accuracy = (floating_point)numCorrect / (floating_point)MNIST_TESTSIZE;

	printf("Total Correct : %d, Accuracy: %f\r\n", numCorrect, accuracy);

	return;
}

int main() {
	//ready_run();
	//rambutan_run();
	serigala_run();

	return 0;
}