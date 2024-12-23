#define _CRT_SECURE_NO_WARNINGS

#define MNIST_TESTSIZE	10000

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mnist_csv.h"
#include "neuralnetwork.h"

#include "hidden_layer_1_weights.h"
#include "hidden_layer_1_biases.h"

#include "hidden_layer_2_weights.h"
#include "hidden_layer_2_biases.h"

#include "output_layer_biases.h"
#include "output_layer_weights.h"

extern uint64_t ns();

csvparser_t parser;

static const char* filename_test = "dataset/mnist_test.csv";
static const char* filename_train = "dataset/mnist_train.csv";

//floating_point* softmax_param_pz;
//uint16_t* softmax_param_num_neurons;

void mnist_getDesiredOutput(uint8_t label, floating_point* desiredOutput) {
	for (uint8_t i = 0; i < 10; i++) {
		desiredOutput[i] = label == i ? 1.0f : 0.0f;
	}

	return;
}

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

int main() {
#define NUMOFLAYERS 4
	neuralnetwork_t serigala;

	uint16_t numOfNeurons[NUMOFLAYERS] = { 784, 20, 20, 10 };

	activation_t activations[NUMOFLAYERS - 1] = { relu, relu, softmax };

	neuralnetwork_init(&serigala, NUMOFLAYERS, numOfNeurons, activations);

	serigala.inputLayer->next->neurons.weights = (float*)weights_input_layer_to_hidden_layer_1;
	serigala.inputLayer->next->neurons.bias = (float*)biases_input_layer_to_hidden_layer_1;

	serigala.inputLayer->next->next->neurons.weights = (float*)weights_hidden_layer_1_to_hidden_layer_2;
	serigala.inputLayer->next->next->neurons.bias = (float*)biases_hidden_layer_1_to_hidden_layer_2;

	serigala.outputLayer->neurons.weights = (float*)weights_hidden_layer_2_to_output_layer;
	serigala.outputLayer->neurons.bias = (float*)biases_hidden_layer_2_to_output_layer;

	csvparser_init(&parser, filename_test);

	char* header = NULL;

	header = csvparser_readLine(&parser);

	mnist_data_t* mnist_out = malloc(sizeof(mnist_data_t) * MNIST_TESTSIZE);
	mnist_parse(&parser, mnist_out, MNIST_TESTSIZE);

	fclose(parser.fp);

	mnist_shuffle(mnist_out, MNIST_TESTSIZE);

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

	return 0;
}