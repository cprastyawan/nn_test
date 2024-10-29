#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mnist_csv.h"
#include "neuralnetwork.h"

extern uint64_t ns();

csvparser_t parser;

static const char* filename_train = "dataset/mnist_train.csv";

static void mnist_getDesiredOutput(uint8_t label, floating_point* desiredOutput) {
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

int serigala_run() {
#define NUMOFLAYERS 4
#define NUMOFINPUTLAYERS	(28 * 28)

#define MNIST_TESTSIZE	1024

	csvparser_init(&parser, filename_train);

	char* header = NULL;

	header = csvparser_readLine(&parser);
	mnist_data_t* mnist_out = malloc(sizeof(mnist_data_t) * MNIST_TESTSIZE);
	mnist_parse(&parser, mnist_out, MNIST_TESTSIZE);

	fclose(parser.fp);

	neuralnetwork_t serigala;
	uint16_t numOfNeurons[NUMOFLAYERS] = { NUMOFINPUTLAYERS, 64, 64, 10 };
	activation_t activations[NUMOFLAYERS - 1] = { relu, relu, softmax };

	neuralnetwork_init(&serigala, NUMOFLAYERS, numOfNeurons, activations);

	neuralnetwork_initRandomWeights(&serigala);
	neuralnetwork_initZeroBias(&serigala);

	mnist_data_t lastMnistOut;

	for (uint16_t i = 0; i < 1000; i++) {
		double loss = 0.f;

		mnist_shuffle(mnist_out, MNIST_TESTSIZE);

		for (uint16_t j = 0; j < MNIST_TESTSIZE; j++) {
			floating_point desiredOutput[10];

			neuralnetwork_input(&serigala, mnist_out[j].data);

			neuralnetwork_feedforward(&serigala);

			mnist_getDesiredOutput(mnist_out[j].label, desiredOutput);

			loss += neuralnetwork_calculateLoss(&serigala, desiredOutput);

			neuralnetwork_backpropagate(&serigala, desiredOutput, 0.01);

			lastMnistOut = mnist_out[j];
		}

		printf("Loss-%d: %f\r\n", i, loss / MNIST_TESTSIZE);
	}

	return 0;
}