#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mnist_csv.h"

typedef struct _neuron {
	float* values;
	float* tmpValues;
	float* biases;
	float** weights;
	uint16_t numOfNeurons;
} neuron_t;

typedef struct _layer {
	char name[32];
	neuron_t neurons;
	void (*activation)(float, float*);
	struct _layer* next;
	struct _layer* prev;
} layer_t;

typedef struct {
	layer_t* inputLayer;
	layer_t* outputLayer;
	layer_t layers;
} neuralnetwork_t;

static const char* filename_train = "dataset/mnist_train.csv";

csvparser_t parser;

static void activation_relu(float input, float* output) {
	*output = input > 0.f ? input : 0.f;

	return;
}

static void activation_sigmoid(float input, float* output) {
	*output = (1.0f / (1.0f + expf(input)));

	return;
}

static void activation_linear(float input, float* output) {
	*output = input;

	return;
}

static void _neuralnetwork_initLayer(layer_t* layer, uint16_t numOfNeurons, 
	layer_t* prevLayer, void (*activation)(float, float*)) {
	if (prevLayer == NULL || activation == NULL) {
		layer->neurons.weights = NULL;
		layer->neurons.biases = NULL;
		layer->neurons.tmpValues = NULL;
	} else {
		prevLayer->next = layer;

		layer->neurons.weights = malloc(sizeof(float*) * numOfNeurons);
		layer->neurons.biases = malloc(sizeof(float) * numOfNeurons);
		layer->neurons.tmpValues = malloc(sizeof(float) * numOfNeurons);

		for (uint16_t i = 0; i < numOfNeurons; i++) {
			layer->neurons.weights[i] = malloc(sizeof(float) * prevLayer->neurons.numOfNeurons);
			layer->neurons.tmpValues[i] = 0.f;
		}
	}

	layer->neurons.numOfNeurons = numOfNeurons;
	layer->neurons.values = malloc(sizeof(float) * numOfNeurons);
	
	layer->activation = activation;
	layer->prev = prevLayer;
	layer->next = NULL;

	return;
}

static uint8_t neuralnetwork_initNeuronWeights(float* weightDst, uint16_t numOfPrevlayerNeurons, float* weightSrc) {
	while (numOfPrevlayerNeurons--) {
		*(weightDst++) = *(weightSrc++);
	}

	return 1;
}

static uint8_t neuralnetwork_initLayerNeuronsWeight(layer_t* layer, float** weightSrc) {
	if (layer->prev == NULL || layer->neurons.weights == NULL) return -1;

	float** weightDst = layer->neurons.weights;
	
	for (uint16_t i = 0; i < layer->neurons.numOfNeurons; i++) {
		neuralnetwork_initNeuronWeights(weightDst[i], layer->prev->neurons.numOfNeurons, weightSrc[i]);
	}
	
	return 1;
}

static uint8_t neuralnetwork_initLayerRandomWeights(layer_t* layer) {
	const float Max = 1.f;
	const float Min = -1.f;
	float** weights = malloc(sizeof(float*) * layer->neurons.numOfNeurons);

	for (uint16_t i = 0; i < layer->neurons.numOfNeurons; i++) {
		*(weights + i) = malloc(sizeof(float) * layer->prev->neurons.numOfNeurons);

		for (uint16_t j = 0; j < layer->prev->neurons.numOfNeurons; j++) {
			(*(weights + i))[j] = (((float)rand() / (float)RAND_MAX) * (Max - Min)) + Min;
		}
	}

	neuralnetwork_initLayerNeuronsWeight(layer, weights);

	for (uint16_t i = 0; i < layer->neurons.numOfNeurons; i++) {
		free(*(weights + i));
	}

	free(weights);

	return 1;
}

void neuralnetwork_initRandomWeights(neuralnetwork_t* nn) {
	uint16_t layerIndex = 0;

	layer_t* pLayer = nn->layers.next;

	while (pLayer != NULL) {
		neuralnetwork_initLayerRandomWeights(pLayer);

		pLayer = pLayer->next;
	}

	return;
}

static void neuralnetwork_initLayerBias(layer_t* layer, float* biases) {
	for (uint16_t i = 0; i < layer->neurons.numOfNeurons; i++) {
		layer->neurons.biases[i] = biases[i];
	}

	return;
}

void neuralnetwork_initZeroBias(neuralnetwork_t* nn) {
	layer_t* pLayer = nn->layers.next;
	uint16_t i = 0;

	while (pLayer != NULL) {
		float* biases = malloc(pLayer->neurons.numOfNeurons * sizeof(float));

		for (uint16_t i = 0; i < pLayer->neurons.numOfNeurons; i++) {
			biases[i] = 0.0f;
		}

		neuralnetwork_initLayerBias(pLayer, biases);

		free(biases);

		pLayer = pLayer->next;
	}

	return;
}

uint8_t neuralnetwork_init(neuralnetwork_t* nn, uint16_t numOfLayers, uint16_t* numOfNeurons,
	void (**activations)(float, float*)) {
	if (numOfLayers < 3) return -1;

	nn->layers.prev = NULL;
	strcpy(nn->layers.name, "Input Layer");

	layer_t* layer = &nn->layers;

	_neuralnetwork_initLayer(layer, *numOfNeurons, NULL, NULL);

	for (uint16_t i = 1; i < numOfLayers; i++) {
		layer->next = malloc(sizeof(layer_t));
		_neuralnetwork_initLayer(layer->next, numOfNeurons[i], layer, activations[i - 1]);

		sprintf(layer->next->name, "Hidden Layer %d", i);

		layer = layer->next;
	}

	strcpy(layer->name, "Output Layer");

	nn->inputLayer = &nn->layers;
	nn->outputLayer = layer;

	return 1;
}

static void matvecmult(float** mat, float* vec, float* out, uint16_t rows, uint16_t columns) {
	for (uint16_t i = 0; i < rows; i++) {
		for (uint16_t j = 0; j < columns; j++) {
			out[i] += (mat[i][j] * vec[j]);
		}
	}

	return;
}

static void vecsadd(float* vec1, float* vec2, float* out, uint16_t vecSize) {
	for (uint16_t i = 0; i < vecSize; i++) {
		out[i] = vec1[i] + vec2[i];
	}

	return;
}

uint8_t neuralnetwork_input(neuralnetwork_t* nn, float* input) {
	for (uint16_t i = 0; i < nn->inputLayer->neurons.numOfNeurons; i++) {
		nn->inputLayer->neurons.values[i] = input[i];
	}

	return 1;
}

uint8_t neuralnetwork_execute(neuralnetwork_t* nn) {
	layer_t* pLayer = nn->inputLayer;

	while (pLayer->next != NULL) {
		matvecmult(pLayer->next->neurons.weights, pLayer->neurons.values, pLayer->next->neurons.tmpValues,
			pLayer->next->neurons.numOfNeurons, pLayer->neurons.numOfNeurons);

		vecsadd(pLayer->next->neurons.tmpValues, pLayer->next->neurons.biases,
			pLayer->next->neurons.values, pLayer->next->neurons.numOfNeurons);

		for (uint16_t i = 0; i < pLayer->next->neurons.numOfNeurons; i++) {
			 pLayer->next->activation(pLayer->next->neurons.values[i], 
				 &pLayer->next->neurons.values[i]);
		}

		pLayer = pLayer->next;
	}

	return 1;
}

void neuralnetwork_calculateError(neuralnetwork_t* nn, float* desiredOutput, float* output) {
	for (uint16_t i = 0; i < nn->outputLayer->neurons.numOfNeurons; i++) {
		output[i] = nn->outputLayer->neurons.values[i] - desiredOutput[i];
	}

	return;
}

float neuralnetwork_calculateLoss(neuralnetwork_t* nn, float* desiredOutput) {
	float loss = 0.0;

	for (uint16_t i = 0; i < nn->outputLayer->neurons.numOfNeurons; i++) {
		loss += powf(nn->outputLayer->neurons.values[i] - desiredOutput[i], 2);
	}

	loss /= nn->outputLayer->neurons.numOfNeurons;

	return loss;
}

void mnist_getDesiredOutput(uint8_t label, float* desiredOutput) {
	for (uint8_t i = 0; i < 10; i++) {
		desiredOutput[i] = label == i ? 1.0f : 0.0f;
	}

	return;
}

uint8_t neuralnetwork_print(neuralnetwork_t* nn) {
	printf("Neural Network Print\r\n\r\n");

	printf("%s\r\n", nn->inputLayer->name);

	printf("Neurons Input Values: ");

	for (uint16_t i = 0; i < nn->inputLayer->neurons.numOfNeurons; i++) {
		printf("%f\t", nn->inputLayer->neurons.values[i]);
	}

	printf("\r\n\r\n");

	layer_t* pLayer = nn->inputLayer->next;

	while (pLayer != NULL) {
		printf("%s\r\n", pLayer->name);

		for (uint16_t i = 0; i < pLayer->neurons.numOfNeurons; i++) {
			printf("Neuron %d\r\n", i + 1);
			printf("Weight: ");

			for (uint16_t j = 0; j < pLayer->prev->neurons.numOfNeurons; j++) {
				printf("%f\t", pLayer->neurons.weights[i][j]);
			}

			printf("\r\n");

			printf("Value: %f\r\n", pLayer->neurons.values[i]);
		}

		printf("\r\n\r\n");
		pLayer = pLayer->next;
	}

	return 1;
}

int main() {
#define NUMOFLAYERS 4
#define NUMOFINPUTLAYERS	(28 * 28)

	srand((uint16_t)time(NULL));

	csvparser_init(&parser, filename_train);

	char* header = NULL;

	header = csvparser_readLine(&parser);

	mnist_data_t* mnist_out = malloc(sizeof(mnist_data_t) * 4096);

	mnist_parse(&parser, mnist_out, 4096);

	fclose(parser.fp);

	mnist_shuffle(mnist_out, 4096);

	neuralnetwork_t serigala;
	uint16_t numOfNeurons[NUMOFLAYERS] = { NUMOFINPUTLAYERS, 16, 16, 10};
	void (*activations[NUMOFLAYERS - 1])(float, float*) = { activation_sigmoid, activation_sigmoid, activation_linear };

	neuralnetwork_init(&serigala, NUMOFLAYERS, numOfNeurons, activations);

	neuralnetwork_initRandomWeights(&serigala);
	neuralnetwork_initZeroBias(&serigala);

	/*
	neuralnetwork_input(&serigala, mnist_out.data);
	neuralnetwork_execute(&serigala);

	float desiredOutput[10];

	mnist_getDesiredOutput(mnist_out.label, desiredOutput);

	float loss = neuralnetwork_calculateLoss(&serigala, desiredOutput);

	printf("Loss: %f\r\n", loss);
	*/
	
	//neuralnetwork_print(&serigala);

	return 0;
}