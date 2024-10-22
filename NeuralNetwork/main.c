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
	float* dZ;
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

extern void init_weightsandbiases();
extern float** get_weightspointer(uint16_t layer);
extern float* get_biaspointer(uint16_t layer);

extern uint64_t ns();

static const char* filename_train = "dataset/mnist_test.csv";

float* softmax_param_pTmpValues;
uint16_t* softmax_param_num_neurons;

csvparser_t parser;

uint16_t macNumber = 0;

static void activation_softmax(float input, float* output) {
	float total = 0.0f;

	for (uint16_t i = 0; i < *softmax_param_num_neurons; i++) {
		total += expf(softmax_param_pTmpValues[i]);
	}

	*output = (expf(input) / total);

	return;
}

static void softmax_init(neuralnetwork_t* nn) {
	softmax_param_pTmpValues = nn->outputLayer->neurons.tmpValues;
	softmax_param_num_neurons = &nn->outputLayer->neurons.numOfNeurons;

	return;
}

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
		layer->neurons.dZ = NULL;
	} else {
		prevLayer->next = layer;

		layer->neurons.weights = malloc(sizeof(float*) * numOfNeurons);
		layer->neurons.biases = malloc(sizeof(float) * numOfNeurons);
		layer->neurons.tmpValues = malloc(sizeof(float) * numOfNeurons);
		layer->neurons.dZ = malloc(sizeof(float) * numOfNeurons);

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
		out[i] = 0.0f;

		for (uint16_t j = 0; j < columns; j++) {
			out[i] += (mat[i][j] * vec[j]);
		}
	}

	return;
}

static void vecsadd(float* vec1, float* vec2, float* out, uint16_t vecSize) {
	while (vecSize--) {
		*(out++) = *(vec1++) + *(vec2)++;
	}

	return;
}

static void vecssub(float* vec1, float* vec2, float* out, uint16_t vecSize) {
	while (vecSize--) {
		*(out++) = *(vec1++) - *(vec2)++;
	}

	return;
}

uint8_t neuralnetwork_input(neuralnetwork_t* nn, float* input) {
	for (uint16_t i = 0; i < nn->inputLayer->neurons.numOfNeurons; i++) {
		nn->inputLayer->neurons.values[i] = input[i];
	}

	return 1;
}

uint8_t neuralnetwork_feedforward(neuralnetwork_t* nn) {
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

float neuralnetwork_derivative(void (*f)(float, float*), float x) {
	const float t = 1.0e-6;

	float output;

	float xA = x + t;
	float xB = x - t;

	float yA;
	float yB;

	f(xA, &yA);
	f(xB, &yB);

	output = (yA - yB) / (xA - xB);

	return output;
}

uint8_t neuralnetwork_layerBackpropagate(layer_t* layer, float *desiredValues) {
	const float alpha = 0.01f;
	float* errors = malloc(sizeof(float) * layer->neurons.numOfNeurons);

	vecssub(layer->neurons.values, desiredValues, errors, layer->neurons.numOfNeurons);

	if (layer->next == NULL) {
		for (uint16_t i = 0; i < layer->neurons.numOfNeurons; i++) {
			layer->neurons.dZ[i] = errors[i] * neuralnetwork_derivative(layer->activation, layer->neurons.tmpValues[i]);

			for (uint16_t j = 0; j < layer->prev->neurons.numOfNeurons; j++) {
				layer->neurons.weights[i][j] -= (alpha * (layer->neurons.dZ[i] * layer->prev->neurons.tmpValues[i]));
			}

			layer->neurons.biases[i] -= (alpha * layer->neurons.dZ[i]);
		}
	}
	else {
		for (uint16_t i = 0; i < layer->neurons.numOfNeurons; i++) {
			float tmp = 0.0f;

			for (uint16_t j = 0; j < layer->next->neurons.numOfNeurons; j++) {
				tmp += layer->next->neurons.dZ[i] * layer->next->neurons.weights[j][i];
			}
		}
	}


	return 1;
}

uint8_t neuralnetwork_backpropagate(neuralnetwork_t* nn, float* desiredOutput) {
	neuralnetwork_layerBackpropagate(nn->outputLayer, desiredOutput);

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

uint8_t xor_dataset[] = { 0, 1, 0, 3, 2, 1, 3, 2, 2, 1, 0, 2, 1, 3 };

void xor_desiredoutput(uint8_t input, float* output) {
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

int main() {
#define NUMOFLAYERS 4
#define NUMOFINPUTLAYERS	(28 * 28)

	srand((uint16_t)time(NULL));

	init_weightsandbiases();

#define MNIST_TESTSIZE	512

	csvparser_init(&parser, filename_train);

	char* header = NULL;

	header = csvparser_readLine(&parser);
	mnist_data_t* mnist_out = malloc(sizeof(mnist_data_t) * MNIST_TESTSIZE);
	mnist_parse(&parser, mnist_out, MNIST_TESTSIZE);

	fclose(parser.fp);

	mnist_shuffle(mnist_out, MNIST_TESTSIZE);

	neuralnetwork_t serigala;
	uint16_t numOfNeurons[NUMOFLAYERS] = { NUMOFINPUTLAYERS, 256, 256, 10};
	void (*activations[NUMOFLAYERS - 1])(float, float*) = { activation_relu, activation_relu, activation_softmax };

	neuralnetwork_init(&serigala, NUMOFLAYERS, numOfNeurons, activations);

	softmax_init(&serigala);

	neuralnetwork_initLayerNeuronsWeight(serigala.inputLayer->next, get_weightspointer(0));
	neuralnetwork_initLayerNeuronsWeight(serigala.inputLayer->next->next, get_weightspointer(1));
	neuralnetwork_initLayerNeuronsWeight(serigala.outputLayer, get_weightspointer(2));

	neuralnetwork_initLayerBias(serigala.inputLayer->next, get_biaspointer(0));
	neuralnetwork_initLayerBias(serigala.inputLayer->next->next, get_biaspointer(1));
	neuralnetwork_initLayerBias(serigala.outputLayer, get_biaspointer(2));
	//neuralnetwork_initRandomWeights(&serigala);
	//neuralnetwork_initZeroBias(&serigala);

	float totalLosses = 0.0f;

	for (uint16_t i = 0; i < MNIST_TESTSIZE; i++) {
		float desiredOutput[MNIST_TESTSIZE];

		neuralnetwork_input(&serigala, mnist_out[i].data);
		
		uint64_t t = ns();

		neuralnetwork_feedforward(&serigala);

		t = ns() - t;
		mnist_getDesiredOutput(mnist_out[i].label, desiredOutput);

		float loss = neuralnetwork_calculateLoss(&serigala, desiredOutput);
		printf("I-%d\tlabel: %d\tLoss: %f\tTime: %llu\r\n", i, mnist_out[i].label, loss, t);

		totalLosses += loss;
	}

	printf("Total Loss: %f, Accuracy: %f\r\n", totalLosses / MNIST_TESTSIZE, (1 - totalLosses / MNIST_TESTSIZE));

	/*
	neuralnetwork_input(&serigala, mnist_out.data);
	neuralnetwork_feedforward(&serigala);

	float desiredOutput[10];

	mnist_getDesiredOutput(mnist_out.label, desiredOutput);

	float loss = neuralnetwork_calculateLoss(&serigala, desiredOutput);

	printf("Loss: %f\r\n", loss);
	*/
	
	//neuralnetwork_print(&serigala);

	return 0;
}