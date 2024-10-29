#define _CRT_SECURE_NO_WARNINGS

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "neuralnetwork.h"

uint16_t macNumber = 0;

static void matvecmult(floating_point** mat, floating_point* vec, floating_point* out, uint16_t rows, uint16_t columns) {
	for (uint16_t i = 0; i < rows; i++) {
		out[i] = 0.0f;

		for (uint16_t j = 0; j < columns; j++) {
			out[i] += (mat[i][j] * vec[j]);
		}
	}

	return;
}

static void vecsadd(floating_point* vec1, floating_point* vec2, floating_point* out, uint16_t vecSize) {
	while (vecSize--) {
		*(out++) = *(vec1++) + *(vec2)++;
	}

	return;
}

static void vecssub(floating_point* vec1, floating_point* vec2, floating_point* out, uint16_t vecSize) {
	while (vecSize--) {
		*(out++) = *(vec1++) - *(vec2)++;
	}

	return;
}

static floating_point _derivative(void (*f)(floating_point, floating_point*), floating_point x) {
	const floating_point t = 1.0e-6;
	//const floating_point t = 1.0f;

	floating_point output;

	floating_point xA = x + t;
	floating_point xB = x - t;

	floating_point yA;
	floating_point yB;

	f(xA, &yA);
	f(xB, &yB);

	output = (yA - yB) / (xA - xB);

	return output;
}

static void _neuralnetwork_initLayer(layer_t* layer, uint16_t numOfNeurons,
	layer_t* prevLayer, activation_t* activation) {
	if (prevLayer == NULL || activation == NULL) {
		layer->neurons.weights = NULL;
		layer->neurons.bias = NULL;
		layer->neurons.z = NULL;
		layer->neurons.dz = NULL;
	}
	else {
		prevLayer->next = layer;

		layer->neurons.weights = malloc(sizeof(floating_point*) * numOfNeurons);
		layer->neurons.dweights = malloc(sizeof(floating_point*) * numOfNeurons);

		layer->neurons.bias = malloc(sizeof(floating_point) * numOfNeurons);
		layer->neurons.dbias = malloc(sizeof(floating_point) * numOfNeurons);

		for (uint16_t i = 0; i < numOfNeurons; i++) {
			layer->neurons.weights[i] = malloc(sizeof(floating_point) * prevLayer->neurons.numOfNeurons);
			layer->neurons.dweights[i] = malloc(sizeof(floating_point) * prevLayer->neurons.numOfNeurons);
			//layer->neurons.z[i] = 0.f;
			//layer->neurons.dz[i] = 0.f;
		}
	}

	layer->neurons.numOfNeurons = numOfNeurons;
	layer->neurons.actv = malloc(sizeof(floating_point) * numOfNeurons);
	layer->neurons.dactv = malloc(sizeof(floating_point) * numOfNeurons);

	layer->neurons.z = malloc(sizeof(floating_point) * numOfNeurons);
	layer->neurons.dz = malloc(sizeof(floating_point) * numOfNeurons);

	layer->activation = activation;
	layer->prev = prevLayer;
	layer->next = NULL;

	return;
}

static uint8_t neuralnetwork_initNeuronWeights(floating_point* weightDst, uint16_t numOfPrevlayerNeurons, floating_point* weightSrc) {
	while (numOfPrevlayerNeurons--) {
		*(weightDst++) = *(weightSrc++);
	}

	return 1;
}

static uint8_t neuralnetwork_initLayerRandomWeights(layer_t* layer) {
	const floating_point Max = 1.0;
	const floating_point Min = -1.0;
	floating_point** weights = malloc(sizeof(floating_point*) * layer->neurons.numOfNeurons);

	for (uint16_t i = 0; i < layer->neurons.numOfNeurons; i++) {
		*(weights + i) = malloc(sizeof(floating_point) * layer->prev->neurons.numOfNeurons);

		for (uint16_t j = 0; j < layer->prev->neurons.numOfNeurons; j++) {
			(*(weights + i))[j] = (((floating_point)rand() / (floating_point)RAND_MAX) * (Max - Min)) + Min;
		}
	}

	neuralnetwork_initLayerNeuronsWeight(layer, weights);

	for (uint16_t i = 0; i < layer->neurons.numOfNeurons; i++) {
		free(*(weights + i));
	}

	free(weights);

	return 1;
}

uint8_t neuralnetwork_initLayerNeuronsWeight(layer_t* layer, floating_point** weightSrc) {
	if (layer->prev == NULL || layer->neurons.weights == NULL) return -1;

	floating_point** weightDst = layer->neurons.weights;

	for (uint16_t i = 0; i < layer->neurons.numOfNeurons; i++) {
		neuralnetwork_initNeuronWeights(weightDst[i], layer->prev->neurons.numOfNeurons, weightSrc[i]);
	}

	return 1;
}

void neuralnetwork_initLayerBias(layer_t* layer, floating_point* bias) {
	for (uint16_t i = 0; i < layer->neurons.numOfNeurons; i++) {
		layer->neurons.bias[i] = bias[i];
	}

	return;
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

void neuralnetwork_initZeroBias(neuralnetwork_t* nn) {
	layer_t* pLayer = nn->layers.next;
	uint16_t i = 0;

	while (pLayer != NULL) {
		floating_point* bias = malloc(pLayer->neurons.numOfNeurons * sizeof(floating_point));

		for (uint16_t i = 0; i < pLayer->neurons.numOfNeurons; i++) {
			bias[i] = 0.0f;
		}

		neuralnetwork_initLayerBias(pLayer, bias);

		free(bias);

		pLayer = pLayer->next;
	}

	return;
}

uint8_t neuralnetwork_init(neuralnetwork_t* nn, uint16_t numOfLayers, uint16_t* numOfNeurons,
	activation_t* activations) {
	if (numOfLayers < 3) return -1;

	srand(time(NULL));

	nn->layers.prev = NULL;
	strcpy(nn->layers.name, "Input Layer");

	layer_t* layer = &nn->layers;

	_neuralnetwork_initLayer(layer, *numOfNeurons, NULL, NULL);

	for (uint16_t i = 1; i < numOfLayers; i++) {
		layer->next = malloc(sizeof(layer_t));
		_neuralnetwork_initLayer(layer->next, numOfNeurons[i], layer, &activations[i - 1]);

		sprintf(layer->next->name, "Hidden Layer %d", i);

		layer = layer->next;
	}

	strcpy(layer->name, "Output Layer");

	nn->inputLayer = &nn->layers;
	nn->outputLayer = layer;

	return 1;
}

uint8_t neuralnetwork_input(neuralnetwork_t* nn, floating_point* input) {
	for (uint16_t i = 0; i < nn->inputLayer->neurons.numOfNeurons; i++) {
		nn->inputLayer->neurons.actv[i] = input[i];
	}

	return 1;
}

uint8_t neuralnetwork_feedforward(neuralnetwork_t* nn) {
	layer_t* pLayer = nn->inputLayer;

	while (pLayer->next != NULL) {
		matvecmult(pLayer->next->neurons.weights, pLayer->neurons.actv, pLayer->next->neurons.z,
			pLayer->next->neurons.numOfNeurons, pLayer->neurons.numOfNeurons);

		vecsadd(pLayer->next->neurons.z, pLayer->next->neurons.bias,
			pLayer->next->neurons.actv, pLayer->next->neurons.numOfNeurons);

		pLayer->next->activation->function(pLayer->next->neurons.actv, pLayer->next->neurons.actv,
			pLayer->next->neurons.numOfNeurons);

		pLayer = pLayer->next;
	}

	return 1;
}

uint8_t neuralnetwork_layerBackpropagate(layer_t* layer, floating_point* desiredActv) {
	if (layer->next == NULL) {
		floating_point* errors = malloc(sizeof(floating_point) * layer->neurons.numOfNeurons);

		vecssub(layer->neurons.actv, desiredActv, errors, layer->neurons.numOfNeurons);

		layer->activation->derivative(layer->neurons.z, layer->neurons.dz, layer->neurons.numOfNeurons);

		for (uint16_t i = 0; i < layer->neurons.numOfNeurons; i++) {
			layer->neurons.dactv[i] = errors[i];
			//layer->neurons.dz[i] = _derivative(layer->activation, layer->neurons.actv[i]);

			floating_point tmp = layer->neurons.dz[i] * layer->neurons.dactv[i];

			for (uint16_t j = 0; j < layer->prev->neurons.numOfNeurons; j++) {
				layer->neurons.dweights[i][j] = tmp * layer->prev->neurons.actv[j];
			}

			layer->neurons.dbias[i] = tmp;
		}
	}
	else {
		layer->activation->derivative(layer->neurons.z, layer->neurons.dz, layer->neurons.numOfNeurons);

		for (uint16_t i = 0; i < layer->neurons.numOfNeurons; i++) {
			//layer->neurons.dz[i] = _derivative(layer->activation, layer->neurons.actv[i]);
			layer->neurons.dactv[i] = 0.f;

			for (uint16_t j = 0; j < layer->next->neurons.numOfNeurons; j++) {
				layer->neurons.dactv[i] += ((layer->next->neurons.dz[j] * layer->next->neurons.dactv[j])
					* layer->next->neurons.weights[j][i]);
			}

			floating_point tmp = layer->neurons.dactv[i] * layer->neurons.dz[i];

			for (uint16_t j = 0; j < layer->prev->neurons.numOfNeurons; j++) {
				layer->neurons.dweights[i][j] = tmp * layer->prev->neurons.z[j];
			}

			layer->neurons.dbias[i] = tmp;
		}
	}

	return 1;
}

uint8_t neuralnetwork_backpropagate(neuralnetwork_t* nn, floating_point* desiredOutput, floating_point alpha) {
	for (uint16_t i = 0; i < nn->inputLayer->neurons.numOfNeurons; i++) {
		nn->inputLayer->neurons.z[i] = nn->inputLayer->neurons.actv[i];
	}

	layer_t* pLayer = nn->outputLayer;

	while (pLayer->prev != NULL) {
		neuralnetwork_layerBackpropagate(pLayer, desiredOutput);

		desiredOutput = NULL;

		pLayer = pLayer->prev;
	}

	pLayer = nn->inputLayer;

	while (pLayer->next != NULL) {
		for (uint16_t i = 0; i < pLayer->next->neurons.numOfNeurons; i++) {
			for (uint16_t j = 0; j < pLayer->neurons.numOfNeurons; j++) {
				pLayer->next->neurons.weights[i][j] -= (pLayer->next->neurons.dweights[i][j] * alpha);
			}

			pLayer->next->neurons.bias[i] -= (pLayer->next->neurons.dbias[i] * alpha);
		}

		pLayer = pLayer->next;
	}

	return 1;
}

void neuralnetwork_calculateError(neuralnetwork_t* nn, floating_point* desiredOutput, floating_point* output) {
	for (uint16_t i = 0; i < nn->outputLayer->neurons.numOfNeurons; i++) {
		output[i] = nn->outputLayer->neurons.actv[i] - desiredOutput[i];
	}

	return;
}

floating_point neuralnetwork_calculateLoss(neuralnetwork_t* nn, floating_point* desiredOutput) {
	floating_point loss = 0.0;

	for (uint16_t i = 0; i < nn->outputLayer->neurons.numOfNeurons; i++) {
		//loss += powf(nn->outputLayer->neurons.actv[i] - desiredOutput[i], 2
		loss += fabsf(nn->outputLayer->neurons.actv[i] - desiredOutput[i]);
	}

	loss /= nn->outputLayer->neurons.numOfNeurons;

	return loss;
}

uint8_t neuralnetwork_print(neuralnetwork_t* nn) {
	printf("Neural Network Print\r\n\r\n");

	printf("%s\r\n", nn->inputLayer->name);

	printf("Neurons Input Values: ");

	for (uint16_t i = 0; i < nn->inputLayer->neurons.numOfNeurons; i++) {
		printf("%f\t", nn->inputLayer->neurons.actv[i]);
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

			printf("Value: %f\r\n", pLayer->neurons.actv[i]);
		}

		printf("\r\n\r\n");
		pLayer = pLayer->next;
	}

	return 1;
}