#pragma once

#include <stdint.h>
#include "activation_function.h"

typedef struct _neuron {
	float* actv;
	float* dactv;

	float* z;
	float* dz;

	float* bias;
	float** weights;

	float* dbias;

	float** dweights;
	uint16_t numOfNeurons;
} neuron_t;

typedef struct _layer {
	char name[32];
	neuron_t neurons;
	activation_t* activation;
	struct _layer* next;
	struct _layer* prev;
} layer_t;

typedef struct {
	layer_t* inputLayer;
	layer_t* outputLayer;
	layer_t layers;
} neuralnetwork_t;

uint8_t neuralnetwork_initLayerNeuronsWeight(layer_t* layer, float** weightSrc);
void neuralnetwork_initLayerBias(layer_t* layer, float* bias);
void neuralnetwork_initRandomWeights(neuralnetwork_t* nn);
void neuralnetwork_initZeroBias(neuralnetwork_t* nn);
uint8_t neuralnetwork_input(neuralnetwork_t* nn, float* input);
uint8_t neuralnetwork_feedforward(neuralnetwork_t* nn);
uint8_t neuralnetwork_layerBackpropagate(layer_t* layer, float* desiredActv);
uint8_t neuralnetwork_backpropagate(neuralnetwork_t* nn, float* desiredOutput, float alpha);
void neuralnetwork_calculateError(neuralnetwork_t* nn, float* desiredOutput, float* output);
float neuralnetwork_calculateLoss(neuralnetwork_t* nn, float* desiredOutput);
uint8_t neuralnetwork_print(neuralnetwork_t* nn);
uint8_t neuralnetwork_init(neuralnetwork_t* nn, uint16_t numOfLayers, uint16_t* numOfNeurons,
	activation_t* activations);