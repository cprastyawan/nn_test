#pragma once

#include <stdint.h>
#include "activation_function.h"
#include "floating_point.h"

typedef struct _neuron {
	floating_point* actv;

	floating_point* bias;
	floating_point* weights;

	uint16_t numOfNeurons;
} neuron_t;

typedef struct _layer {
	char name[32];
	neuron_t neurons;
	activation_t activation;
	struct _layer* next;
	struct _layer* prev;
} layer_t;

typedef struct {
	layer_t* inputLayer;
	layer_t* outputLayer;
	layer_t layers;
} neuralnetwork_t;

uint8_t neuralnetwork_initLayerNeuronsWeight(layer_t* layer, floating_point** weightSrc);
void neuralnetwork_initLayerBias(layer_t* layer, floating_point* bias);
void neuralnetwork_initRandomWeights(neuralnetwork_t* nn);
void neuralnetwork_initZeroBias(neuralnetwork_t* nn);
uint8_t neuralnetwork_input(neuralnetwork_t* nn, floating_point* input);
uint8_t neuralnetwork_feedforward(neuralnetwork_t* nn);
uint8_t neuralnetwork_layerBackpropagate(layer_t* layer, floating_point* desiredActv);
uint8_t neuralnetwork_backpropagate(neuralnetwork_t* nn, floating_point* desiredOutput, floating_point alpha);
void neuralnetwork_calculateError(neuralnetwork_t* nn, floating_point* desiredOutput, floating_point* output);
floating_point neuralnetwork_calculateLoss(neuralnetwork_t* nn, floating_point* desiredOutput);
uint8_t neuralnetwork_print(neuralnetwork_t* nn);
uint8_t neuralnetwork_init(neuralnetwork_t* nn, uint16_t numOfLayers, uint16_t* numOfNeurons,
	activation_t* activations);