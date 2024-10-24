#include "activation_function.h"

#include <math.h>

static void f_relu(float* input, float* output, uint16_t size) {
	while (size--) {
		float vInput = *(input++);

		*(output++) = (vInput) > 0.f ? (vInput) : 0.f;
	}
	
	return;
}

static void d_relu(float* input, float* output, uint16_t size) {
	while (size--) {
		*(output++) = *(input++) > 0.f ? 1 : 0.f;
	}

	return;
}

static void f_sigmoid(float* input, float* output, uint16_t size) {
	while (size--) {
		*(output++) = (1.0f / (1.0f + expf(-(*(input++)))));
	}
	
	return;
}

static void d_sigmoid(float* input, float* output, uint16_t size) {
	f_sigmoid(input, output, size);

	while (size--) {
		float vInput = *(output);

		*(output++) = vInput / (1 - vInput);
	}

	return;
}

static void f_softmax(float* input, float* output, uint16_t size) {
	float maxVal = *input;
	float total = 0.0f;

	for (uint16_t i = 1; i < size; i++) {
		maxVal = input[i] > maxVal ? input[i] : maxVal;
	}

	for (uint16_t i = 0; i < size; i++) {
		total += expf(input[i] - maxVal);
	}

	while (size--) {
		*(output++) = expf(*(input++) - maxVal) / total;
	}

	return;
}

static void d_softmax(float* input, float* output, uint16_t size) {
	return;
}

static void f_linear(float* input, float* output, uint16_t size) {
	while (size--) {
		*(output++) = *(input++);
	}

	return;
}

static void d_linear(float* input, float* output, uint16_t size) {
	while (size--) {
		*(output++) = 1;
	}

	return;
}

activation_t relu = { .function = f_relu, .derivative = d_relu };
activation_t sigmoid = { .function = f_sigmoid, .derivative = d_sigmoid };
activation_t softmax = { .function = f_softmax, .derivative = d_softmax };
activation_t linear = { .function = f_linear, .derivative = d_linear };