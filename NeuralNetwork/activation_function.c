#include "activation_function.h"

#include <math.h>

static void f_relu(floating_point* input, floating_point* output, uint16_t size) {
	while (size--) {
		floating_point vInput = *(input++);

		*(output++) = (vInput) > 0.0 ? (vInput) : 0.0;
	}
	
	return;
}

static floating_point _expo(floating_point input) {
	floating_point result;

#if FLOATING_POINT==FLOAT
	result = expf(input);

	if (isnan(result)) {
		result = 0.001f;
	}

	return result;
#else
	floating_point result = exp(input);

	if (isnan(result)) {
		result = 1e-6;
	}
#endif

	return result;
}

static void d_relu(floating_point* input, floating_point* output, uint16_t size) {
	while (size--) {
		*(output++) = *(input++) > 0.0 ? 1 : 0.0;
	}

	return;
}

static void f_sigmoid(floating_point* input, floating_point* output, uint16_t size) {
	while (size--) {
		*(output++) = (1.0 / (1.0 + _expo(-(*(input++)))));
	}
	
	return;
}

static void d_sigmoid(floating_point* input, floating_point* output, uint16_t size) {
	f_sigmoid(input, output, size);

	while (size--) {
		floating_point vInput = *(output);

		*(output++) = vInput / (1 - vInput);
	}

	return;
}

static void f_softmax(floating_point* input, floating_point* output, uint16_t size) {
	floating_point maxVal = *input;
	floating_point total = 0.0;

	for (uint16_t i = 1; i < size; i++) {
		maxVal = input[i] > maxVal ? input[i] : maxVal;
	}

	for (uint16_t i = 0; i < size; i++) {
		total += _expo(input[i] - maxVal);
	}

	while (size--) {
		*(output++) = _expo(*(input++) - maxVal) / total;
	}

	return;
}

static void d_softmax(floating_point* input, floating_point* output, uint16_t size) {
	f_softmax(input, output, size);

	while (size--) {
		floating_point vInput = *output;
		*(output++) = vInput * (1 - vInput);
	}
	return;
}

static void f_linear(floating_point* input, floating_point* output, uint16_t size) {
	while (size--) {
		*(output++) = *(input++);
	}

	return;
}

static void d_linear(floating_point* input, floating_point* output, uint16_t size) {
	while (size--) {
		*(output++) = 1;
	}

	return;
}

static floating_point func_tanh(floating_point input) {
#if FLOATING_POINT==FLOAT
	return tanhf(input);
#else FLOATING_POINT==DOUBLE
	return tanh(input);
#endif
}

static void f_tanh(floating_point* input, floating_point* output, uint16_t size) {
	while (size--) {
		*(output++) = func_tanh(*input++);
	}

	return;
}

static void d_tanh(floating_point* input, floating_point* output, uint16_t size) {
	f_tanh(input, output, size);

	while (size--) {
		floating_point vInput = *output;

		*(output++) = 1 - pow(vInput, 2);
	}

	return;
}

activation_t relu = { .function = f_relu, .derivative = d_relu };
activation_t sigmoid = { .function = f_sigmoid, .derivative = d_sigmoid };
activation_t softmax = { .function = f_softmax, .derivative = d_softmax };
activation_t linear = { .function = f_linear, .derivative = d_linear };
activation_t act_tanh = { .function = f_tanh, .derivative = d_tanh };