#pragma once
#include <stdint.h>

typedef struct {
	void (*function)(float*, float*, uint16_t);
	void (*derivative)(float*, float*, uint16_t);
} activation_t;

extern activation_t relu;
extern activation_t sigmoid;
extern activation_t softmax;
extern activation_t linear;