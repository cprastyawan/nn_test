#pragma once
#include <stdint.h>
#include "floating_point.h"

typedef struct {
	void (*function)(floating_point*, floating_point*, uint16_t);
	void (*derivative)(floating_point*, floating_point*, uint16_t);
} activation_t;

extern activation_t relu;
extern activation_t sigmoid;
extern activation_t softmax;
extern activation_t linear;