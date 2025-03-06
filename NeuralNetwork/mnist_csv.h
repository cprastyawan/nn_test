#pragma once

#define _CRT_SECURE_NO_WARNINGS

#include <stdint.h>
#include <stdio.h>
#include "floating_point.h"

#include "csvparser.h"

typedef struct {
	uint8_t label;
	floating_point data[28 * 28];
} mnist_data_t;

void mnist_shuffle(mnist_data_t* mnist_data, uint16_t size);
void mnist_parse(csvparser_t* parser, mnist_data_t* out, uint16_t size);
