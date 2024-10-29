#pragma once

#define _CRT_SECURE_NO_WARNINGS

#include <stdint.h>
#include <stdio.h>
#include "floating_point.h"

#define MAX_LINE_LENGTH 8192
#define MAX_ROW_SIZE 1024

typedef struct {
	FILE* fp;
	const char* filename;
	char buffer[MAX_LINE_LENGTH];
	char* parsed_buffer[MAX_ROW_SIZE];
} csvparser_t;

typedef struct {
	uint8_t label;
	floating_point data[28 * 28];
} mnist_data_t;

uint8_t csvparser_init(csvparser_t* parser, const char* filename);
char* csvparser_readLine(csvparser_t* parser);
char** csvparser_parseLine(csvparser_t* parser, char* line);
void mnist_shuffle(mnist_data_t* mnist_data, uint16_t size);
void mnist_parse(csvparser_t* parser, mnist_data_t* out, uint16_t size);
