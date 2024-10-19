#include "mnist_csv.h"

#include <stdlib.h>

uint8_t csvparser_init(csvparser_t* parser, const char* filename) {
	parser->fp = fopen(filename, "r");

	if (parser->fp == NULL) return -1;

	return 1;
}

char* csvparser_readLine(csvparser_t* parser) {
	fgets(parser->buffer, MAX_LINE_LENGTH, parser->fp);

	return parser->buffer;
}

char** csvparser_parseLine(csvparser_t* parser, char* line) {
	char* pStart = line;
	char* pEnd = line;
	uint16_t count = 0;

	for (uint16_t i = 0; i < MAX_ROW_SIZE; i++) {
		parser->parsed_buffer[i] = NULL;
	}

	while (count < MAX_ROW_SIZE && *pEnd != 0) {
		if (*pEnd == ',' || *pEnd == '\n') {
			parser->parsed_buffer[count++] = pStart;

			*pEnd = 0;

			pStart = pEnd + 1;
		}

		pEnd++;
	}

	return parser->parsed_buffer;
}

void mnist_shuffle(mnist_data_t* mnist_data, uint16_t size) {
	for (uint16_t i = 0; i < size; i++) {
		uint16_t idx_array = rand() % size;

		mnist_data_t tmp = mnist_data[i];
		mnist_data[i] = mnist_data[idx_array];
		mnist_data[idx_array] = tmp;
	}

	return;
}

static void parse_mnist_line(csvparser_t* parser, mnist_data_t* out) {
	out->label = atoi(parser->parsed_buffer[0]);

	for (uint16_t i = 0; i < (28 * 28); i++) {
		out->data[i] = (float)atoi(parser->parsed_buffer[i + 1]) / 255.f;
	}

	return;
}

void mnist_parse(csvparser_t* parser, mnist_data_t* out, uint16_t size) {
	for (uint16_t i = 0; i < size; i++) {
		char* line = csvparser_readLine(parser);

		csvparser_parseLine(parser, line);

		parse_mnist_line(parser, &out[i]);
	}

	return;
}