#include "mnist_csv.h"
#include "floating_point.h"

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
		out->data[i] = (floating_point)atoi(parser->parsed_buffer[i + 1]) / 255.0f;
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