#include <stdint.h>
#include "csvparser.h"

uint8_t csvparser_init(csvparser_t* parser, const char* filename) {
	 fopen_s(&parser->fp, filename, "r");

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
