#pragma once

#define _CRT_SECURE_NO_WARNINGS

#include <stdlib.h>
#include <stdio.h>

#define MAX_LINE_LENGTH 8192
#define MAX_ROW_SIZE 1024

typedef struct {
	FILE* fp;
	const char* filename;
	char buffer[MAX_LINE_LENGTH];
	char* parsed_buffer[MAX_ROW_SIZE];
} csvparser_t;

uint8_t csvparser_init(csvparser_t* parser, const char* filename);
char* csvparser_readLine(csvparser_t* parser);
char** csvparser_parseLine(csvparser_t* parser, char* line);