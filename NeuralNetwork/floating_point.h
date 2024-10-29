#pragma once
#define FLOAT 0
#define DOUBLE 1

#define FLOATING_POINT FLOAT

#if FLOATING_POINT==FLOAT
typedef float floating_point;
#else
typedef double floating_point;
#endif