#ifndef _LSTM_H
#define _LSTM_H
#include <stdio.h>
#include <math.h>
#include <omp.h>
typedef enum {
    TEMP_GATE    = 0,
    GATE_I       = 1,
    GATE_F       = 2,
    GATE_O       = 3,
    GATE_C       = 4,
    C_TANH       = 5,
    GRAD_OUTPUT_C   = 6,
    GRAD_GATE_F  = 7,
    GRAD_GATE_I  = 8,
    GRAD_GATE_C  = 9,
    GRAD_GATE_O  = 10,
    GRAD_TEMP_GATE    = 11,
    TEMP_BIAS         = 12,

} LSTMTensorIndex;
#endif