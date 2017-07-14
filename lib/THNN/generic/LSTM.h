#ifndef _LSTM_H
#define _LSTM_H
#include <stdio.h>
#include <math.h>
#include <omp.h>
typedef enum {
    INPUT_C_COPY = 0,
    INPUT_H_COPY = 1,
    INPUT_X_COPY = 2,
    OUTPUT_C     = 3,
    OUTPUT_H     = 4,
    TEMP_GATE    = 5,
    GATE_I       = 6,
    GATE_F       = 7,
    GATE_O       = 8,
    GATE_C       = 9,
    C_TANH       = 10,
    GRAD_OUTPUT_C       = 11,
    GRAD_GATE_F  = 12,
    GRAD_GATE_I  = 13,
    GRAD_GATE_C  = 14,
    GRAD_GATE_O  = 15,
    GRAD_TEMP_GATE    = 16,
    TEMP_BIAS         = 17,

} LSTMTensorIndex;
#endif