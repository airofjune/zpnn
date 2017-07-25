#ifndef _LSTM_H
#define _LSTM_H
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>

#define LOG_ON 0

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


struct Profiler
{
    float gemm;
    float tanh;
    float sig;
    float lin_f;
    float lin_b;
    float gate_sig_f;
    float tanh_dot_f;
    float tanh_dot_b;
    float copy_split_f;
    float copy_split_b;
    float dot_out_f;
    float dot_out_b;
    float sum;
};

struct Timer
{
     struct timeval t0;
};

static void Start(struct Timer* a)
{
    gettimeofday(&a->t0,NULL);
}
static float End(struct Timer* a)
{
    struct timeval t1;
    gettimeofday(&t1, NULL);
    return (t1.tv_sec - a->t0.tv_sec) + (double)(t1.tv_usec - a->t0.tv_usec)/1000/1000;
}
static struct Profiler prof = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

#endif
