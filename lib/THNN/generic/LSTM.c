#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/LSTM.c"
#else
#include "mkl.h"
#include <stdio.h>
#include <math.h>
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

static float Sigmoid(float a)
{
    return 1 / ( 1 + exp(-1*a) );
}
static float Tanh(float a)
{
    return 1 - 2 / ( 1 + exp(2*a) );
}

static void PrintVec(float* a, const long len)
{
    printf("print vec:");
    for(long i=0; i<len; ++i)
        printf("%f\t", a[i]);
    printf("\n");
}

static int CompareVec(float*a, float* b, const long len, const float gap)
{
    for(long i=0; i<len; ++i)
    {
        if(abs(a[i]-b[i]) > gap)
        {
            printf("%f\t%f\n", a[i], b[i]);
            PrintVec(a, len);
            PrintVec(b, len);
            return 1;
        }
    }
    return 0;
}

static void Linear_fprop(const long bs, const long hs, const long xl,
            float * temp_bias, float* bias_h, float* bias_x,
            float* temp_gate,  float* weight_x, float* weight_h,
             float* input_x, float *input_h)
{
    //gate = input_x * weight_x  + input_h * weight_h + bias_x + bias_h;
    //add bias for each batch
    //TODO, combine two bias together
    const long hs4 = hs*4;

    cblas_scopy(hs4, bias_x, 1, temp_bias, 1);
    cblas_saxpy(hs4, 1.0, temp_bias, 1, bias_h, 1);
    #pragma omp parallel for
    for(int i=0; i<bs; ++i)
    {
        cblas_scopy(hs4, temp_bias, 1, temp_gate+i*hs4, 1);
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bs, hs4, xl, 1, input_x, xl,
                    weight_x, hs4, 1, temp_gate, hs4);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bs, hs4, hs, 1, input_h, hs,
                    weight_h, hs4, 1, temp_gate, hs4);
}

static void Linear_bprop(float* input_h, float* input_x, float* grad_gate,
            float* weight_h,  float* grad_weight_x, float* grad_weight_h,
            float* grad_bias_x, float* grad_bias_h, float* grad_input_h,
            const long bs, const long xl, const long hs)
{
    //gate = input_x * weight_x  + input_h * weight_h + bias;
    //TODO grad must be +=
    const long hs4 = hs*4;


    //grad_input_h (bs*hs) = grad_gate(bs*hs4) * T(weight_h) (hs4*hs)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, bs, hs, hs4, 1, grad_gate, hs4,
                     weight_h, hs4, 0, grad_input_h, hs);

    //grad_weight_h (hs*hs4) = T(input_h) * grad_gate(bs*hs4)
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, hs, hs4, bs, 1, input_h, hs,
                     grad_gate, hs4, 0, grad_weight_h, hs4);

    //grad_weight_x (xl*hs4) = T(input_x) * grad_gate(bs*hs4)
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, xl, hs4, bs, 1, input_x, xl,
                     grad_gate, hs4, 0, grad_weight_x, hs4);

    //bias
    #pragma omp parallel for
    for(long i=0; i<hs4; ++i)
    {
        grad_bias_h[i] = 0;
        for(long j=0; j<bs; ++j)
            grad_bias_h[i] += grad_gate[j*hs4+i];
        grad_bias_x[i] = grad_bias_h[i];
    }
}

static void GateSigmoid_fprop(float* gateValue, const long bs, const long hs)
{
    const long hs4 = hs * 4;
    #pragma omp parallel for collapse(2)
    for(long i=0; i<bs; ++i)
    {
        for(long j=0; j<3*hs; ++j)
        {
            gateValue[i*hs4 + j] = Sigmoid(gateValue[i*hs4 + j]);
        }
    }

    #pragma omp parallel for
    for(long i=0; i<bs; ++i)
    {
        for(long j=3*hs; j<hs4; ++j)
        {
            gateValue[i*hs4 + j] = Tanh(gateValue[i*hs4 + j]);
        }
    }
}

static void CopySplit_fprop(float* pIn, const long bs, const long hs,
    float* p1, float* p2, float* p3, float* p4)
{
    #pragma omp parallel for
    for(long i=0; i<bs; ++i)
    {
        float* pStart = pIn + i*hs*4;
        const long ihs = i * hs;
        for(long j=0; j<hs; ++j)
            p1[ihs+j] = pStart[j];
        pStart += hs;
        for(long j=0; j<hs; ++j)
            p2[ihs+j] = pStart[j];
        pStart += hs;
        for(long j=0; j<hs; ++j)
            p3[ihs+j] = pStart[j];
        pStart += hs;
        for(long j=0; j<hs; ++j)
            p4[ihs+j] = pStart[j];
    }
}

static void CopySplit_bprop(float* pIn, const long bs, const long hs,
    float* p1, float* p2, float* p3, float* p4)
{
    //sigmoid and split
    //gate_i = sig(gate[:,0])
    //gate_f = sig(gate[:,1])
    //gate_o = sig(gate[:,2])
    //gate_c = tanh(gate[:,3])
    #pragma omp parallel for
    for(long i=0; i<bs; ++i)
    {
        float* pStart = pIn + i*hs*4;
        const long ihs = i * hs;
        for(long j=0; j<hs; ++j)
        {
            const float c = pStart[j];
            pStart[j] = p1[ihs+j]*c*(1-c);
        }
        pStart += hs;
        for(long j=0; j<hs; ++j)
        {
            const float c = pStart[j];
            pStart[j] = p2[ihs+j]*c*(1-c);
        }
        pStart += hs;
        for(long j=0; j<hs; ++j)
        {
            const float c = pStart[j];
            pStart[j] = p3[ihs+j]*c*(1-c);
        }
        pStart += hs;
        for(long j=0; j<hs; ++j)
        {
            const float c = pStart[j];
            pStart[j] = p4[ihs+j]*(1-c*c);
        }
    }
}

static void DotMul(float* z, const long len, const float* x, const float*y, int add)
{
    if(add)
    {
        #pragma omp parallel for
        for(long i=0; i<len; ++i)
            z[i] += x[i] * y[i];
    }
    else
    {
        #pragma omp parallel for
        for(long i=0; i<len; ++i)
            z[i] = x[i] * y[i];
    }
}

static void DotOutput_fprop(float* output_c, const long len, float* gate_f,
                    float* gate_i, float* gate_c, float* input_c)
{
    //output_c = gate_f .* input_c + gate_i .* gate_c
    DotMul(output_c, len, gate_f, input_c, 0);
    DotMul(output_c, len, gate_i, gate_c,  1);
}

static void DotOutput_bprop(float* gate_f,float* gate_i, float* gate_c, float* input_c,
                float* grad_output_c, float* grad_input_c, float* grad_gate_f,
                float* grad_gate_i, float* grad_gate_c, const long len)
{
    //output_c = gate_f .* input_c + gate_i .* gate_c
    //grad_input_c = grad_output_c .* gate_f
    //grad_gate_f  = grad_output_c .* input_c
    //grad_gate_i  = grad_output_c .* gate_c
    //grad_gate_c  = grad_output_c .* gate_i
    #pragma omp parallel for
    for(long i=0; i<len; ++i)
    {
        const float c = grad_output_c[i];
        grad_input_c[i] = c * gate_f[i];
        grad_gate_f[i]  = c * input_c[i];
        grad_gate_i[i]  = c * gate_c[i];
        grad_gate_c[i]  = c * gate_i[i];
    }
}

static void TanhDot_fprop(float* out, float* tanC, const float* in, const float* dotIn, const long len)
{
    //c_tanh = tanh(output_c)
    //output_h = gate_o .* c_tanh
    #pragma omp parallel for
    for(long i=0; i<len; ++i)
    {
        tanC[i] = Tanh(in[i]);
        out[i] = tanC[i] * dotIn[i];
    }
}

static void TanhDot_bprop(float* grad_out_c0, float* grad_out_h, float* tanhC,
    float* gate_o, float* grad_out_c1, float* grad_gate_o, const long len)
{
    //c_tanh = tanh(output_c)
    //output_h = gate_o .* c_tanh
    //grad_output_c  = grad_output_c + (1-c_tanh.*c_tanh).*gate_o.*grad_output_h
    //grad_gate_o    = grad_output_h .* c_thanh
    cblas_scopy(len, grad_out_c0, 1, grad_out_c1, 1);
    #pragma omp parallel for
    for(long i=0; i<len; ++i)
    {
        const float c = tanhC[i];
        grad_gate_o[i] = grad_out_h[i] * c;
        grad_out_c1[i] += (1 - c*c) * gate_o[i] * grad_out_h[i];
    }
}

static int InternalMemAlloc(float** prim, const long bs, const long hs)
{
    const long hs4 = hs * 4;
    prim[TEMP_GATE]    = (float*)malloc(sizeof(float)*bs*hs4);
    prim[GATE_I]       = (float*)malloc(sizeof(float)*bs*hs);
    prim[GATE_F]       = (float*)malloc(sizeof(float)*bs*hs);
    prim[GATE_C]       = (float*)malloc(sizeof(float)*bs*hs);
    prim[GATE_O]       = (float*)malloc(sizeof(float)*bs*hs);
    prim[GRAD_GATE_I]  = (float*)malloc(sizeof(float)*bs*hs);
    prim[GRAD_GATE_F]  = (float*)malloc(sizeof(float)*bs*hs);
    prim[GRAD_GATE_C]  = (float*)malloc(sizeof(float)*bs*hs);
    prim[GRAD_GATE_O]  = (float*)malloc(sizeof(float)*bs*hs);
    prim[TEMP_BIAS]    = (float*)malloc(sizeof(float)*hs4);
    prim[C_TANH]       = (float*)malloc(sizeof(float)*bs*hs);
    prim[GRAD_OUTPUT_C]       = (float*)malloc(sizeof(float)*bs*hs);
    prim[GRAD_TEMP_GATE]    = (float*)malloc(sizeof(float)*bs*hs4);

    return 0;
}

//prim, pointer to internal tensors
static void Fprop(
      float **prim,
      float *input_c,  float *input_h, float *input_x,
      float *output_c, float *output_h,
      float *weight_x, float *weight_h,
      float *bias_x, float *bias_h,
      const long bs, const long xl, const long hs,
      int init_ok)
{
    const long hs4 = hs * 4;            //4 * hidden length, weight size
    if (!init_ok)
    {
        InternalMemAlloc(prim, bs, hs);
    }

    Linear_fprop(bs, hs, xl, prim[TEMP_BIAS], bias_h, bias_x, prim[TEMP_GATE],
                weight_x, weight_h, input_x, input_h);

    //sigmoid and split
    //gate_i = sig(gate[:,0])
    //gate_f = sig(gate[:,1])
    //gate_o = sig(gate[:,2])
    //gate_c = tanh(gate[:,3])
    GateSigmoid_fprop(prim[TEMP_GATE], bs, hs);
    CopySplit_fprop(prim[TEMP_GATE], bs, hs, prim[GATE_I], prim[GATE_F], prim[GATE_O], prim[GATE_C]);

    DotOutput_fprop(output_c, bs*hs, prim[GATE_F],
                    prim[GATE_I], prim[GATE_C], input_c);

    TanhDot_fprop(output_h, prim[C_TANH], output_c, prim[GATE_O], bs*hs);
}

static void Bprop(
      float **prim,
      float *input_c, float *input_h, float *input_x,
      float *grad_out_h,  float *grad_out_c,
      float *weight_x,float *weight_h,
      float *grad_weight_x,   float *grad_weight_h,
      float *grad_bias_x,  float *grad_bias_h,
      float *grad_input_h, float *grad_input_c,
      const long bs, const long xl, const long hs)
{
    const long hs4 = hs * 4;
    TanhDot_bprop(grad_out_c, grad_out_h, prim[C_TANH], prim[GATE_O],
                        prim[GRAD_OUTPUT_C], prim[GRAD_GATE_O], bs*hs);

    DotOutput_bprop(prim[GATE_F], prim[GATE_I], prim[GATE_C], input_c,
                prim[GRAD_OUTPUT_C], grad_input_c, prim[GRAD_GATE_F],
                prim[GRAD_GATE_I], prim[GRAD_GATE_C], bs*hs);

    CopySplit_bprop(prim[TEMP_GATE], bs, hs, prim[GRAD_GATE_I],
                    prim[GRAD_GATE_F], prim[GRAD_GATE_O], prim[GRAD_GATE_C]);

    Linear_bprop(input_x, input_h, prim[TEMP_GATE], weight_h, grad_weight_x, grad_weight_h,
                grad_bias_x, grad_bias_h, grad_input_h, bs, xl, hs);
}

//prim, pointer to internal tensors
void THNN_(LSTM_updateOutput)(
      THNNState *state,
      THLongTensor *primitives,
      int initOK,
      THTensor *input_c,
      THTensor *input_h,
      THTensor *input_x,
      THTensor *output_c,
      THTensor *output_h,
      THTensor *weight_x,
      THTensor *weight_h,
      THTensor *bias_x,
      THTensor *bias_h)
{
    long bs = input_x->size[0];   //batch size
    long xl = input_x->size[1];   //input feature length
    long hs = input_h->size[0];   //hidden length
    long hs4 = hs * 4;            //4 * hidden length, weight size

    float* in_c = input_c->storage->data;
    float* in_h = input_h->storage->data;
    float* in_x = input_x->storage->data;
    float* out_c = output_c->storage->data;
    float* out_h = output_h->storage->data;
    float* w_x = weight_x->storage->data;
    float* w_h = weight_h->storage->data;
    float* b_x = bias_x->storage->data;
    float* b_h = bias_h->storage->data;

    float ** prim = (float**)primitives->storage->data;
    void Fprop(prim, in_c, in_h, in_x, out_c, out_h,
                w_x, w_h, b_x, b_h, init_ok);
}

void THNN_(LSTM_updateGradInput)(
      THNNState *state,
      THLongTensor *primitives,
      THTensor *input_c,
      THTensor *input_h,
      THTensor *input_x,
      THTensor *weight_x,
      THTensor *weight_h,
      THTensor *grad_output_h,
      THTensor *grad_output_c,
      THTensor *grad_weight_x,
      THTensor *grad_weight_h,
      THTensor *grad_bias_x,
      THTensor *grad_bias_h,
      THTensor *grad_input_h,
      THTensor *grad_input_c)
{

    long bs = input_x->size[0];   //batch size
    long xl = input_x->size[1];   //input feature length
    long hs = input_h->size[0];   //hidden length
    long hs4 = hs * 4;            //4 * hidden length, weight size

    float* in_c = input_c->storage->data;
    float* in_h = input_h->storage->data;
    float* in_x = input_x->storage->data;
    float* w_x = weight_x->storage->data;
    float* w_h = weight_h->storage->data;
    float* b_x = bias_x->storage->data;
    float* b_h = bias_x->storage->data;
    float* grad_in_c = grad_input_c->storage->data;
    float* grad_in_h = grad_input_h->storage->data;
    float* grad_out_c = grad_output_c->storage->data;
    float* grad_out_h = grad_output_h->storage->data;
    float* grad_w_x = grad_weight_x->storage->data;
    float* grad_w_h = grad_weight_h->storage->data;
    float* grad_b_x = grad_bias_x->storage->data;
    float* grad_b_h = grad_bias_h->storage->data;

    float ** prim = (float**)primitives->storage->data;
    void Bprop(prim, in_c, in_h, in_x, grad_out_h,  grad_out_c,
        w_x, w_h, grad_w_x, grad_w_h, grad_b_x, grad_b_h,
        grad_in_h, grad_in_c, bs, xl, hs);
}

#endif