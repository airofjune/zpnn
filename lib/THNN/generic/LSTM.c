#include "LSTM.h"
#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/LSTM.c"
#else

#define temp real
#undef real
#include "mkl.h"
#define real temp

static float THNN_(Sigmoid)(float a)
{
    return 1.0f / ( 1.0f + expf(-1*a) );
}

static void THNN_(PrintVec)(float* a, const long len)
{
    printf("print vec:");
    for(long i=0; i<len; ++i)
        printf("%f\t", a[i]);
    printf("\n");
}

static void THNN_(Sum)(float* a, const long len)
{
    float sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for(long i=0; i<len; ++i)
        sum += a[i];
    printf("sum is %f\n", sum);
}

static int THNN_(CompareVec)(float*a, float* b, const long len, const float gap)
{
    for(long i=0; i<len; ++i)
    {
        if(abs(a[i]-b[i]) > gap)
        {
            printf("%f\t%f\n", a[i], b[i]);
            THNN_(PrintVec)(a, len);
            THNN_(PrintVec)(b, len);
            return 1;
        }
    }
    return 0;
}

static void THNN_(Linear_fprop)(
    const long bs,
    const long hs,
    const long xl,
    float* temp_bias,
    float* bias_h,
    float* bias_x,
    float* temp_gate,
    float* weight_h,
    float* weight_x,
    float* input_h,
    float* input_x)
{
    //gate = input_x * weight_x  + input_h * weight_h + bias_x + bias_h;
    //add bias for each batch
    const long hs4 = hs*4;
    #pragma omp parallel for
    for(int i=0; i<bs; ++i)
    {
        cblas_scopy(hs4, bias_h, 1, temp_gate+i*hs4, 1);
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bs, hs4, xl, 1, input_x, xl,
                    weight_x, hs4, 1, temp_gate, hs4);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bs, hs4, hs, 1, input_h, hs,
                    weight_h, hs4, 1, temp_gate, hs4);
}

static void THNN_(Linear_bprop)(
    float* input_h,
    float* input_x,
    float* grad_gate,
    float* weight_h,
    float* weight_x,
    float* grad_weight_h,
    float* grad_weight_x,
    float* grad_bias_h,
    float* grad_bias_x,
    float* grad_input_h,
    float* grad_input_x,
    const long bs,
    const long xl,
    const long hs)
{
    //gate = input_x * weight_x  + input_h * weight_h + bias;
    const long hs4 = hs*4;

    //grad_input_h (bs*hs) = grad_gate(bs*hs4) * T(weight_h) (hs4*hs)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, bs, hs, hs4, 1, grad_gate, hs4,
                     weight_h, hs4, 0, grad_input_h, hs);

    //grad_input_x (bs*xl) = grad_gate(bs*hs4) * T(weight_x) (hs4*xl)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, bs, xl, hs4, 1, grad_gate, hs4,
                     weight_x, hs4, 0, grad_input_x, xl);

    //grad_weight_h (hs*hs4) = T(input_h) * grad_gate(bs*hs4)
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, hs, hs4, bs, 1, input_h, hs,
                     grad_gate, hs4, 1, grad_weight_h, hs4);

    //grad_weight_x (xl*hs4) = T(input_x) * grad_gate(bs*hs4)
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, xl, hs4, bs, 1, input_x, xl,
                     grad_gate, hs4, 1, grad_weight_x, hs4);

    //bias
    #pragma omp parallel for
    for(long i=0; i<hs4; ++i)
    {
        for(long j=0; j<bs; ++j)
            grad_bias_h[i] += grad_gate[j*hs4+i];
    }
}

static void THNN_(CopySplit_fprop)(
    float* pIn, const long bs, const long hs,
    float* p1, float* p2, float* p3, float* p4)
{
    #pragma omp parallel for
    for(long i=0; i<bs; ++i)
    {
        const long ihs = i * hs;
        float* pStart = pIn + ihs*4;
        for(long j=0; j<hs; ++j)
        {
	        pStart[j] = THNN_(Sigmoid)(pStart[j]);
            p1[ihs+j] = pStart[j];
        }
        pStart += hs;
        for(long j=0; j<hs; ++j)
        {
	        pStart[j] = THNN_(Sigmoid)(pStart[j]);
            p2[ihs+j] = pStart[j];
        }
        pStart += hs;
        for(long j=0; j<hs; ++j)
        {
	        pStart[j] = THNN_(Sigmoid)(pStart[j]);
            p3[ihs+j] = pStart[j];
        }
        pStart += hs;
        for(long j=0; j<hs; ++j)
        {
	        pStart[j] = tanh(pStart[j]);
            p4[ihs+j] = pStart[j];
        }
    }
}

static void THNN_(CopySplit_bprop)(
    float* pIn, const long bs, const long hs,
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

static void THNN_(TanhDot_fprop)(
    float* output_c,
    float* output_h,
    float* tanC,
    float* gate_f,
    float* gate_i,
    float* gate_c,
    float* gate_o,
    float* input_c,
    const long len)
{
    //output_c = gate_f .* input_c + gate_i .* gate_c
    //c_tanh = tanh(output_c)
    //output_h = gate_o .* c_tanh
    #pragma omp parallel for
    for(long i=0; i<len; ++i)
    {
        output_c[i] = gate_f[i] * input_c[i] + gate_i[i] * gate_c[i];
        tanC[i] = tanh(output_c[i]);
        output_h[i] = tanC[i] * gate_o[i];
    }
}

static void THNN_(TanhDot_bprop)(
    float* grad_out_c,
    float* grad_out_h,
    float* tanhC,
    float* input_c,
    float* gate_i,
    float* gate_f,
    float* gate_o,
    float* gate_c,
    float* grad_gate_i,
    float* grad_gate_f,
    float* grad_gate_o,
    float* grad_gate_c,
    float* grad_input_c,
    const long len)
{
    //output_c = gate_f .* input_c + gate_i .* gate_c
    //grad_input_c = grad_output_c .* gate_f
    //grad_gate_f  = grad_output_c .* input_c
    //grad_gate_i  = grad_output_c .* gate_c
    //grad_gate_c  = grad_output_c .* gate_i
    //c_tanh = tanh(output_c)
    //output_h = gate_o .* c_tanh
    //grad_output_c  = grad_output_c + (1-c_tanh.*c_tanh).*gate_o.*grad_output_h
    //grad_gate_o    = grad_output_h .* c_thanh
    #pragma omp parallel for
    for(long i=0; i<len; ++i)
    {
        const float c = tanhC[i];
        grad_gate_o[i] = grad_out_h[i] * c;
        const float d = grad_out_c[i] + (1 - c*c) * gate_o[i] * grad_out_h[i];
        grad_input_c[i] = d * gate_f[i];
        grad_gate_f[i]  = d * input_c[i];
        grad_gate_i[i]  = d * gate_c[i];
        grad_gate_c[i]  = d * gate_i[i];
    }
}

static int THNN_(InternalMemAlloc)(float** prim, float* pMem, const long bs, const long hs)
{
    const long hs4 = hs * 4;
    prim[TEMP_GATE]    = pMem;
    prim[GATE_I]       = prim[TEMP_GATE] + bs * hs4;
    prim[GATE_F]       = prim[GATE_I]    + bs * hs;
    prim[GATE_C]       = prim[GATE_F]    + bs * hs;
    prim[GATE_O]       = prim[GATE_C]    + bs * hs;
    prim[GRAD_GATE_I]  = prim[GATE_O]    + bs * hs;
    prim[GRAD_GATE_F]  = prim[GRAD_GATE_I] + bs*hs;
    prim[GRAD_GATE_C]  = prim[GRAD_GATE_F] + bs*hs;
    prim[GRAD_GATE_O]  = prim[GRAD_GATE_C] + bs*hs;
    prim[TEMP_BIAS]    = prim[GRAD_GATE_O] + bs*hs;
    prim[C_TANH]       = prim[TEMP_BIAS]   + hs4;
    prim[GRAD_OUTPUT_C]= prim[C_TANH]      + bs*hs;

    return 0;
}


//prim, pointer to internal tensors
static void THNN_(Fprop)(
    float **prim,
    float *input_c,
    float *input_h,
    float *input_x,
    float *output_c,
    float *output_h,
    float *weight_h,
    float *weight_x,
    float *bias_h,
    float *bias_x,
    const long bs,
    const long xl,
    const long hs)
{
    //4 * hidden length, weight size
    const long hs4 = hs * 4;
    THNN_(Linear_fprop)(bs, hs, xl, prim[TEMP_BIAS], bias_h, bias_x, prim[TEMP_GATE],
                weight_h, weight_x, input_h, input_x);

    //sigmoid and split
    //gate_i = sig(gate[:,0])
    //gate_f = sig(gate[:,1])
    //gate_o = sig(gate[:,2])
    //gate_c = tanh(gate[:,3])
    THNN_(CopySplit_fprop)(prim[TEMP_GATE], bs, hs, prim[GATE_I], prim[GATE_F], prim[GATE_O], prim[GATE_C]);

    THNN_(TanhDot_fprop)(output_c, output_h, prim[C_TANH],
           prim[GATE_F], prim[GATE_I], prim[GATE_C], prim[GATE_O],
           input_c, bs*hs);
}

static void THNN_(Bprop)(
    float **prim,
    float *input_c,
    float *input_h,
    float *input_x,
    float *grad_out_c,
    float *grad_out_h,
    float *weight_h,
    float *weight_x,
    float *grad_weight_h,
    float *grad_weight_x,
    float *grad_bias_h,
    float *grad_bias_x,
    float *grad_input_c,
    float *grad_input_h,
    float* grad_input_x,
    const long bs,
    const long xl,
    const long hs)
{
    const long hs4 = hs * 4;
    THNN_(TanhDot_bprop)(grad_out_c, grad_out_h, prim[C_TANH], input_c,
            prim[GATE_I], prim[GATE_F], prim[GATE_O], prim[GATE_C],
            prim[GRAD_GATE_I], prim[GRAD_GATE_F], prim[GRAD_GATE_O], prim[GRAD_GATE_C],
            grad_input_c, bs*hs);

    THNN_(CopySplit_bprop)(prim[TEMP_GATE], bs, hs, prim[GRAD_GATE_I],
                prim[GRAD_GATE_F], prim[GRAD_GATE_O], prim[GRAD_GATE_C]);

    THNN_(Linear_bprop)(input_h, input_x, prim[TEMP_GATE], weight_h, weight_x,
                grad_weight_h, grad_weight_x, grad_bias_h, grad_bias_x,
                grad_input_h, grad_input_x, bs, xl, hs);
}

//prim, pointer to internal tensors
void THNN_(LSTM_updateOutput)(
    THNNState *state,
    THFloatTensor *primitives,
    THTensor *pMem,
    THTensor *input_c,
    THTensor *input_h,
    THTensor *input_x,
    THTensor *output_c,
    THTensor *output_h,
    THTensor *weight,
    THTensor *bias)
{

    long bs = input_x->size[0];   //batch size
    long xl = input_x->size[1];   //input feature length
    long hs = input_h->size[1];   //hidden length
    long hs4 = hs * 4;            //4 * hidden length, weight size

    float* in_c = (float*)THTensor_(data)(input_c);
    float* in_h = (float*)THTensor_(data)(input_h);
    float* in_x = (float*)THTensor_(data)(input_x);
    float* out_c = (float*)THTensor_(data)(output_c);
    float* out_h = (float*)THTensor_(data)(output_h);
    float* w_h = (float*)THTensor_(data)(weight);
    float* w_x = w_h + hs * hs4;
    float* b_h = (float*)THTensor_(data)(bias);
    float* b_x = b_h + hs4;
    float ** prim = (float**)primitives->storage->data + primitives->storageOffset;

    THNN_(InternalMemAlloc)(prim, (float*)THTensor_(data)(pMem), bs, hs);

    THNN_(Fprop)(prim, in_c, in_h, in_x, out_c, out_h,
                 w_h, w_x, b_h, b_x, bs, xl, hs);
}

void THNN_(LSTM_updateGradInput)(
    THNNState *state,
    THFloatTensor *primitives,
    THTensor *input_c,
    THTensor *input_h,
    THTensor *input_x,
    THTensor *weight,
    THTensor *grad_output_c,
    THTensor *grad_output_h,
    THTensor *grad_weight,
    THTensor *grad_bias,
    THTensor *grad_input_c,
    THTensor *grad_input_h,
    THTensor *grad_input_x)
{
    long bs = input_x->size[0];   //batch size
    long xl = input_x->size[1];   //input feature length
    long hs = input_h->size[1];   //hidden length
    long hs4 = hs * 4;            //4 * hidden length, weight size

    float* in_c = (float*)THTensor_(data)(input_c);
    float* in_h = (float*)THTensor_(data)(input_h);
    float* in_x = (float*)THTensor_(data)(input_x);
    float* w_h = (float*)THTensor_(data)(weight);
    float* w_x = w_h + hs * hs4;
    float* grad_in_c = (float*)THTensor_(data)(grad_input_c);
    float* grad_in_h = (float*)THTensor_(data)(grad_input_h);
    float* grad_in_x = (float*)THTensor_(data)(grad_input_x);
    float* grad_out_c = (float*)THTensor_(data)(grad_output_c);
    float* grad_out_h = (float*)THTensor_(data)(grad_output_h);
    float* grad_w_h = (float*)THTensor_(data)(grad_weight);
    float* grad_w_x = grad_w_h + hs * hs4;
    float* grad_b_h = (float*)THTensor_(data)(grad_bias);
    float* grad_b_x = grad_b_h + hs4;

    float ** prim = (float**)(primitives->storage->data + primitives->storageOffset);
    THNN_(Bprop)(prim, in_c, in_h, in_x, grad_out_c, grad_out_h,
        w_h, w_x, grad_w_h, grad_w_x, grad_b_h, grad_b_x,
        grad_in_c, grad_in_h, grad_in_x, bs, xl, hs);
}
#endif
