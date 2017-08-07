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


inline void THNN_(copy_avx_512)(float* src, float* dst, const long len)
{
    long j;
    for (j=0; j<=((len)-16); j+=16) {
        _mm512_storeu_ps(dst+j, _mm512_loadu_ps(src+j));
    }
    long off = (len) - ((len)%16);
    for (j=off; j<len; ++j) {
        dst[j] = src[j];
    }
}


static void THNN_(Linear_fprop)(const long bs, const long hs, const long xl,
            float * temp_bias, float* bias_h, float* bias_x,
            float* temp_gate,  float* weight_h, float* weight_x,
             float* input_h, float *input_x)
{
    //gate = input_x * weight_x  + input_h * weight_h + bias_x + bias_h;
    //add bias for each batch
    const long hs4 = hs*4;
    #pragma omp parallel for
    for(int i=0; i<bs; ++i)
    {
        cblas_scopy(hs4, bias_h, 1, temp_gate+i*hs4, 1);
    }

#if LOG_ON
    struct Timer gemm;
    Start(&gemm);
#endif

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bs, hs4, xl, 1, input_x, xl,
                    weight_x, hs4, 1, temp_gate, hs4);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bs, hs4, hs, 1, input_h, hs,
                    weight_h, hs4, 1, temp_gate, hs4);
#if LOG_ON
    prof.gemm += End(&gemm);
#endif
}

static void THNN_(Linear_bprop)(float* input_h, float* input_x, float* grad_gate,
            float* weight_h, float* weight_x,  float* grad_weight_h, float* grad_weight_x,
            float* grad_bias_h, float* grad_bias_x, float* grad_input_h,  float* grad_input_x,
            const long bs, const long xl, const long hs)
{
    //gate = input_x * weight_x  + input_h * weight_h + bias;
    const long hs4 = hs*4;
#if LOG_ON
    struct Timer gemm;
    Start(&gemm);
#endif
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
#if LOG_ON
    prof.gemm += End(&gemm);
#endif
    //bias
    #pragma omp parallel for
    for(long i=0; i<hs4; ++i)
    {
        for(long j=0; j<bs; ++j)
            grad_bias_h[i] += grad_gate[j*hs4+i];
    }
}

static void THNN_(GateSigmoid_fprop)(float* gateValue, const long bs, const long hs)
{
#if LOG_ON
    struct Timer sig;
    Start(&sig);
#endif
    const long hs4 = hs * 4;
    #pragma omp parallel for 
    for(long i=0; i<bs; ++i)
    {
        const long ihs4 = i*hs4;
        long j = 0;
        for( ; j<3*hs; ++j)
        {
            gateValue[ihs4 + j] = THNN_(Sigmoid)(gateValue[ihs4 + j]);
        }
        for( ; j<hs4; ++j)
        {
            gateValue[ihs4 + j] = tanh(gateValue[ihs4 + j]);
	}
    }
#if LOG_ON
    prof.sig += End(&sig);
#endif
}

static void THNN_(CopySplit_fprop)(float* pIn, const long bs, const long hs,
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

static void THNN_(CopySplit_bprop)(float* pIn, const long bs, const long hs,
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

static void THNN_(DotMul)(float* z, const long len, const float* x, const float*y, int add)
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

static void THNN_(DotOutput_fprop)(float* output_c, const long len, float* gate_f,
                    float* gate_i, float* gate_c, float* input_c)
{
    //output_c = gate_f .* input_c + gate_i .* gate_c
    #pragma omp parallel for
    for(long i=0; i<len; ++i)
        output_c[i] = gate_f[i] * input_c[i] + gate_i[i] * gate_c[i];
    
}

static void THNN_(DotOutput_bprop)(float* gate_f,float* gate_i, float* gate_c, float* input_c,
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

static void THNN_(TanhDot_fprop)(float* out, float* tanC, const float* in, const float* dotIn, const long len)
{
    //c_tanh = tanh(output_c)
    //output_h = gate_o .* c_tanh
    #pragma omp parallel for
    for(long i=0; i<len; ++i)
    {
        tanC[i] = tanh(in[i]);
        out[i] = tanC[i] * dotIn[i];
    }
}

static void THNN_(TanhDot_bprop)(float* grad_out_c0, float* grad_out_h, float* tanhC,
    float* gate_o, float* grad_out_c1, float* grad_gate_o, const long len)
{
    //c_tanh = tanh(output_c)
    //output_h = gate_o .* c_tanh
    //grad_output_c  = grad_output_c + (1-c_tanh.*c_tanh).*gate_o.*grad_output_h
    //grad_gate_o    = grad_output_h .* c_thanh
    //cblas_scopy(len, grad_out_c0, 1, grad_out_c1, 1);
    #pragma omp parallel for
    for(long i=0; i<len; ++i)
    {
        const float c = tanhC[i];
        grad_gate_o[i] = grad_out_h[i] * c;
        grad_out_c1[i] = grad_out_c0[i] + (1 - c*c) * gate_o[i] * grad_out_h[i];
    }
}

static int THNN_(InternalMemAlloc)(float** prim, const long bs, const long hs)
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
    prim[GRAD_OUTPUT_C]     = (float*)malloc(sizeof(float)*bs*hs);
    prim[GRAD_TEMP_GATE]    = (float*)malloc(sizeof(float)*bs*hs4);

    return 0;
}

//prim, pointer to internal tensors
static void THNN_(Fprop)(
      float **prim,
      float *input_c,  float *input_h, float *input_x,
      float *output_c, float *output_h,
      float *weight_h, float *weight_x,
      float *bias_h, float *bias_x,
      const long bs, const long xl, const long hs,
      int init_ok)
{
    const long hs4 = hs * 4;            //4 * hidden length, weight size
    if (!init_ok)
    {
        THNN_(InternalMemAlloc)(prim, 64, hs);
    }

#if LOG_ON
    struct Timer t;
    Start(&t);
#endif

    THNN_(Linear_fprop)(bs, hs, xl, prim[TEMP_BIAS], bias_h, bias_x, prim[TEMP_GATE],
                weight_h, weight_x, input_h, input_x);

#if LOG_ON
    prof.lin_f += End(&t);
    Start(&t);
#endif
    //sigmoid and split
    //gate_i = sig(gate[:,0])
    //gate_f = sig(gate[:,1])
    //gate_o = sig(gate[:,2])
    //gate_c = tanh(gate[:,3])
    //THNN_(GateSigmoid_fprop)(prim[TEMP_GATE], bs, hs);
#if LOG_ON
    prof.gate_sig_f += End(&t);
    Start(&t);
#endif
    THNN_(CopySplit_fprop)(prim[TEMP_GATE], bs, hs, prim[GATE_I], prim[GATE_F], prim[GATE_O], prim[GATE_C]);
#if LOG_ON
    prof.copy_split_f += End(&t);
    Start(&t);
#endif
    THNN_(DotOutput_fprop)(output_c, bs*hs, prim[GATE_F],
                    prim[GATE_I], prim[GATE_C], input_c);
#if LOG_ON
    prof.dot_out_f += End(&t);
    Start(&t);
#endif
    THNN_(TanhDot_fprop)(output_h, prim[C_TANH], output_c, prim[GATE_O], bs*hs);
#if LOG_ON
    prof.tanh_dot_f += End(&t);
#endif
}

static void THNN_(Bprop)(
      float **prim,
      float *input_c, float *input_h, float *input_x,
      float *grad_out_c, float *grad_out_h,
      float *weight_h, float *weight_x,
      float *grad_weight_h, float *grad_weight_x,
      float *grad_bias_h, float *grad_bias_x,
      float *grad_input_c, float *grad_input_h, float* grad_input_x,
      const long bs, const long xl, const long hs)
{
    const long hs4 = hs * 4;
#if LOG_ON
    struct Timer t;
    Start(&t);
#endif
    THNN_(TanhDot_bprop)(grad_out_c, grad_out_h, prim[C_TANH], prim[GATE_O],
                prim[GRAD_OUTPUT_C], prim[GRAD_GATE_O], bs*hs);
#if LOG_ON
    prof.tanh_dot_b += End(&t);
    Start(&t);
#endif
    THNN_(DotOutput_bprop)(prim[GATE_F], prim[GATE_I], prim[GATE_C], input_c,
                prim[GRAD_OUTPUT_C], grad_input_c, prim[GRAD_GATE_F],
                prim[GRAD_GATE_I], prim[GRAD_GATE_C], bs*hs);
#if LOG_ON
    prof.dot_out_b += End(&t);
    Start(&t);
#endif
    THNN_(CopySplit_bprop)(prim[TEMP_GATE], bs, hs, prim[GRAD_GATE_I],
                prim[GRAD_GATE_F], prim[GRAD_GATE_O], prim[GRAD_GATE_C]);
#if LOG_ON
    prof.copy_split_b += End(&t);
    Start(&t);
#endif
    THNN_(Linear_bprop)(input_h, input_x, prim[TEMP_GATE], weight_h, weight_x,
                grad_weight_h, grad_weight_x, grad_bias_h, grad_bias_x,
                grad_input_h, grad_input_x, bs, xl, hs);
#if LOG_ON
    prof.lin_b += End(&t);
    Start(&t);
#endif
}

//prim, pointer to internal tensors
void THNN_(LSTM_updateOutput)(
      THNNState *state,
      THFloatTensor *primitives,
      int init_ok,
      THTensor *input_c,
      THTensor *input_h,
      THTensor *input_x,
      THTensor *output_c,
      THTensor *output_h,
      THTensor *weight_h,
      THTensor *weight_x,
      THTensor *bias_h,
      THTensor *bias_x)
{
#if LOG_ON
    struct Timer sum;
    Start(&sum);
#endif

    long bs = input_x->size[0];   //batch size
    long xl = input_x->size[1];   //input feature length
    long hs = input_h->size[1];   //hidden length
    long hs4 = hs * 4;            //4 * hidden length, weight size

    float* in_c = (float*)THTensor_(data)(input_c);
    float* in_h = (float*)THTensor_(data)(input_h);
    float* in_x = (float*)THTensor_(data)(input_x);
    float* out_c = (float*)THTensor_(data)(output_c);
    float* out_h = (float*)THTensor_(data)(output_h);
    float* w_x = (float*)THTensor_(data)(weight_x);
    float* w_h = (float*)THTensor_(data)(weight_h);
    float* b_x = (float*)THTensor_(data)(bias_x);
    float* b_h = (float*)THTensor_(data)(bias_h);
    float ** prim = (float**)primitives->storage->data;
    THNN_(Fprop)(prim, in_c, in_h, in_x, out_c, out_h,
                w_h, w_x, b_h, b_x, bs, xl, hs, init_ok);
#if LOG_ON
    prof.sum += End(&sum);
#endif
}

void THNN_(LSTM_updateGradInput)(
      THNNState *state,
      THFloatTensor *primitives,
      THTensor *input_c,
      THTensor *input_h,
      THTensor *input_x,
      THTensor *weight_h,
      THTensor *weight_x,
      THTensor *grad_output_c,
      THTensor *grad_output_h,
      THTensor *grad_weight_h,
      THTensor *grad_weight_x,
      THTensor *grad_bias_h,
      THTensor *grad_bias_x,
      THTensor *grad_input_c,
      THTensor *grad_input_h,
      THTensor *grad_input_x)
{

#if LOG_ON
    struct Timer sum;
    Start(&sum);
#endif

    long bs = input_x->size[0];   //batch size
    long xl = input_x->size[1];   //input feature length
    long hs = input_h->size[1];   //hidden length
    long hs4 = hs * 4;            //4 * hidden length, weight size

    float* in_c = (float*)THTensor_(data)(input_c);
    float* in_h = (float*)THTensor_(data)(input_h);
    float* in_x = (float*)THTensor_(data)(input_x);
    float* w_x = (float*)THTensor_(data)(weight_x);
    float* w_h = (float*)THTensor_(data)(weight_h);
    float* grad_in_c = (float*)THTensor_(data)(grad_input_c);
    float* grad_in_h = (float*)THTensor_(data)(grad_input_h);
    float* grad_in_x = (float*)THTensor_(data)(grad_input_x);
    float* grad_out_c = (float*)THTensor_(data)(grad_output_c);
    float* grad_out_h = (float*)THTensor_(data)(grad_output_h);
    float* grad_w_x = (float*)THTensor_(data)(grad_weight_x);
    float* grad_w_h = (float*)THTensor_(data)(grad_weight_h);
    float* grad_b_x = (float*)THTensor_(data)(grad_bias_x);
    float* grad_b_h = (float*)THTensor_(data)(grad_bias_h);

    float ** prim = (float**)primitives->storage->data;
    THNN_(Bprop)(prim, in_c, in_h, in_x, grad_out_c, grad_out_h,
        w_h, w_x, grad_w_h, grad_w_x, grad_b_h, grad_b_x,
        grad_in_c, grad_in_h, grad_in_x, bs, xl, hs);
#if LOG_ON
    prof.sum += End(&sum);
#endif
}

void THNN_(LSTM_profile)(THNNState *state)
{
    struct Profiler* a = &prof;
    printf("------------- profile result -----------\n");
    printf("sum  time  is %f\n", a->sum);
    if(a->sum<0.0000001f)
    {
        printf("please set LOG_ON 1 to profile LSTM\n");
        printf("------------- profile done -----------\n");
        return;
    }
    printf("gemm time  is %f\t%f%\n", a->gemm, a->gemm/a->sum*100);
    printf("sig  time  is %f\t%f%\n", a->tanh, a->tanh/a->sum*100);
    printf("--------------------\n");
    printf("gate_sig_f is %f\t%f%\n", a->gate_sig_f, a->gate_sig_f/a->sum*100);
    printf("lin_f      is %f\t%f%\n", a->lin_f, a->lin_f/a->sum*100);
    printf("lin_b      is %f\t%f%\n", a->lin_b, a->lin_b/a->sum*100);
    printf("tanh_dot_f is %f\t%f%\n", a->tanh_dot_f, a->tanh_dot_f/a->sum*100);
    printf("tanh_dot_b is %f\t%f%\n", a->tanh_dot_b, a->tanh_dot_b/a->sum*100);
    printf("split_f    is %f\t%f%\n", a->copy_split_f, a->copy_split_f/a->sum*100);
    printf("split_b    is %f\t%f%\n", a->copy_split_b, a->copy_split_b/a->sum*100);
    printf("dot_out_f  is %f\t%f%\n", a->dot_out_f, a->dot_out_f/a->sum*100);
    printf("dot_out_b  is %f\t%f%\n", a->dot_out_b, a->dot_out_b/a->sum*100);
    float sum = a->gate_sig_f + a->tanh_dot_b + a->tanh_dot_f + a->lin_b + a->lin_f
              + a->copy_split_b + a->copy_split_f + a->dot_out_f + a->dot_out_b;
    printf("sum all is %f%\n", sum/a->sum*100);
    printf("------------- profile done -----------\n");

}

void THNN_(LSTM_resetProfile)(THNNState *state)
{
    prof.gemm = 0.0f;
    prof.tanh = 0.0f;
    prof.sig = 0.0f;
    prof.lin_f = 0.0f;
    prof.lin_b = 0.0f;
    prof.gate_sig_f = 0.0f;
    prof.tanh_dot_f = 0.0f;
    prof.tanh_dot_b = 0.0f;
    prof.copy_split_f = 0.0f;
    prof.copy_split_b = 0.0f;
    prof.dot_out_f = 0.0f;
    prof.dot_out_b = 0.0f;
    prof.sum = 0.0f;

}
#endif
