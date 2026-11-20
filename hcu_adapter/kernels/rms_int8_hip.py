# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native RMS/BF16-storage/dynamic-INT8 fusion for admitted gfx938 W4A8 calls.

The reduction and quantization sequence follow vLLM layernorm_kernels.cu and
quantization/w8a8/int8/scaled_quant.cu (Apache-2.0). The adapter owns this kernel; native weight loading and selected GEMM stay
unchanged. The model prepares the extension before any graph capture.
"""
from functools import lru_cache

from torch.utils.cpp_extension import load_inline


SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <hipcub/hipcub.hpp>
#include <cmath>
#include <tuple>

using BF16 = c10::BFloat16;
struct alignas(16) BF16x8 { BF16 value[8]; };

__global__ __launch_bounds__(1024) void exact_rms_int8_kernel(
    const BF16* __restrict__ x, const BF16* __restrict__ w,
    BF16* __restrict__ y, int8_t* __restrict__ q,
    float* __restrict__ scales, float eps, int hidden) {
    const int offset=blockIdx.x*hidden;
    const int col=threadIdx.x*8;
    BF16x8 src=*reinterpret_cast<const BF16x8*>(x+offset+col);
    BF16x8 weight=*reinterpret_cast<const BF16x8*>(w+col);
    float sum=0.f;
    #pragma unroll
    for(int j=0;j<8;++j) {
        float v=float(src.value[j]);
        sum+=v*v;
    }
    using Reduce=hipcub::BlockReduce<float,1024>;
    __shared__ typename Reduce::TempStorage sum_temp;
    __shared__ typename Reduce::TempStorage max_temp;
    __shared__ float inverse_rms;
    __shared__ float absmax;
    sum=Reduce(sum_temp).Reduce(sum,hipcub::Sum{},blockDim.x);
    if(threadIdx.x==0)inverse_rms=rsqrtf(sum/hidden+eps);
    __syncthreads();
    BF16x8 rounded;
    float local_max=0.f;
    #pragma unroll
    for(int j=0;j<8;++j) {
        // Keep the native BF16 tensor boundary before INT8 range reduction.
        rounded.value[j]=BF16(float(src.value[j])*inverse_rms*float(weight.value[j]));
        local_max=fmaxf(local_max,fabsf(float(rounded.value[j])));
    }
    *reinterpret_cast<BF16x8*>(y+offset+col)=rounded;
    float maximum=Reduce(max_temp).Reduce(local_max,hipcub::Max{},blockDim.x);
    if(threadIdx.x==0) {
        absmax=maximum;
        scales[blockIdx.x]=maximum/127.f;
    }
    __syncthreads();
    const float inv_scale=absmax==0.f?0.f:127.f/absmax;
    #pragma unroll
    for(int j=0;j<8;++j) {
        float v=std::nearbyint(float(rounded.value[j])*inv_scale);
        v=(v < -128.f)?-128.f:((v > 127.f)?127.f:v);
        q[offset+col+j]=static_cast<int8_t>(v);
    }
}

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor>
apply(torch::Tensor x,torch::Tensor weight,double epsilon) {
    TORCH_CHECK(x.is_cuda() && weight.device()==x.device());
    TORCH_CHECK(x.scalar_type()==torch::kBFloat16 && weight.scalar_type()==torch::kBFloat16);
    TORCH_CHECK(x.dim()==2 && x.size(1)==5120 && x.size(0)>0 && x.size(0)<=64);
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && weight.sizes()==torch::IntArrayRef({5120}));
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0);
    const at::cuda::OptionalCUDAGuard guard(device_of(x));
    auto y=torch::empty_like(x);
    auto q=torch::empty_like(x,x.options().dtype(torch::kInt8));
    auto scales=torch::empty({x.size(0),1},x.options().dtype(torch::kFloat32));
    exact_rms_int8_kernel<<<x.size(0),640,0,at::cuda::getCurrentCUDAStream().stream()>>>(
        x.data_ptr<BF16>(),weight.data_ptr<BF16>(),y.data_ptr<BF16>(),
        q.data_ptr<int8_t>(),scales.data_ptr<float>(),float(epsilon),x.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {y,q,scales};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("apply",&apply);}
'''


@lru_cache(maxsize=1)
def module():
    return load_inline(name='hcu_rms_int8_5120_v1', cpp_sources='',
                       cuda_sources=SOURCE, extra_cflags=['-O3'],
                       extra_cuda_cflags=['-O3'], verbose=True)
