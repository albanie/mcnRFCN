// @file psroipooling_gpu.cu
// @brief Position Sensitive Region of interest pooling block implementation (GPU)
// @author Samuel Albanie
// @author Hakan Bilen
// @author Abishek Dutta
// @author Andrea Vedaldi
/*
Copyright (C) 2017 Samuel Albanie, Hakan Bilen, Abishek Dutta, and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "psroipooling.hpp"
#include "../datacu.hpp"
#include <assert.h>
#include <stdio.h>
#include <cfloat>
#include <algorithm>
#include <sm_20_atomic_functions.h>


/* ---------------------------------------------------------------- */
/*                                                          Helpers */
/* ---------------------------------------------------------------- */

// Define a struct to hold the dimensions of the grid of the pooling 
// windows (which will be HxW = subdivisions[0] x subdivisions[1]), 
// and the six parameters of the affine transformation to be applied
// to inputs
template<typename T>
struct Geom {
  int subdivisions[2] ;
  T transform[6] ;
  Geom(int const subdivisions[2], double const transform[6])
  {
    this->subdivisions[0] = subdivisions[0] ;
    this->subdivisions[1] = subdivisions[1] ;
    this->transform[0] = transform[0] ;
    this->transform[1] = transform[1] ;
    this->transform[2] = transform[2] ;
    this->transform[3] = transform[3] ;
    this->transform[4] = transform[4] ;
    this->transform[5] = transform[5] ;
  }
} ;

struct Bounds {
  int image, offset, hstart, hend, wstart, wend ;
  bool isEmpty ;
} ;

template<typename T>
__device__ __forceinline__ static Bounds
getBounds(int outputIndex,
          int height, int width, int numChannels, int size,
          const T* rois, int numROIs, int outChannels,
          Geom<T> geom)
{
  Bounds b ;

  int ph = outputIndex ; // height is fastest moving index
  int pw = ph / geom.subdivisions[0] ;
  int pc = pw / geom.subdivisions[1] ;
  int pr = pc / outChannels ; 

  ph %= geom.subdivisions[0] ;
  pw %= geom.subdivisions[1] ;
  pc %= outChannels ;

  rois += 5 * pr ;

  // Apply scale and offset to each ROI coordinate.
  T u1_ = rois[1] ;
  T v1_ = rois[2] ;
  T u2_ = rois[3] ;
  T v2_ = rois[4] ;

  int roi_image = (int)rois[0];

  // TODO(sam): The official position sensitive roi pooling handles rounding 
  // slightly differently to the original version released with Faster R-CNN
  // To provide numerical equivalency, a slightly different bound implementation
  // has been added below.  This does not yet support the general affine 
  // transformation of input points provided by the original. 
  // The implementations should be integrated in future.
  bool NUMERICAL_MATCH = 1 ;

  if (NUMERICAL_MATCH) {

    // Modified bound implementation 
    T roi_start_w = static_cast<T>(round(u1_-1)) * geom.transform[0] ;
    T roi_start_h = static_cast<T>(round(v1_-1)) * geom.transform[3] ;
    T roi_end_w = static_cast<T>(round(u2_)) * geom.transform[0] ; 
    T roi_end_h = static_cast<T>(round(v2_)) * geom.transform[3] ; 

    // prevent rois from having a height or width of zero
    T roi_height = max(roi_end_h - roi_start_h, 0.1) ;
    T roi_width = max(roi_end_w - roi_start_w, 0.1) ;
    T bin_size_h = roi_height / static_cast<T>(geom.subdivisions[0]) ;
    T bin_size_w = roi_width / static_cast<T>(geom.subdivisions[1]) ;

    roi_image = min(max(roi_image - 1,0), (int)size - 1) ;

    // select position sensitive feature map by applying additional offset
    int c = (pc * geom.subdivisions[0] + ph) * geom.subdivisions[1] + pw ;
    b.offset = (roi_image * numChannels + c) * (width*height) ;

    b.hstart = (int)floor(((T)ph) * bin_size_h + roi_start_h) ;
    b.wstart = (int)floor(((T)pw) * bin_size_w + roi_start_w) ;
    b.hend = (int)ceil(((T)(ph + 1)) * bin_size_h + roi_start_h) ;
    b.wend = (int)ceil(((T)(pw + 1)) * bin_size_w + roi_start_w) ;

    // clip to boundaries of input features - ignore issue of offset
    b.wstart = min(max(b.wstart, 0), (int)width) ;
    b.wend = min(max(b.wend, 0), (int)width) ;
    b.hstart = min(max(b.hstart, 0), (int)height) ;
    b.hend = min(max(b.hend, 0), (int)height) ;

  } else {
    // Conform to original RoiPooling conventions
    T u1 = geom.transform[0] * u1_ + geom.transform[2] * v1_ + geom.transform[4] ;
    T v1 = geom.transform[1] * u1_ + geom.transform[3] * v1_ + geom.transform[5] ;
    T u2 = geom.transform[0] * u2_ + geom.transform[2] * v2_ + geom.transform[4] ;
    T v2 = geom.transform[1] * u2_ + geom.transform[3] * v2_ + geom.transform[5] ;

    // First and last pixel of each ROI (rounded for compatibility with 
    // the Caffe definition).
    int roi_start_h = (int)round(v1) - 1 ;
    int roi_start_w = (int)round(u1) - 1 ;
    int roi_end_h = (int)round(v2) - 1 ;
    int roi_end_w = (int)round(u2) - 1 ;

    // prevent rois from having a height or width of zero
    int roi_height = max(roi_end_h - roi_start_h + 1, 1) ;
    int roi_width = max(roi_end_w - roi_start_w + 1, 1) ;

    T bin_size_h = (T)roi_height / geom.subdivisions[0] ;
    T bin_size_w = (T)roi_width / geom.subdivisions[1] ;

    roi_image = min(max(roi_image - 1,0), (int)size - 1) ;

    // select position sensitive feature map by applying additional offset
    int c = (pc * geom.subdivisions[1] + ph) * geom.subdivisions[0] + pw ;
    b.offset = (roi_image * numChannels + c) * (width*height) ;

    b.wstart = (int)floor(((T)pw) * bin_size_w) ;
    b.wend = (int)ceil(((T)(pw + 1)) * bin_size_w) ;
    b.hstart = (int)floor(((T)ph) * bin_size_h) ;
    b.hend = (int)ceil(((T)(ph + 1)) * bin_size_h) ;

    b.wstart = min(max(b.wstart + roi_start_w, 0), (int)width) ;
    b.wend = min(max(b.wend + roi_start_w, 0), (int)width) ;
    b.hstart = min(max(b.hstart + roi_start_h, 0), (int)height) ;
    b.hend = min(max(b.hend + roi_start_h, 0), (int)height) ;
  }
  b.isEmpty = (b.hend <= b.hstart) || (b.wend <= b.wstart) ;
  return b ;
}

/* ---------------------------------------------------------------- */
/*                                     psroipooling_average_forward */
/* ---------------------------------------------------------------- */

template<typename T> __global__ void
psroipooling_average_kernel
(T* output,
 const T* data, int height, int width, int numChannels, int size,
 const T* rois, int numROIs, int outChannels,
 Geom<T> geom)
{
  int outputIndex = threadIdx.x + blockIdx.x * blockDim.x;
  int outputVolume = geom.subdivisions[0] * geom.subdivisions[1] * outChannels * numROIs;
  if (outputIndex < outputVolume) {
    Bounds b = getBounds<T>(outputIndex,
                            height, width, numChannels, size,
                            rois, numROIs, outChannels,
                            geom) ;
    data += b.offset ;
    T bestValue = 0;
    const T coeff = ((T)1.) / (T)((b.wend-b.wstart) * (b.hend-b.hstart));
    for (int w = b.wstart; w < b.wend; ++w) {
      for (int h = b.hstart; h < b.hend; ++h) {
        int index = w * height + h ;
        bestValue += data[index] * coeff ;
      }
    }
    output[outputIndex] = bestValue ;
    if (b.offset==2161278) {
      if (b.hstart==32&&b.hend==34&&b.wstart==38&&b.wend==40) {
          printf("GPU::outputIndex: (%d)\n", outputIndex) ;
          printf("GPU:: (hstart,hend,wstart,wend): (%d,%d,%d,%d)\n", 
                                  b.hstart,b.hend,b.wstart,b.wend) ;
      }
    }
  }
}

/* ---------------------------------------------------------------- */
/*                                         psroipooling_max_forward */
/* ---------------------------------------------------------------- */

template<typename T> __global__ void
psroipooling_max_kernel
(T* output,
 const T* data, int height, int width, int numChannels, int size,
 const T* rois, int numROIs, int outChannels,
 Geom<T> geom)
{
  int outputIndex = threadIdx.x + blockIdx.x * blockDim.x ;
  int outputVolume = geom.subdivisions[0] * geom.subdivisions[1] * numChannels * numROIs ;
  if (outputIndex < outputVolume) {
    Bounds b = getBounds<T>(outputIndex,
                            height, width, numChannels, size,
                            rois, numROIs, outChannels, 
                            geom) ;
    data += b.offset ;
    if (! b.isEmpty) {
      T bestValue = -FLT_MAX;
      for (int w = b.wstart; w < b.wend; ++w) {
        for (int h = b.hstart; h < b.hend; ++h) {
          int index = w * height + h ;
          bestValue = max(bestValue, data[index]) ;
        }
      }
      output[outputIndex] = bestValue ;
    } else {
      output[outputIndex] = 0 ;
    }
  }
}

/* ---------------------------------------------------------------- */
/*                                                        atomicAdd */
/* ---------------------------------------------------------------- */

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
// an implementation of atomicAdd() for double (really slow)
static __device__ double atomicAdd(double* address, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                                         __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

/* ---------------------------------------------------------------- */
/*                                    psroipooling_average_backward */
/* ---------------------------------------------------------------- */

template<typename T> __global__ void
psroipooling_average_backward_kernel
(T* derData,
 const T* data, int height, int width, int numChannels, int size,
 const T* rois, int numROIs, int outChannels,
 const T* derOutput,
 Geom<T> geom)
{
  int outputIndex = threadIdx.x + blockIdx.x * blockDim.x;
  int outputVolume = geom.subdivisions[0] * geom.subdivisions[1] * numChannels * numROIs ;
  if (outputIndex < outputVolume) {

    Bounds b = getBounds<T>(outputIndex,
                            height, width, numChannels, size,
                            rois, numROIs, outChannels,
                            geom) ;
    data += b.offset ;
    derData += b.offset ;
    const T coeff = ((T)1.) / (T)((b.wend-b.wstart)*(b.hend-b.hstart)) ;
    // TODO(sam): Fix for channel placement
    for (int h = b.hstart; h < b.hend; ++h) {
      for (int w = b.wstart; w < b.wend; ++w) {
        int index = w * height + h ;
        atomicAdd(derData + index, derOutput[outputIndex] * coeff) ;
      }
    }
  }
}

/* ---------------------------------------------------------------- */
/*                                        psroipooling_max_backward */
/* ---------------------------------------------------------------- */

template<typename T> __global__ void
psroipooling_max_backward_kernel
(T* derData,
 const T* data, int height, int width, int numChannels, int size,
 const T* rois, int numROIs, int outChannels,
 const T* derOutput,
 Geom<T> geom)
{
  int outputIndex = threadIdx.x + blockIdx.x * blockDim.x;
  int outputVolume = geom.subdivisions[0] * geom.subdivisions[1] * numChannels * numROIs;

  if (outputIndex < outputVolume) {

    Bounds b = getBounds<T>(outputIndex,
                            height, width, numChannels, size,
                            rois, numROIs, outChannels,
                            geom) ;
    // TODO(sam): Fix for channel placement
    if (! b.isEmpty) {
      data += b.offset ;
      derData += b.offset ;
      int bestIndex = min(b.wstart,width-1) * height + min(b.hstart,height-1);
      T bestValue = -FLT_MAX;
      for (int h = b.hstart; h < b.hend; ++h) {
        for (int w = b.wstart; w < b.wend; ++w) {
          int index = w * height + h ;
          T value = data[index] ;
          if (value > bestValue) {
            bestValue = value ;
            bestIndex = index ;
          }
        }
      }
      atomicAdd(derData + bestIndex, derOutput[outputIndex]) ;
    }
  }
}

/* ---------------------------------------------------------------- */
/*                                                        Interface */
/* ---------------------------------------------------------------- */

namespace vl { namespace impl {

  template <typename type>
  struct psroipooling_max<vl::VLDT_GPU, type>
  {
    static vl::ErrorCode
    forward(type* output,
            type const* data,
            size_t height, size_t width, size_t numChannels, size_t size,
            type const* rois,
            size_t numROIs,
            int const subdivisions[2],
            double const transform[6], 
            int const outChannels)
    {
      int outputVolume = subdivisions[0] * subdivisions[1] * outChannels * numROIs ;

      psroipooling_max_kernel<type>
      <<< divideAndRoundUp(outputVolume, VL_CUDA_NUM_THREADS),VL_CUDA_NUM_THREADS >>>
      (output,
       data, height, width, numChannels, size,
       rois, numROIs, outChannels,
       Geom<type>(subdivisions,transform)) ;

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }

    static vl::ErrorCode
    backward(type* derData,
             type const* data,
             size_t height, size_t width, size_t numChannels, size_t size,
             type const* rois,
             size_t numROIs,
             type const* derOutput,
             int const subdivisions[2],
             double const transform[6],
             int const outChannels)
    {
      int outputVolume = subdivisions[0] * subdivisions[1] * outChannels * numROIs ;

      psroipooling_max_backward_kernel<type>
      <<< divideAndRoundUp(outputVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (derData, data,
       height, width, numChannels, size,
       rois, numROIs, outChannels,
       derOutput,
       Geom<type>(subdivisions,transform)) ;

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }
  } ; // psroipooling_max

  template <typename type>
  struct psroipooling_average<vl::VLDT_GPU, type>
  {
    static vl::ErrorCode
    forward(type* output,
            type const* data,
            size_t height, size_t width, size_t numChannels, size_t size,
            type const* rois,
            size_t numROIs,
            int const subdivisions[2],
            double const transform[6],
            int const outChannels)
    {
      int outputVolume = subdivisions[0] * subdivisions[1] * outChannels * numROIs ;

      psroipooling_average_kernel<type>
      <<< divideAndRoundUp(outputVolume, VL_CUDA_NUM_THREADS), 
                                         VL_CUDA_NUM_THREADS >>>
      (output, data,
       height, width, numChannels, size,
       rois, numROIs, outChannels, 
       Geom<type>(subdivisions,transform)) ;

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }

    static vl::ErrorCode
    backward(type* derData,
             type const* data, // <- this is not needed for avg pooling
             size_t height, size_t width, size_t numChannels, size_t size,
             type const* rois,
             size_t numROIs,
             type const* derOutput,
             int const subdivisions[2],
             double const transform[6],
             int const outChannels)
    {
      int outputVolume = subdivisions[0] * subdivisions[1] * outChannels * numROIs ;

      psroipooling_average_backward_kernel<type>
      <<< divideAndRoundUp(outputVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (derData, data,
       height, width, numChannels, size,
       rois, numROIs, outChannels,
       derOutput,
       Geom<type>(subdivisions,transform)) ;

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }
  } ; // psroipooling_average
} } ; // namespace vl::impl

// Instantiations
template struct vl::impl::psroipooling_max<vl::VLDT_GPU, float> ;
template struct vl::impl::psroipooling_average<vl::VLDT_GPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::psroipooling_max<vl::VLDT_GPU, double> ;
template struct vl::impl::psroipooling_average<vl::VLDT_GPU, double> ;
#endif
