// @file nnpsroipooling.cu
// @brief psroipooling block
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


#include "nnpsroipooling.hpp"
#include "impl/psroipooling.hpp"

#if ENABLE_GPU
#include <bits/datacu.hpp>
#endif

#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                           nnpsroipooling_forward */
/* ---------------------------------------------------------------- */

#define DISPATCH(deviceType, op, type) \
status = vl::impl::op<deviceType, type>::forward \
((type*)output.getMemory(), \
(type const*)data.getMemory(), \
data.getHeight(), data.getWidth(), \
data.getDepth(), data.getSize(), \
(type const *)rois.getMemory(), \
rois.getNumElements() / 5, \
subdivisions, transform, outChannels) ;

#define DISPATCH2(deviceType, op) \
switch (dataType) { \
case VLDT_Float : DISPATCH(deviceType, op, float) ; break ; \
IF_DOUBLE(case VLDT_Double : DISPATCH(deviceType, op, double) ; break ;) \
default: assert(false) ; return VLE_Unknown ; \
}

#define DISPATCH3(deviceType) \
switch (method) { \
case vlPSROIPoolingAverage : DISPATCH2(deviceType, psroipooling_average) ; break ; \
case vlPSROIPoolingMax : DISPATCH2(deviceType, psroipooling_max) ; break ; \
default: assert(false) ; return VLE_Unknown ; \
}

vl::ErrorCode
vl::nnpsroipooling_forward(vl::Context& context,
                         vl::Tensor output,
                         vl::Tensor data,
                         vl::Tensor rois,
                         PSROIPoolingMethod method,
                         int const subdivisions[2],
                         double const transform[6],
                         int const outChannels)
{
  vl::ErrorCode status = VLE_Success ;
  vl::DeviceType deviceType = output.getDeviceType() ;
  vl::DataType dataType = output.getDataType() ;
  switch (deviceType) {
    default:
      assert(false) ;
      return vl::VLE_Unknown ;

    case vl::VLDT_CPU:
      DISPATCH3(vl::VLDT_CPU) ;
      break ;

#ifdef ENABLE_GPU
    case vl::VLDT_GPU:
      DISPATCH3(vl::VLDT_GPU) ;
      if (status == vl::VLE_Cuda) {
        context.setError(context.getCudaHelper().catchCudaError(__func__)) ;
      }
      break ;
#endif
  }
  return context.passError(status, "nnpsroipooling_forward") ;
}

/* ---------------------------------------------------------------- */
/*                                          nnpsroipooling_backward */
/* ---------------------------------------------------------------- */

#undef DISPATCH
#undef DISPATCH2

// backward max and average want slightly different argument lists
#define DISPATCH_psroipooling_average(deviceType, type) \
status = vl::impl::psroipooling_average<deviceType, type>::backward \
((type*)derData.getMemory(), \
(type const*)data.getMemory(), \
derData.getHeight(), \
derData.getWidth(), \
derData.getDepth(), \
derData.getSize(), \
(const type *)rois.getMemory(), \
rois.getNumElements() / 5, \
(type const*)derOutput.getMemory(), \
subdivisions, transform, outChannels) ;

#define DISPATCH_psroipooling_max(deviceType, type) \
status = vl::impl::psroipooling_max<deviceType, type>::backward \
((type*)derData.getMemory(), (type const*)data.getMemory(), \
derData.getHeight(), derData.getWidth(), derData.getDepth(), derData.getSize(), \
(const type *)rois.getMemory(), rois.getNumElements() / 5, \
(type const*)derOutput.getMemory(), \
subdivisions, transform, outChannels) ;

#define DISPATCH2(deviceType, op) \
switch (dataType) { \
case VLDT_Float : DISPATCH_ ## op (deviceType, float) ; break ; \
IF_DOUBLE(case VLDT_Double : DISPATCH_ ## op (deviceType, double) ; break ;) \
default: assert(false) ; return vl::VLE_Unknown ; \
}

vl::ErrorCode
vl::nnpsroipooling_backward(vl::Context& context,
                          vl::Tensor derData,
                          vl::Tensor data,
                          vl::Tensor rois,
                          vl::Tensor derOutput,
                          PSROIPoolingMethod method,
                          int const subdivisions[2],
                          double const transform[6],
                          int const outChannels)
{
  vl::ErrorCode status = VLE_Success ;
  vl::DeviceType deviceType = derOutput.getDeviceType() ;
  vl::DataType dataType = derOutput.getDataType() ;

  switch (deviceType) {
    default:
      assert(false) ;
      return vl::VLE_Unknown ;

    case vl::VLDT_CPU:
      DISPATCH3(vl::VLDT_CPU) ;
      break ;

#if ENABLE_GPU
    case vl::VLDT_GPU:
      DISPATCH3(vl::VLDT_GPU) ;
      if (status == vl::VLE_Cuda) {
        context.setError(context.getCudaHelper().catchCudaError(__func__)) ;
      }
      break ;
#endif
  }

  return context.passError(status, "nnpsroipooling_backward") ;
}
