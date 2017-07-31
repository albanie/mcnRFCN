// @file nnpsroipooling.hpp
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

#ifndef __vl__nnpsroipooling__
#define __vl__nnpsroipooling__

#include "data.hpp"
#include <stdio.h>

namespace vl {
  enum PSROIPoolingMethod { vlPSROIPoolingMax, vlPSROIPoolingAverage } ;

  vl::ErrorCode
  nnpsroipooling_forward(vl::Context& context,
                       vl::Tensor output,
                       vl::Tensor data,
                       vl::Tensor rois,
                       PSROIPoolingMethod method,
                       int const subdivisions[2],
                       double const transform[6],
                       int const outChannels) ;

  vl::ErrorCode
  nnpsroipooling_backward(vl::Context& context,
                        vl::Tensor derData,
                        vl::Tensor data,
                        vl::Tensor rois,
                        vl::Tensor derOutput,
                        PSROIPoolingMethod method,
                        int const subdivisions[2],
                        double const transform[6],
                        int const outChannels) ;
}

#endif /* defined(__vl__nnpsroipooling__) */
