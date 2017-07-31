// @file psroipooling.hpp
// @brief psroipooling block implementation
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

#ifndef VL_PSROIPOOLING_H
#define VL_PSROIPOOLING_H

#include <bits/data.hpp>
#include <cstddef>

namespace vl { namespace impl {

    template<vl::DeviceType dev, typename type>
    struct psroipooling_max {
      typedef type data_type ;

      static vl::ErrorCode
      forward(type* output,
              type const* data,
              size_t height, size_t width, size_t depth, size_t size,
              type const* rois,
              size_t numROIs,
              int const subdivisions[2],
              double const transform[6],
              int const outChannels) ;

      static vl::ErrorCode
      backward(type* derData,
               type const* data,
               size_t height, size_t width, size_t depth, size_t size,
               type const* rois,
               size_t numROIs,
               type const* derOutput,
               int const subdivisions[2],
               double const transform[6],
               int const outChannels) ;
    };

    template<vl::DeviceType dev, typename type>
    struct psroipooling_average {
      typedef type data_type ;

      static vl::ErrorCode
      forward(type* output,
              type const* data,
              size_t height, size_t width, size_t depth, size_t size,
              type const* rois,
              size_t numROIs,
              int const subdivisions[2],
              double const transform[6],
              int const outChannels) ;

      static vl::ErrorCode
      backward(type* derData,
               type const* data, // <- todo: this is not needed for avg pooling
               size_t height, size_t width, size_t depth, size_t size,
               type const* rois,
               size_t numROIs,
               type const* derOutput,
               int const subdivisions[2],
               double const transform[6],
               int const outChannels) ;
    };
} }
#endif /* defined(VL_PSROIPOOLING_H) */
