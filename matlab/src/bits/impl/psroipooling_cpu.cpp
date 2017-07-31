// @file roipooling_cpu.cpp
// @brief Position Sensitive Pooling block implementation (CPU)
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

#include "matrix.h"
#include "../data.hpp"
#include "psroipooling.hpp"

#include <limits>
#include <algorithm>
#include <cmath>

using std::max ;
using std::min ;

/* ---------------------------------------------------------------- */
/*                                          max psroipooling helper */
/* ---------------------------------------------------------------- */

template <typename type>
struct acc_max
{
  inline acc_max(int poolHeight, int poolWidth, type derOutput = 0)
  :
  value(-std::numeric_limits<type>::infinity()),
  derOutput(derOutput),
  derDataActivePt(NULL)
  { }

  inline void accumulate_forward(type x) {
    value = std::max(value, x) ;
  }

  inline void accumulate_backward(type const* data, type* derDataPt) {
    type x = *data ;
    if (x > value) {
      value = x ;
      derDataActivePt = derDataPt ;
    }
  }

  inline type done_forward() const {
    return value ;
  }

  inline void done_backward() const {
    if (derDataActivePt) { *derDataActivePt += derOutput ; }
  }

  type value ;
  type derOutput ;
  type* derDataActivePt ; } ; 
/* ---------------------------------------------------------------- */
/*                                      average psroipooling helper */
/* ---------------------------------------------------------------- */

template <typename type>
struct acc_sum
{
  inline acc_sum(int poolHeight, int poolWidth, type derOutput = 0)
  :
  value(0),
  scale(type(1)/type(poolHeight*poolWidth)),
  derOutput(derOutput)
  { }

  inline void accumulate_forward(type x) {
    value += x ;
  }

  inline void accumulate_backward(type const* data, type* derDataPt) {
    *derDataPt += derOutput * scale ;
  }

  inline type done_forward() const {
    return value * scale ;
  }

  inline void done_backward() const { }

  type value ;
  type derOutput ;
  type scale;
} ;

/* ---------------------------------------------------------------- */
/*                                           psroipooling_*_forward */
/* ---------------------------------------------------------------- */

template<typename type, typename Accumulator> static inline void
psroipooling_forward_cpu(type* pooled,
                       type const* data,
                       size_t height, size_t width, size_t depth, size_t size,
                       type const* rois,
                       size_t numROIs,
                       size_t outChannels,
                       int const subdivisions[2],
                       double const transform[6])
{
  // For each ROI R = [t x1 y1 x2 y2].
  for (int roi = 0; roi < numROIs; ++roi) {

    // Apply scale and offset to each ROI coordinate.
    type u1_ = rois[5 * roi + 1] ;
    type v1_ = rois[5 * roi + 2] ;
    type u2_ = rois[5 * roi + 3] ;
    type v2_ = rois[5 * roi + 4] ;

    int roi_image   = (int)rois[5 * roi + 0];
    // As in the GPU code, we add a (temporary) second implementation to 
    // achieve numerical consistency at rounding stages
    bool NUMERICAL_MATCH = 1 ;

    if (NUMERICAL_MATCH) {

      // Modified bound implementation 
      // This code computes the pooling bounds use by the (ps)roi pooling 
      // operation, with the objective of reproducing the same quantization
      // used in the original author's implementation.

      // The upper left coordinates are shifted by 1 to ensure 0-indexing.
      // Each coordinate is then quantize in image-space to lie exactly over
      // a set of pixels, then transform to feature space with an axis 
      // aligned scaling 
      type roi_start_w = static_cast<type>(round(u1_-1)) * transform[0] ;
      type roi_start_h = static_cast<type>(round(v1_-1)) * transform[3] ;
      type roi_end_w = static_cast<type>(round(u2_)) * transform[0] ; 
      type roi_end_h = static_cast<type>(round(v2_)) * transform[3] ; 

      // To avoid pooling over empty bins, ensure that every roi in feature
      // space has non-zero height and width
      type roi_height = max(roi_end_h - roi_start_h, static_cast<type>(0.1)) ;
      type roi_width = max(roi_end_w - roi_start_w, static_cast<type>(0.1)) ;

      // compute the spatial dimensions of each bin in feature space 
      type bin_size_h = roi_height / static_cast<type>(subdivisions[0]) ;
      type bin_size_w = roi_width / static_cast<type>(subdivisions[1]) ;

      // determine the image id associated with the current roi (there should
      // be at most one per batch)
      roi_image = min(max(roi_image - 1,0), (int)size - 1) ;

      // shift the input data pointer to skip inputs associated with previous
      // image
      type const * data_offset = data + (roi_image * depth) * (width*height) ;

        // For each selected out channel.
        for (int z = 0; z < outChannels; ++z) {

          // For each column of tiles.
          for (int pw = 0; pw < subdivisions[1]; ++pw) {

            // compute the horizontal bounds of the pooling window for 
            // the current cell, constaining it to lie in the feature map
            int wstart = (int)floor(((type)pw) * bin_size_w + roi_start_w) ;
            int wend = (int)ceil(((type)(pw + 1)) * bin_size_w + roi_start_w) ;
            wstart = min(max(wstart, 0), (int)width) ;
            wend = min(max(wend, 0), (int)width) ;

            // For each tile in a column.
            for (int ph = 0; ph < subdivisions[0]; ++ph) {

              // compute the vertical bounds of the pooling window for 
              // the current tile, constaining it to lie in the feature map
              int hstart = (int)floor(((type)ph) * bin_size_h + roi_start_h) ;
              int hend = (int)ceil(((type)(ph + 1)) * bin_size_h + roi_start_h) ;
              hstart = min(max(hstart, 0), (int)height) ;
              hend = min(max(hend, 0), (int)height) ;

              bool is_empty = (hend <= hstart) || (wend <= wstart);

              // Finally, we add an additional offset to the input data pointer
              // which is responsible for selecting the correct "position 
              // sensitive" features associated with the current tile. In this
              // case, the channel positions are stored in row major order, so
              // we must compute offsets accordingly
              int num_feat_pixels = width * height ;
              int step_over = (z*subdivisions[0] + ph)*subdivisions[1] + pw - z ;
              int off = step_over * num_feat_pixels ; 

              if (is_empty) {
                *pooled++ = 0 ;
              }
              else {
                Accumulator acc(hend - hstart, wend - wstart) ;
                for (int w = wstart ; w < wend; ++w) {
                  for (int h = hstart ; h < hend; ++h) {
                    const int index = off + w * height + h ;
                    acc.accumulate_forward(data_offset[index]) ;
                  }
                }
                *pooled++ = acc.done_forward() ;
              }
            } // end of ph
          } // end of pw
          data_offset += width*height;
        } // end of z

    } else {
        type u1 = transform[0] * u1_ + transform[2] * v1_ + transform[4] ;
        type v1 = transform[1] * u1_ + transform[3] * v1_ + transform[5] ;
        type u2 = transform[0] * u2_ + transform[2] * v2_ + transform[4] ;
        type v2 = transform[1] * u2_ + transform[3] * v2_ + transform[5] ;

        // First and last pixel of each ROI (rounded
        // for compatibility with the Caffe definition).
        int roi_start_h = (int)round(v1) - 1 ;
        int roi_start_w = (int)round(u1) - 1 ;
        int roi_end_h   = (int)round(v2) - 1 ;
        int roi_end_w   = (int)round(u2) - 1 ;
        int roi_height  = max(roi_end_h - roi_start_h + 1, 1) ;
        int roi_width   = max(roi_end_w - roi_start_w + 1, 1) ;

        roi_image = min(max(roi_image - 1,0), (int)size - 1) ;
        type const * data_offset = data + (roi_image * depth) * (width*height) ;

        type bin_size_h = (type)roi_height / subdivisions[0] ;
        type bin_size_w = (type)roi_width / subdivisions[1] ;

        // For each feature channel.
        for (int z = 0; z < depth; ++z) {

          // For each column of tiles.
          for (int pw = 0; pw < subdivisions[1]; ++pw) {
            int wstart = (int)floor(((type)pw) * bin_size_w) ;
            int wend = (int)ceil(((type)(pw + 1)) * bin_size_w) ;
            wstart = min(max(wstart + roi_start_w, 0), (int)width) ;
            wend = min(max(wend + roi_start_w, 0), (int)width) ;

            // For each tile in a column.
            for (int ph = 0; ph < subdivisions[0]; ++ph) {
              int hstart = (int)floor(((type)ph) * bin_size_h) ;
              int hend = (int)ceil(((type)(ph + 1)) * bin_size_h) ;
              hstart = min(max(hstart + roi_start_h, 0), (int)height) ;
              hend = min(max(hend + roi_start_h, 0), (int)height) ;

              bool is_empty = (hend <= hstart) || (wend <= wstart);

              if (is_empty) {
                *pooled++ = 0 ;
              }
              else {
                Accumulator acc(hend - hstart, wend - wstart) ;
                for (int w = wstart ; w < wend; ++w) {
                  for (int h = hstart ; h < hend; ++h) {
                    const int index = w * height + h ;
                    acc.accumulate_forward(data_offset[index]) ;
                  }
                }
                *pooled++ = acc.done_forward() ;
              }
            } // end of ph
          } // end of pw
          data_offset += width*height;
        } // end of z
    } // different numerical implementations
  } // end of n
}

/* ---------------------------------------------------------------- */
/*                                          psroipooling_*_backward */
/* ---------------------------------------------------------------- */

/*
 assume the output array to be cleared or otherwise
 properly initialised: accumulates the derivative
 */

template<typename type, typename Accumulator> static inline void
psroipooling_backward_cpu (type* derData,
                         type const* data,
                         size_t height, size_t width, size_t depth, size_t size,
                         type const* rois,
                         size_t numROIs,
                         size_t outChannels,
                         type const* derOutput,
                         int const subdivisions[2],
                         double const transform[6])
{
  // For each ROI R = [t x1 y1 x2 y2].
  for (size_t roi = 0; roi < numROIs ; ++roi) {

    // Apply sacle and offset to each ROI coordinate.
    type u1_ = rois[5 * roi + 1] ;
    type v1_ = rois[5 * roi + 2] ;
    type u2_ = rois[5 * roi + 3] ;
    type v2_ = rois[5 * roi + 4] ;

    type u1 = transform[0] * u1_ + transform[2] * v1_ + transform[4] ;
    type v1 = transform[1] * u1_ + transform[3] * v1_ + transform[5] ;
    type u2 = transform[0] * u2_ + transform[2] * v2_ + transform[4] ;
    type v2 = transform[1] * u2_ + transform[3] * v2_ + transform[5] ;

    // First and last pixel of each ROI (rounded
    // for compatibility with the Caffe definition).
    int roi_image   = (int)rois[5 * roi + 0];
    int roi_start_h = (int)round(v1) - 1 ;
    int roi_start_w = (int)round(u1) - 1 ;
    int roi_end_h   = (int)round(v2) - 1 ;
    int roi_end_w   = (int)round(u2) - 1 ;
    int roi_height = max(roi_end_h - roi_start_h + 1, 1) ;
    int roi_width = max(roi_end_w - roi_start_w + 1, 1) ;

    roi_image = min(max(roi_image - 1,0), (int)size - 1) ;
    type const * data_offset = data + (roi_image * depth) * (width*height);
    type * derData_offset = derData + (roi_image * depth) * (width*height);

    const type bin_size_h = (double)roi_height / subdivisions[0] ;
    const type bin_size_w = (double)roi_width / subdivisions[1] ;

    // For each feature channel.
    for (int z = 0; z < depth; ++z) {

      // For each column of tiles.
      for (int pw = 0; pw < subdivisions[1]; ++pw) {
        int wstart = (int)floor(((type)pw) * bin_size_w) ;
        int wend = (int)ceil(((type)(pw + 1)) * bin_size_w) ;
        wstart = min(max(wstart + roi_start_w, 0), (int)width) ;
        wend = min(max(wend + roi_start_w, 0), (int)width) ;

        // For each tile in a column.
        for (int ph = 0; ph < subdivisions[0]; ++ph) {
          int hstart = (int)floor(((type)ph) * bin_size_h) ;
          int hend = (int)ceil(((type)(ph + 1)) * bin_size_h) ;
          hstart = min(max(hstart + roi_start_h, 0), (int)height) ;
          hend = min(max(hend + roi_start_h, 0), (int)height) ;

          Accumulator acc(hend - hstart, wend - wstart, *derOutput++) ;
          for (int w = wstart; w < wend; ++w) {
            for (int h = hstart; h < hend; ++h) {
              const int index = w * height + h ;
              acc.accumulate_backward(&data_offset[index],
                                      &derData_offset[index]) ;
            }
          }
          acc.done_backward() ;
        } // end of pw
      } // end of ph
      data_offset += width*height ;
      derData_offset += width*height ;
    } // end of z
  } // end of n
}

/* ---------------------------------------------------------------- */
/*                                                        Interface */
/* ---------------------------------------------------------------- */

namespace vl { namespace impl {

  template <typename type>
  struct psroipooling_max<vl::VLDT_CPU, type>
  {
    static vl::ErrorCode
    forward(type* output,
            type const* data,
            size_t height, size_t width, size_t depth, size_t size,
            type const* rois,
            size_t numROIs,
            int const subdivisions[2],
            double const transform[6], 
            int const outChannels)
    {
      psroipooling_forward_cpu<type, acc_max<type> > (output,
                                                    data, height, width, depth, size,
                                                    rois, numROIs, outChannels,
                                                    subdivisions, transform) ;
      return VLE_Success ;
    }

    static vl::ErrorCode
    backward(type* derData,
             type const* data,
             size_t height, size_t width, size_t depth, size_t size,
             type const* rois,
             size_t numROIs,
             type const* derOutput,
             int const subdivisions[2],
             double const transform[6], 
             int const outChannels)
    {
      psroipooling_backward_cpu<type, acc_max<type> > (derData,
                                                     data, height, width, depth, size,
                                                     rois, numROIs, outChannels,
                                                     derOutput,
                                                     subdivisions, transform) ;
      return VLE_Success ;
    }
  } ; // psroipooling_max


  template <typename type>
  struct psroipooling_average<vl::VLDT_CPU, type>
  {
    static vl::ErrorCode
    forward(type* output,
            type const* data,
            size_t height, size_t width, size_t depth, size_t size,
            type const* rois,
            size_t numROIs,
            int const subdivisions[2],
            double const transform[6], 
            int const outChannels)
    {
      psroipooling_forward_cpu<type, acc_sum<type> > (output,
                                                    data, height, width, depth, size,
                                                    rois, numROIs, outChannels,
                                                    subdivisions, transform) ;
      return VLE_Success ;
    }

    static vl::ErrorCode
    backward(type* derData,
             type const* data, // <- this is not needed for avg pooling
             size_t height, size_t width, size_t depth, size_t size,
             type const* rois,
             size_t numROIs,
             type const* derOutput,
             int const subdivisions[2],
             double const transform[6], 
             int const outChannels)
    {
      psroipooling_backward_cpu<type, acc_sum<type> > (derData,
                                                     data, height, width, depth, size,
                                                     rois, numROIs, outChannels,
                                                     derOutput,
                                                     subdivisions, transform) ;
      return VLE_Success ;
    }
  } ; // psroipooling_average

} } ; // namespace vl::impl

// Instantiations
template struct vl::impl::psroipooling_average<vl::VLDT_CPU, float> ;
template struct vl::impl::psroipooling_max<vl::VLDT_CPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::psroipooling_average<vl::VLDT_CPU, double> ;
template struct vl::impl::psroipooling_max<vl::VLDT_CPU, double> ;
#endif
