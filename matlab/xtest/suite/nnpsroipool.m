classdef nnpsroipool < nntest
%NNPSROIPOOL - based on NNROIPOILL

  properties (TestParameter)
    %method = {'avg', 'max'}
    %subdivisions = {[1 1], [2 1], [1 2], [7 7]}
    outChannels = {1 2 3 5} ;
    method = {'max'}
    subdivisions = {[7 7]}
    %outChannels = {4 5} ;
  end

  methods (Test)
    function basic(test, method, subdivisions, outChannels)
      % NOTE: It is quite difficult to do good numerical checks on 
      % pooling derivatives.  One approach is to ensure that all 
      % numerical values of the inputs are unique (which avoids 
      % tie-breaking for max pooling derivatives)
      rng(0) ;

      % ensure that all inputs are unique
      numChannels = prod(subdivisions) * outChannels ;
      x = test.randn(15,14,numChannels,2) ;
      x(:) = randperm(numel(x))' ;

      R = [1  1 1 2  2 2 1 1 ;
           0  1 2 0  1 2 1 1 ;
           0  4 3 0  1 2 1 1 ;
           15 5 6 15 4 2 9 0 ;
           14 7 9 14 4 8 1 0] ; % supply some invalid rois

      R = test.toDevice(test.toDataType(R)) ;
      %R = R + 1 ;

      test.range = 10 ;
      if strcmp(test.currentDevice,'gpu'), x = gpuArray(x) ; end
      args = {'method', method, ...
              'subdivisions', subdivisions, ...
              'outChannels', outChannels} ;
      y = vl_nnpsroipool(x,R,args{:}) ;
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnpsroipool(x,R,dzdy,args{:}) ;
      test.der(@(x) vl_nnpsroipool(x,R,args{:}), ...
               x, dzdy, dzdx, test.range * 1e-2) ;
    end
  end
end
