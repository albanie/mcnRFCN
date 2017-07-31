function fix_imports(varargin)
%FIX_IMPORTS - clean up imported caffe models
%   FIX_IMPORTS performs some additional clean up work
%   on models imported from caffe to ensure that they are
%   consistent with matconvnet conventions
%
%  TODO: It is much less brittle to try to fix these issues
%  in the caffe import script. The functionality below should be 
%  moved there once the interface is considered stable.

  opts.modelDir = fullfile(vl_rootnn, 'data/models-import') ;
  opts = vl_argparse(opts, varargin) ;

  % Res 50 model
  modelPath = fullfile(opts.modelDir, 'rfcn-res50-pascal.mat') ;
  net = load(modelPath) ; net = dagnn.DagNN.loadobj(net) ; 
  pIdx = net.getLayerIndex('proposal') ; % fix proposal layer opt types
  scales = double(net.layers(pIdx).block.scales) ;
  net.layers(pIdx).block.scales = reshape(scales, 1, [])  ;
  net.layers(pIdx).block.featStride = double(net.layers(pIdx).block.featStride) ;

  % fix reshaping layers
  for ii = 1:numel(net.layers)
    if isa(net.layers(ii).block, 'dagnn.Reshape')
      s = net.layers(ii).block.shape ; % ensure consistency of shape/type
      net.layers(ii).block.shape = double(reshape(s, 1, [])) ; 
    end
  end

  % fix psroipooling layers
  for ii = 1:numel(net.layers)
    if isa(net.layers(ii).block, 'dagnn.PSROIPooling')
      s = net.layers(ii).block ; % ensure consistency of types
      net.layers(ii).block.outchannels = double(s.outchannels) ; 
    end
  end

  % fix post-psroipooling pooling layers -> this arises from the broken
  % shape issue, which causes the pooling layers following PSROIs to be 
  % set to NaNs
  for ii = 1:numel(net.layers)
    if isa(net.layers(ii).block, 'dagnn.Pooling')
      s = net.layers(ii).block.pad ; 
      if all(isnan(s)), s = [0 0 0 0] ; end % replace
      if any(isnan(s)), error('unexpected nan count') ; end
      net.layers(ii).block.pad = s ; 
    end
  end

  net = net.saveobj() ; save(modelPath, '-struct', 'net') ; %#ok
