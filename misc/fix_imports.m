function fix_imports(varargin)
%FIX_IMPORTS - clean up imported caffe models
%   FIX_IMPORTS performs some additional clean up work
%   on models imported from caffe to ensure that they are
%   consistent with matconvnet conventions
%
%  TODO: It is much less brittle to try to fix these issues
%  in the caffe import script. The functionality below should be 
%  moved there once the interface is considered stable.

  opts.dataset = 'pascal' ;
  opts.modelName = 'rfcn-res50-pascal.mat' ;
  opts.modelDir = fullfile(vl_rootnn, 'data/models-import') ;
  opts = vl_argparse(opts, varargin) ;

  % select model
  modelPath = fullfile(opts.modelDir, opts.modelName) ;
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

  switch opts.dataset % add classes if not present
    case 'pascal'
      classes = {'background', 'aeroplane', 'bicycle', 'bird', ...
         'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', ...
         'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', ...
         'sofa', 'train', 'tvmonitor'} ; 
    case 'coco'
      [map, labels] = getCocoLabelMap() ;
      net.meta.classes.labelMap = map ; % useful for converting predictions
      classes = labels ;
  end
  net.meta.classes.name = classes ; 
  net.meta.classes.description = classes ; % conform to standard interface

  if isempty(net.meta.normalization.averageImage)
    rgb = [122.771, 115.9465, 102.9801] ; % imagenet mean used in orig faster rcnn
    net.meta.normalization.averageImage = permute(rgb, [3 1 2]) ;
  end

  net.meta.classAgnosticReg = 1 ; % by default, this is the R-FCN approach
  net = net.saveobj() ; save(modelPath, '-struct', 'net') ; %#ok
