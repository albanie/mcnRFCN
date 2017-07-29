function fix_imports(varargin)
%FIX_IMPORTS - clean up imported caffe models
%   FIX_IMPORTS performs some additional clean up work
%   on models imported from caffe to ensure that they are
%   consistent with matconvnet conventions

  opts.modelDir = fullfile(vl_rootnn, 'data/models-import') ;
  opts = vl_argparse(opts, varargin) ;

  % Res 50 model
  modelPath = fullfile(opts.modelDir, 'rfcn-res50-pascal.mat') ;
  net = load(modelPath) ; net = dagnn.DagNN.loadobj(net) ; 
  pIdx = net.getLayerIndex('proposal') ; % fix proposal layer opt types
  scales = double(net.layers(pIdx).block.scales) ;
  net.layers(pIdx).block.scales = reshape(scales, 1, [])  ;
  net.layers(pIdx).block.featStride = double(net.layers(pIdx).block.featStride) ;
  net = net.saveobj() ; save(modelPath, '-struct', 'net') ; %#ok
