function net = rfcn_init(opts, varargin)
% RFCN_INIT Initialize a Faster R-CNN Detector Network
%   RFCN_INIT(OPTS) - constructs a Faster R-CNN detector 
%   according to the options provided using the autonn matconvnet
%   wrapper.

  modelName = opts.modelOpts.architecture ;
  modelDir = fullfile(vl_rootnn, 'data/models-import') ;

  switch modelName
    case 'vgg16'
      trunkPath = fullfile(modelDir, 'imagenet-vgg-verydeep-16.mat') ;
      rootUrl = 'http://www.vlfeat.org/matconvnet/models' ;
      trunkUrl = [rootUrl '/imagenet-vgg-verydeep-16.mat' ] ;
    case 'vgg16-reduced'
      trunkPath = fullfile(modelDir, 'vgg-vd-16-reduced.mat') ;
      rootUrl = 'http://www.robots.ox.ac.uk/~albanie/models' ;
      trunkUrl = [rootUrl '/ssd/vgg-vd-16-reduced.mat'] ;
    case 'resnet50'
      trunkPath = fullfile(modelDir, 'imagenet-resnet-50-dag.mat') ;
      rootUrl = 'http://www.vlfeat.org/matconvnet/models' ;
      trunkUrl = [rootUrl 'imagenet-resnet-50-dag.mat'] ;
    case 'resnext101_64x4d', error('%s not yet supported', modelName) ;
    otherwise, error('architecture %d is not recognised', modelName) ;
  end

  if ~exist(trunkPath, 'file')
    fprintf('%s not found, downloading... \n', opts.modelOpts.architecture) ;
    mkdir(fileparts(trunkPath)) ; urlwrite(trunkUrl, trunkPath) ;
  end

  net = dagnn.DagNN.loadobj(load(trunkPath)) ;
  net.removeLayer('prob') ;  net.removeLayer('fc1000') ; net.removeLayer('pool5') ; 
  rng(42) ; % for reproducibility, fix the seed

  % freeze early layers and modify trunk biases to match caffe
  net = freezeAndMatchLayers(net, opts) ;

  % configure autonn inputs
  gtBoxes = Input('gtBoxes') ; 
  gtLabels = Input('gtLabels') ; 
  imInfo = Input('imInfo') ;

  % convert to autonn
  stored = Layer.fromDagNN(net) ; net = stored{1} ;

  % Region proposal network 
  src = net.find('res4f_relu', 1) ; 
  largs = {'stride', [1 1], 'pad', [1 1 1 1], 'dilate', [1 1]} ; 
  sz = [3 3 512 512] ; addRelu = 1 ; 
  rpn_conv = add_block(src, 'rpn_conv_3x3', opts, sz, addRelu, largs{:}) ;
  numAnchors = numel(opts.modelOpts.scales) * numel(opts.modelOpts.ratios) ;

  name = 'rpn_cls_score' ; c = 2 ; sz = [1 1 512 numAnchors*c ] ; 
  addRelu = 0 ; largs = {'stride', [1 1], 'pad', [0 0 0 0], 'dilate', [1 1]} ;  
  rpn_cls = add_block(rpn_conv, name, opts, sz, addRelu, largs{:}) ;

  name = 'rpn_bbox_pred' ; b = 4 ; sz = [1 1 512 numAnchors*b ] ; addRelu = 0 ;
  largs = {'stride', [1 1], 'pad', [0 0 0 0], 'dilate', [1 1]} ;
  rpn_bbox_pred = add_block(rpn_conv, name, opts, sz, addRelu, largs{:}) ;

  largs = {'name', 'rpn_cls_score_reshape'} ;
  args = {rpn_cls, [0 -1 c 0]} ; 
  rpn_cls_reshape = Layer.create(@vl_nnreshape, args, largs{:}) ;

  args = {rpn_cls, gtBoxes, imInfo} ; % note: first input used to determine shape
  largs = {'name', 'anchor_targets', 'numInputDer', 0} ;
  [rpn_labels, rpn_bbox_targets, rpn_iw, rpn_ow, rpn_cw] = ...
                            Layer.create(@vl_nnanchortargets, args, largs{:}) ;

  % rpn losses
  args = {rpn_cls_reshape, rpn_labels, 'instanceWeights', rpn_cw} ;
  largs = {'name', 'rpn_loss_cls', 'numInputDer', 1} ;
  rpn_loss_cls = Layer.create(@vl_nnloss, args, largs{:}) ;

  weighting = {'insideWeights', rpn_iw, 'outsideWeights', rpn_ow} ;
  args = [{rpn_bbox_pred, rpn_bbox_targets, 'sigma', 3}, weighting] ;
  largs = {'name', 'rpn_loss_bbox', 'numInputDer', 1} ;
  rpn_loss_bbox = Layer.create(@vl_nnsmoothL1loss, args, largs{:}) ;

  args = {rpn_loss_cls, rpn_loss_bbox, 'locWeight', opts.modelOpts.locWeight} ;
  largs = {'name', 'rpn_multitask_loss'} ;
  rpn_multitask_loss = Layer.create(@vl_nnmultitaskloss, args, largs{:}) ;

  % RoI proposals 
  largs = {'name', 'rpn_cls_prob', 'numInputDer', 0} ;
  rpn_cls_prob = Layer.create(@vl_nnsoftmax, {rpn_cls_reshape}, largs{:}) ;

  args = {rpn_cls_prob, [0 -1 numAnchors*c 0]} ; 
  largs = {'name', 'rpn_cls_prob_reshape', 'numInputDer', 0} ; 
  rpn_cls_prob_reshape = Layer.create(@vl_nnreshape, args, largs{:}) ;

  proposalConf = {'postNMSTopN', 2000, 'preNMSTopN', 12000} ;
  featOpts = [{'featStride', opts.modelOpts.featStride}, proposalConf] ;
  args = {rpn_cls_prob_reshape, rpn_bbox_pred, imInfo, featOpts{:}} ; %#ok
  largs = {'name', 'proposal', 'numInputDer', 0} ; 
  proposals = Layer.create(@vl_nnproposalrpn, args, largs{:}) ;

  args = {proposals, gtBoxes, gtLabels, 'numClasses', opts.modelOpts.numClasses} ;
  largs = {'name', 'roi_data', 'numInputDer', 0} ;
  [rois, labels, bbox_targets, bbox_in_w, bbox_out_w, cw] = ...
                   Layer.create(@vl_nnproposaltargets, args, largs{:}) ;

  % additional convolutional layer
  src = net.find('res5c_relu', 1) ; sz = [1 1 2048 1024] ; addRelu = 1 ; 
  largs = {'stride', [1 1], 'pad', 0, 'dilate', [1 1]} ; 
  new_conv = add_block(src, 'conv_new_1', opts, sz, addRelu, largs{:}) ;

  % position sensitive features (cls)
  largs = {'stride', [1 1], 'pad', 0, 'dilate', [1 1]} ; 
  channelsOut = opts.modelOpts.numClasses * prod(opts.modelOpts.subdivisions) ;
  sz = [1 1 1024 channelsOut] ; addRelu = 0 ; 
  rfcn_cls = add_block(new_conv, 'rfcn_cls', opts, sz, addRelu, largs{:}) ;

  % position sensitive features (bbox)
  largs = {'stride', [1 1], 'pad', 0, 'dilate', [1 1]} ; 
  if opts.modelOpts.classAgnosticReg
    out = 4 * 2 * prod(opts.modelOpts.subdivisions) ;
  else
    out = 4 * opts.modelOpts.numClasses * prod(opts.modelOpts.subdivisions) ;
  end
  sz = [1 1 1024 out] ; addRelu = 0 ; 
  rfcn_bbox = add_block(new_conv, 'rfcn_bbox', opts, sz, addRelu, largs{:}) ;

  % psroipool
  group_size = opts.modelOpts.subdivisions(1) ; % caffe naming convention
  outChannels = group_size ;
  largs = {'name', 'psroipooled_cls_rois', 'numInputDer', 1} ;
  args = {rfcn_cls, rois, 'method', 'avg', 'Transform', 1/16, ...
    'Subdivisions', opts.modelOpts.subdivisions, 'outchannels', outChannels} ;
  psroipool_cls = Layer.create(@vl_nnpsroipool, args, largs{:}) ;

  largs = {'name', 'psroipooled_bbox_rois', 'numInputDer', 1} ;
  outChannels = opts.modelOpts.subdivisions(1) ;
  args = {rfcn_bbox, rois, 'method', 'avg', 'Transform', 1/16, ...
    'Subdivisions', opts.modelOpts.subdivisions, 'outchannels', outChannels} ;
  psroipool_bbox = Layer.create(@vl_nnpsroipool, args, largs{:}) ;

  cls_score = vl_nnpool(psroipool_cls, [group_size group_size], ...
                     'method', 'avg', 'stride', group_size, 'pad', 0) ;
  cls_score.name = 'ave_cls_score_rois' ;
  bbox_pred = vl_nnpool(psroipool_bbox, [group_size group_size], ...
                     'method', 'avg', 'stride', group_size, 'pad', 0) ;
  bbox_pred.name = 'ave_bbox_pred_rois' ;

  % r-cnn losses
  largs = {'name', 'loss_cls', 'numInputDer', 1} ;
  args = {cls_score, labels, 'instanceWeights', cw} ;
  loss_cls = Layer.create(@vl_nnloss, args, largs{:}) ;

  weighting = {'insideWeights', bbox_in_w, 'outsideWeights', bbox_out_w} ;
  args = [{bbox_pred, bbox_targets, 'sigma', 1}, weighting] ;
  largs = {'name', 'loss_bbox', 'numInputDer', 1} ;
  loss_bbox = Layer.create(@vl_nnsmoothL1loss, args, largs{:}) ;

  args = {loss_cls, loss_bbox, 'locWeight', opts.modelOpts.locWeight} ;
  largs = {'name', 'multitask_loss'} ;
  multitask_loss = Layer.create(@vl_nnmultitaskloss, args, largs{:}) ;



  if strcmp(opts.modelOpts.architecture, 'vgg16-reduced') % match caffe
    net.layers(net.getLayerIndex('fc6')).block.dilate = [6 6] ;
    net.layers(net.getLayerIndex('fc6')).block.pad = [6 6] ;
    net.layers(net.getLayerIndex('pool5')).block.stride = [1 1] ;
    net.layers(net.getLayerIndex('pool5')).block.poolSize = [3 3] ;
    net.layers(net.getLayerIndex('pool5')).block.pad = [1 1 1 1] ;
  end

  checkLearningParams(rpn_multitask_loss, multitask_loss, opts) ;
  net = Net(rpn_multitask_loss, multitask_loss) ;

  % set meta information to match original training code
  rgb = [122.771, 115.9465, 102.9801] ;
  net.meta.normalization.averageImage = permute(rgb, [3 1 2]) ;


% ---------------------------------------------------------------------
function net = add_block(net, name, opts, sz, nonLinearity, varargin)
% ---------------------------------------------------------------------

  filters = Param('value', init_weight(sz, 'single', opts), 'learningRate', 1) ;
  biases = Param('value', zeros(sz(4), 1, 'single'), 'learningRate', 2) ;
  cudaOpts = {'CudnnWorkspaceLimit', opts.modelOpts.CudnnWorkspaceLimit} ;
  net = vl_nnconv(net, filters, biases, varargin{:}, cudaOpts{:}) ;
  net.name = name ;

  if nonLinearity
    bn = opts.modelOpts.batchNormalization ;
    rn = opts.modelOpts.batchRenormalization ;
    assert(bn + rn < 2, 'cannot add both batch norm and renorm') ;
    if bn
      net = vl_nnbnorm(net, 'learningRate', [2 1 0.05], 'testMode', false) ;
      net.name = sprintf('%s_bn', name) ;
    elseif rn
      net = vl_nnbrenorm_auto(net, opts.clips, opts.renormLR{:}) ; 
      net.name = sprintf('%s_rn', name) ;
    end
    net = vl_nnrelu(net) ;
    net.name = sprintf('%s_relu', name) ;
  end

% ------------------------------------------------
function checkLearningParams(rpn_loss, loss, opts)
% ------------------------------------------------
% compare parameters against caffe.  

  % Algo: we first parse the prototxt and build a set of basic "layer" 
  % objects to store parameters.  These can then be directly compared against
  % their mcn equivalents to reduced the risk of incorrect initialisation.
  caffeLayers = parseCaffeLayers(opts) ;

  % loop over layers and check against network
  for ii = 1:numel(caffeLayers)
    layer = caffeLayers{ii} ;
    msg = 'checking layer settings (%d/%d): %s\n' ;
    fprintf(msg, ii, numel(caffeLayers), layer.name) ;
    ignore = {'ReLU', 'Scale', 'Silence', 'Python', 'Eltwise'} ;
    if ismember(layer.type, ignore), continue ; end
    mcnLayer = rpn_loss.find(layer.name) ;
    if ~isempty(mcnLayer)
      mcn = mcnLayer{1} ; 
    else 
      mcn = loss.find(layer.name, 1) ;
    end
    switch layer.type
      case 'Convolution'
        checkFields = {'stride', 'pad', 'dilate', 'out', 'kernel_size', ...
                       'lr_mult', 'decay_mult'} ;
        hasBias = isfield(layer, 'lr_multx') ; 
        mcnFilters = mcn.inputs{2} ; % assume square filters
        msg = 'code must be modified to handle non-square filter checks' ;
        assert(size(mcnFilters.value,1) == size(mcnFilters.value,2), msg) ;
        filterOpts = {'kernel_size', size(mcnFilters.value, 1), ...
                      'out', size(mcnFilters.value, 4), ...
                      'lr_mult', mcnFilters.learningRate, ...
                      'decay_mult', mcnFilters.weightDecay} ;
        mcnArgs = [ mcn.inputs filterOpts ] ;

        if hasBias
          mcnBias = mcnArgs{3} ; 
          biasOpts = {'lr_multx', mcnBias.learningRate, ...
                      'decay_multx', mcnBias.weightDecay} ;
          mcnArgs = [ mcnArgs biasOpts ] ; %#ok
          checkFields = [checkFields biasOpts([1 3])] ; %#ok
        end
        mcnArgs(strcmp(mcnArgs, 'CuDNN')) = [] ;

        % extract params and convert to canonical shape
        pad = str2double(layer.pad) ; 
        if isfield(layer, 'stride') 
          stride = str2double(layer.stride) ;
          if numel(stride) == 1, stride = repmat(stride, [1 2]) ; end
        else
          stride = [1 1] ; 
        end
        if numel(pad) == 1, pad = repmat(pad, [1 4]) ; end
        % handle caffe defaults
        if isfield(layer, 'dilation') 
          dilate = str2double(layer.dilation) ; 
          if numel(dilate) == 1, dilate = repmat(dilate, [1 2]) ; end
        else
          dilate = [1 1] ;
        end
        caffe.lr_mult = str2double(layer.lr_mult) ; 
        if ~isfield(layer, 'decay_mult'), caffe.decay_mult = 1 ; end
        if hasBias, caffe.lr_multx = str2double(layer.lr_multx) ; end
        if hasBias && ~isfield(layer, 'decay_multx'), caffe.decay_multx = 1 ; end
        caffe.out = str2double(layer.num_output) ;
        caffe.kernel_size = str2double(layer.kernel_size) ;
        caffe.pad = pad ; caffe.stride = stride ; caffe.dilate = dilate ; 
      case 'BatchNorm' 
        checkFields = {'lr_mult', 'lr_multx', 'lr_multxx', ...
                       'decay_mult', 'decay_multx', 'decay_multxx'} ;
        mcnMult = mcn.inputs{2} ; mcnBias = mcn.inputs{3} ; 
        mcnMoments = mcn.inputs{4} ; 
        mcnArgs = {'lr_mult', mcnMult.learningRate, ...
                   'decay_mult', mcnMult.weightDecay, ...
                   'lr_multx', mcnBias.learningRate, ...
                   'decay_multx', mcnBias.weightDecay, ...
                   'lr_multxx', mcnMoments.learningRate, ...
                   'decay_multxx', mcnMoments.weightDecay} ;
        for jj = 1:numel(checkFields)
          caffe.(checkFields{jj}) = str2double(layer.(checkFields{jj})) ;
        end
      case 'Pooling' 
        checkFields = {'stride', 'pad', 'method', 'kernel_size'} ;
        kernel_size = str2double(layer.kernel_size) ;
        if isfield(layer, 'pad'), pad = str2double(layer.pad) ; else, pad = 0 ; end
        if numel(kernel_size) == 1, kernel_size = repmat(kernel_size, [1 2]) ; end
        if numel(stride) == 1, stride = repmat(stride, [1 2]) ; end
        if numel(pad) == 1, pad = repmat(pad, [1 4]) ; end
        caffe.method = lower(layer.pool) ; caffe.pad = pad ; 
        caffe.stride = stride ; caffe.kernel_size = kernel_size ;
        poolOpts = mcn.inputs(3:end) ;
        poolOpts(strcmp(poolOpts, 'CuDNN')) = [] ;
        mcnArgs = [poolOpts {'kernel_size', mcn.inputs{2}}] ;
        
      otherwise, fprintf('checking layer type: %s\n', layer.type) ;
    end
    % run checks
    for jj = 1:numel(checkFields)
      fieldName = checkFields{jj} ;
      mcnPos = find(strcmp(mcnArgs, fieldName)) + 1 ;
      value = mcnArgs{mcnPos} ; cValue = caffe.(fieldName) ;
      if strcmp(fieldName, 'pad')
        % since mcn and caffe handle padding slightly differntly, produce a 
        % warning rather than an error for different padding settings
        if ~all(value == cValue)
          msg = 'WARNING:: pad values do not match for %s: %s vs %s\n' ;
          fprintf(msg, layer.name, mat2str(value), mat2str(cValue)) ;
        end
      else
        msg = sprintf('unmatched parameters for %s', fieldName) ;
        assert(all(value == cValue), msg) ;
      end
    end
  end

% --------------------------------------
function layers = parseCaffeLayers(opts)
% --------------------------------------
  % create name map
  nameMap = containers.Map ; 
  nameMap('rpn_conv/3x3') = 'rpn_conv_3x3' ;
  proto = fileread(opts.modelOpts.protoPath) ;

  % mini parser
  stack = {} ; tokens = strsplit(proto, '\n') ;
  assert(contains(tokens{1}, 'ResNet-50'), 'wrong proto') ; tokens(1) = [] ; 
  layers = {} ; layer = struct() ;
  while ~isempty(tokens)
    head = tokens{1} ; tokens(1) = [] ; clean = cleanStr(head) ;
    if isempty(clean) || strcmp(clean(1), '#') 
      % comment or blank proto line (do nothing)
    elseif contains(head, '}') && contains(head, '{') 
      % NOTE: it's not always necessary to flatten out subfields
      pair = strsplit(head, '{') ; key = cleanStr(pair{1}) ; 
      value = strjoin(pair(2:end), '{') ; 
      ptr = numel(value) - strfind(fliplr(value), '}') ; 
      value = value(1:ptr) ;
      ignore = {'reshape_param'} ; % caffe and mcn use different values
      examine = {'param', 'weight_filler', 'bias_filler', 'smooth_l1_loss_param'} ;
      switch key
        case ignore, continue ;
        case examine, pairs = parseInlinePairs(value) ;
        otherwise, error('nested key %s not recognised', key) ;
      end
      for jj = 1:numel(pairs)
        pair = strsplit(pairs{jj}, ':') ; 
        layer = put(layer, cleanStr(pair{1}), cleanStr(pair{2})) ;
      end
    elseif contains(head, '}'), stack(end) = [] ; 
    elseif contains(head, '{'), stack{end+1} = head ; %#ok
    else % handle some messy cases
      tuple = strsplit(head, ':') ; 
      if numel(tuple) > 2
        if strcmp(cleanStr(tuple{1}), 'param_str')
          if numel(tuple) == 3 
            % standard param_str spec form. E.g.
            %   param_str: "'feat_stride': 16"
            tuple(1) = [] ; % pop param_str specifier 
          else, keyboard
          end
        elseif numel(tuple) == 4 
          pairs = parseInlinePairs(head) ;
          for jj = 1:numel(pairs) 
            pair = strsplit(pairs{jj}, ':') ; 
            layer = put(layer, cleanStr(pair{1}), cleanStr(pair{2})) ;
          end
        else, keyboard ; 
        end
      end
      key = cleanStr(tuple{1}) ; value = cleanStr(tuple{2}) ;
      %if contains(head, 'rpn_conv/3x3'), keyboard ; end
      if isKey(nameMap, value), value = nameMap(value) ; end
      layer = put(layer, key, value) ;
    end
    if isempty(stack) && ~isempty(layer)
      layers{end+1} = layer ; layer = {} ; %#ok
    end
  end

% -------------------------------------
function layer = put(layer, key, value)
% -------------------------------------
% store key-value pairs in layer without overwriting previous
% values. Due to MATLAB key naming restrictions, an x-suffix count is used
% for indexing
    while isfield(layer, key), key = sprintf('%sx', key) ; end 
    layer.(key) = value ;

% ------------------------------------
function pairs = parseInlinePairs(str) 
% ------------------------------------
% PARSIiNLINEPAIRS parses prototxt strings in which key-value pairs 
% are supplied in a line, delimited only by whitespace.  For example:
%     kernel_size: 3 pad: 1 stride: 1

  str = strtrim(str) ; % remove leading/trailing whitespace
  dividers = strfind(str, ' ') ; 
  assert(mod(numel(dividers),2) == 1, 'expected odd number of dividers') ;
  starts = [1 dividers(2:2:end)+1] ; 
  ends = [dividers(2:2:end)-1 numel(str)] ;
  pairs = arrayfun(@(s,e) {str(s:e)}, starts, ends) ;

% --------------------------------------------
function net = freezeAndMatchLayers(net, opts)
% --------------------------------------------

  modelName = opts.modelOpts.architecture ;
  switch modelName
    case 'resnet50'
      base = {'a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'} ;
      leaves = {'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'} ;
      template = 'res2%s_branch2%s' ;
      resUnits = cellfun(@(x,y) {sprintf(template, x,y)}, base,leaves) ;
      freeze = [{'conv1', 'res2a_branch1'}, resUnits] ;
    case 'resnext101_64x4d', error('%s not yet supported', modelName) ;
    otherwise, error('architecture %d is not recognised', modelName) ;
  end

  for ii = 1:length(freeze)
    pIdx = net.getParamIndex(net.layers(net.getLayerIndex(freeze{ii})).params) ;
    [net.params(pIdx).learningRate] = deal(0) ;
    % In the original code weight decay is kept on in the conv layers
    [net.params(pIdx).weightDecay] = deal(1) ;
  end

  % freeze all batch norms during learning
  bnLayerIdx = find(arrayfun(@(x) isa(x.block, 'dagnn.BatchNorm'), net.layers)) ;

  for ii = 1:length(bnLayerIdx)
    lIdx = bnLayerIdx(ii) ;
    pIdx = net.getParamIndex(net.layers(lIdx).params) ;
    [net.params(pIdx).learningRate] = deal(0) ;
    [net.params(pIdx).weightDecay] = deal(0) ;
  end

  % Unit 5 of the resnet is modified slightly for detection
  net.layers(net.getLayerIndex('res5a_branch1')).block.stride = [1 1] ;
  net.layers(net.getLayerIndex('res5a_branch2a')).block.stride = [1 1] ;
  dilateLayers = {'res5a_branch2b', 'res5b_branch2b', 'res5c_branch2b' } ;
  for ii = 1:numel(dilateLayers)
    lIdx = net.getLayerIndex(dilateLayers{ii}) ;
    net.layers(lIdx).block.dilate = [2 2] ;
    net.layers(lIdx).block.pad = [2 2 2 2] ;
  end

  %for ii = 1:numel(net.layers)
    %if isa(net.layers(ii).block, 'dagnn.Conv')
      %%fprintf('dilation: %s\n', mat2str(net.layers(ii).block.dilate)) ;
      %msg = '%s stride: %s\n' ;
      %fprintf(msg, net.layers(ii).name, mat2str(net.layers(ii).block.stride)) ;
    %end
  %end

% --------------------------
function str = cleanStr(str)
% --------------------------
% prune unused space and punctuation from prototxt files
  % clean up 
  str = strrep(strrep(strrep(str, '"', ''), ' ', ''), '''', '') ;
  % stop at comments
  comment = strfind(str, '#') ;
  if ~isempty(comment)
    str = str(1:comment(1)-1) ; % stop at first #
  end


% --------------------------------------------
function weights = init_weight(sz, type, opts)
% --------------------------------------------
% Match caffe fixed scale initialisation, which seems to do 
% better than the Xavier heuristic here

  switch opts.modelOpts.initMethod
    case 'gaussian'
      % in the original code, bounding box regressors are initialised 
      % slightly differently
      numRegressors = opts.modelOpts.numClasses * 4 ;
      if sz(4) ~= numRegressors, sc = 0.01 ; else, sc = 0.001 ; end
    case 'xavier', sc = sqrt(1/(sz(1)*sz(2)*sz(3))) ;
    otherwise, error('%s method not recognised', opts.modelOpts.initMethod) ;
  end
  weights = randn(sz, type)*sc ;
