% utility script

featDir = fullfile(vl_rootnn, 'contrib/mcnRFCN/feats') ;
featPath = fullfile(featDir, 'blobs-ResNet-50.mat') ;
feats = load(featPath) ;

imMinusPath = fullfile(featDir, 'im-minus.mat') ;
imMinusData = load(imMinusPath) ;
imMinus = imMinusData.im_minus(:,:, [3 2 1]) ;
checkPreprocessing = 0 ;

if checkPreprocessing
  imPath = fullfile(vl_rootnn, 'contrib/mcnRFCN/misc/python/000067.jpg') ; %#ok
  im = single(imread(imPath)) ;

  sz = double(size(im)) ; imsz = sz(1:2) ;
  sc = 600 ; maxSc = 1000 ; 
  factor = max(sc ./ imsz) ; 
  if any((imsz * factor) > maxSc), factor = min(maxSc ./ imsz) ; end
  newSz = factor .* imsz ; im_info = [ round(newSz) factor ] ;
  imMean = dag.meta.normalization.averageImage ;
  im = bsxfun(@minus, im, imMean) ;
  data = imresize(im, factor, 'bilinear') ;
else
  data = permute(feats.data, [3 4 2 1]) ;
  data = data(:,:, [3 2 1]) ;
  im_info = feats.im_info ;
end

in = {'data', gpuArray(data), 'im_info', im_info} ;
dag.move('gpu') ; % only cuda version of psroipool works now

dag.conserveMemory = 0 ;
dag.mode = 'test' ;
dag.eval(in) ;

% determine name map
map = containers.Map() ; 
xName = 'data' ; prev = xName ;
for ii = 1:numel(dag.layers)
  prevPrev = prev ;
  prev = xName ;
  xName = dag.layers(ii).name ;
  fprintf('%d: processing %s\n', ii, xName) ;
  % from convs, only relu outputs are stored

  % special rules
  if strcmp(xName, 'conv1'), map(xName) = 'conv1_raw'; continue ; end 
  % mcn bnorm should incorporate following scale layer:
  if strcmp(xName, 'bn_conv1'), map('conv1x') = 'conv1_scale'; continue ; end 
  if strcmp(xName, 'conv1_relu'), map('conv1xxx') = 'conv1'; continue ; end 

  if contains(xName, 'norm'), map(xName) = xName; end % norm uses same naming
  if contains(xName, 'pool'), map(xName) = xName; end % pool uses same naming
  % conv -> BN -> scale -> ReLU
  if contains(xName, 'relu') %slightly more complex logic for resnet
    if contains(prev, 'bn')
      map(sprintf('%sxxx', prevPrev)) = prevPrev ; % skip back over BN
    else
      map(sprintf('%sxxx', prev)) = prev ; 
    end
  end
end

keepers = {'rpn_cls_score', 'rpn_bbox_pred', 'rois', 'pool5', ...
           'bbox_pred', 'cls_prob', 'rfcn_cls', 'rfcn_bbox', ...
           'psroipooled_cls_rois', 'psroipooled_loc_rosi'} ;
for ii = 1:numel(keepers)
  map(keepers{ii}) = keepers{ii} ;
end

% save for psroi dev
%rois = dag.vars(dag.getVarIndex('rois')).value ;
%rfcn_cls = permute(feats.rfcn_cls, [3 4 2 1]) ;
%rfcn_bbox = permute(feats.rfcn_bbox, [3 4 2 1]) ;
%psroipooled_cls_rois = permute(feats.psroipooled_cls_rois, [3 4 2 1]) ;
%psroipooled_loc_rois = permute(feats.psroipooled_loc_rois, [3 4 2 1]) ;
%s.psroipooled_loc_rois = psroipooled_loc_rois ;
%s.psroipooled_cls_rois = psroipooled_cls_rois ;
%s.rois = feats.rois ; s.rfcn_cls = rfcn_cls ; s.rfcn_bbox = rfcn_bbox ;
%save(fullfile(vl_rootnn, 'data/psroi-inputs.mat'), '-struct', 's') ;
  
for ii = 1:numel(dag.vars)
  xName = dag.vars(ii).name ;
  %fprintf('%d: %s\n', ii, xName) ;
  if ~isKey(map, xName), continue ; end 
  xName_ = map(xName) ;

  x = gather(dag.vars(dag.getVarIndex(xName)).value) ;
  x_ = feats.(xName_) ;
  x_ = permute(x_, [3 4 2 1]) ;

  if strcmp(xName, 'rois') 
    x_ = squeeze(x_) ; x_ = x_(2:end,:) ; % remove the image index
    x = x(2:end,:) - 1 ; % fix off by one in MATLAB
  end

  %if strcmp(xName, 'bbox_pred') 
    %keyboard
  %end
  diff = x(:) - x_(:) ;
  fprintf('%d: %s vs %s\n', ii, xName, xName_) ;
  fprintf('diff: %g\n', mean(abs(diff))) ;
end
