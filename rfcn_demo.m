function rfcn_demo(varargin)
%RFCN_DEMO Minimalistic demonstration of the R-FCN detector

  opts.modelPath = '' ;
  opts.gpus = 2 ;
  opts.scale = 600 ;
  opts.maxScale = 1000 ;
  opts.nmsThresh = 0.3 ;
  opts.confThresh = 0.8 ;
  opts = vl_argparse(opts, varargin) ;

  % The network is trained to prediction occurences
  % of the following classes from the pascal VOC challenge
  classes = {'none_of_the_above', 'aeroplane', 'bicycle', 'bird', ...
     'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', ...
     'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', ...
     'sofa', 'train', 'tvmonitor'} ;

  % Load or download an example R-FCN model:
  modelName = 'rfcn-res50-pascal.mat' ;
  paths = {opts.modelPath, ...
           modelName, ...
           fullfile(vl_rootnn, 'data', 'models-import', modelName)} ;
  ok = find(cellfun(@(x) exist(x, 'file'), paths), 1) ;

  if isempty(ok)
    fprintf('Downloading the R-FCN model ... this may take a while\n') ;
    opts.modelPath = fullfile(vl_rootnn, 'data/models', modelName) ;
    mkdir(fileparts(opts.modelPath)) ;
    baseUrl = 'http://www.robots.ox.ac.uk/~albanie/models' ;
    url = fullfile(baseUrl, sprintf('/r-fcn-models/%s', modelName)) ;
    urlwrite(url, opts.modelPath) ;
  else
    opts.modelPath = paths{ok} ;
  end

  % Load the network and put it in test mode.
  net = load(opts.modelPath) ; net = dagnn.DagNN.loadobj(net);
  net.mode = 'test' ;

  % Load example pascal image
  imPath = fullfile(vl_rootnn, 'contrib/mcnRFCN/misc/000017.jpg') ;
  im = single(imread(imPath)) ;

  % choose variables to track
  net.conserveMemory = 0 ;
  clsIdx = net.getVarIndex('cls_prob') ;
  bboxIdx = net.getVarIndex('bbox_pred') ;
  roisIdx = net.getVarIndex('rois') ;
  %[net.vars([clsIdx bboxIdx roisIdx]).precious] = deal(true) ;

  % resize to meet the r-fcn size criteria
  imsz = [size(im,1) size(im,2)] ; maxSc = opts.maxScale ; 
  factor = max(opts.scale ./ imsz) ; 
  if any((imsz * factor) > maxSc), factor = min(maxSc ./ imsz) ; end
  newSz = factor .* imsz ; imInfo = [ round(newSz) factor ] ;
  data = imresize(im, factor, 'bilinear') ; 

  % run network and retrieve results
  net.eval({'data', data, 'im_info', single(imInfo)}) ;
  keyboard

  if 1
    %net.removeLayer('classifier_0') ;
    for ii = 1:numel(net.layers)
      lName = net.layers(ii).name ;
      out = net.layers(ii).outputs{1} ;
      sz = size(net.vars(net.getVarIndex(out)).value) ;
      if numel(sz) == 2, sz = [ sz 1 ] ; end %#ok
      fprintf('size %s: [%d x %d x %d]\n', lName, sz(1), sz(2), sz(3)) ;
    end
  end

  probs = squeeze(net.vars(clsIdx).value) ;
  deltas = squeeze(net.vars(bboxIdx).value) ;
  boxes = net.vars(roisIdx).value(2:end,:)' / imInfo(3) ;

  % Visualize results for one class at a time
  for i = 2:numel(classes)
    c = find(strcmp(classes{i}, net.meta.classes.name)) ;
    cprobs = probs(c,:) ;
    cdeltas = deltas(4*(c-1)+(1:4),:)' ;

    cboxes = bbox_transform_inv(boxes, cdeltas);
    cls_dets = [cboxes cprobs'] ;

    keep = bbox_nms(cls_dets, opts.nmsThresh) ;
    cls_dets = cls_dets(keep, :) ;

    sel_boxes = find(cls_dets(:,end) >= opts.confThresh) ;
    if numel(sel_boxes) == 0, continue ; end

    bbox_draw(im/255,cls_dets(sel_boxes,:));
    title(sprintf('Dets for class ''%s''', classes{i})) ;
    if exist('zs_dispFig', 'file'), zs_dispFig ; end

    fprintf('Detections for category ''%s'':\n', classes{i});
    for j=1:size(sel_boxes,1)
      bbox_id = sel_boxes(j,1);
      fprintf('\t(%.1f,%.1f)\t(%.1f,%.1f)\tprobability=%.6f\n', ...
              cls_dets(bbox_id,1), cls_dets(bbox_id,2), ...
              cls_dets(bbox_id,3), cls_dets(bbox_id,4), ...
              cls_dets(bbox_id,end));
    end
  end
