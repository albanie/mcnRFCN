function results = rfcn_coco_evaluation(varargin)
%RFCN_COCO_EVALUATION  Evaluate a trained R-FCN model on MS COCO
%   RFCN_COCO_EVALUATION computes and evaluates a set of detections
%   for a given R-FCN detector on COCO.
%
%   RFCN_COCO_EVALUATION(..'name', value) accepts the following 
%   options:
%
%   `net` :: []
%    The `autonn` network object to be evaluated.  If not supplied, a network
%    will be loaded instead by name from the detector zoo.
%
%   `gpus` :: []
%    If provided, the gpu ids to be used for processing.
%
%   `dataRoot` :: fullfile(vl_rootnn, 'data/datasets')
%    The path to the directory containing the coco data
%
%   `nms` :: 'cpu'
%    NMS can be run on either the gpu if the dependency has been installed
%    (see README.md for details), or on the cpu (slower).
%
%   `modelName` :: 'rfcn-res101-coco'
%    The name of the detector to be evaluated (used to generate output
%    file names, caches etc.)
%
%   `refreshCache` :: false
%    If true, overwrite previous predictions by any detector sharing the 
%    same model name, otherwise, load results directly from cache.
%
%   `useMiniVal` :: false
%    If true (and the testset is set to `val`), evaluate on the `mini-val` 
%    subsection of the coco data, rather than the full validation set.  This
%    setting is useful for evaluating models trained on coco-trainval135k.
%
%   `year` :: 2014
%    Select year of coco data to run evaluation on.
%
% Copyright (C) 2017 Samuel Albanie 
% All rights reserved.

  opts.net = [] ;
  opts.expDir = '' ; % preserve interface
  opts.gpus = 4 ;
  opts.nms = 'gpu' ;  
  opts.testset = 'val' ;
  opts.useMiniVal = false ; 
  opts.refreshCache = true ;
  opts.modelName = 'rfcn-res101-coco' ;
  opts.year = 2014 ; % 2015 only contains test instances
  opts.dataRoot = fullfile(vl_rootnn, 'data/datasets') ;
  opts.debug = false ; % used for visualisation purposes
  opts.benchmarkTiming = false ; 
  opts = vl_argparse(opts, varargin) ;

  % load network and convert to autonn
  if isempty(opts.net), opts.net = rfcn_zoo(opts.modelName) ; end
  layers = Layer.fromDagNN(opts.net, @rfcn_autonn_custom_fn) ;
  net = Net(layers{:}) ;

  net = configureNet(net, opts) ; % configure NMS/varNames if required

  % evaluation options
  opts.prefetch = true ; 
  opts.fixedSizeInputs = false ;

  % configure batch opts
  batchOpts.scale = 600 ;
  batchOpts.maxScale = 1000 ;
  batchOpts.use_vl_imreadjpeg = 1 ; 
  batchOpts.batchSize = numel(opts.gpus) * 1 ;
  batchOpts.numThreads = numel(opts.gpus) * 4 ;
  batchOpts.averageImage = net.meta.normalization.averageImage ;

  % cache configuration 
  cacheOpts.refreshCache = opts.refreshCache ;

  % configure model options
  modelOpts.maxPreds = 300 ; 
  modelOpts.numClasses = 81 ;
  modelOpts.nmsThresh = 0.3 ;
  modelOpts.confThresh = 0.05 ;
  modelOpts.maxPredsPerImage = 100 ; 
  modelOpts.classAgnosticReg = true ; 
  modelOpts.get_eval_batch = @faster_rcnn_eval_get_batch ; % re-use

  % configure dataset options
  dataOpts.name = 'coco' ;
  dataOpts.year = opts.year ;
  dataOpts.getImdb = @getCocoImdb ;
  dataOpts.resultsFormat = 'minWH' ; 
  dataOpts.dataRoot = opts.dataRoot ;
  dataOpts.eval_func = @coco_eval_func ;
  dataOpts.displayResults = @displayCocoResults ;
  dataOpts.configureImdbOpts = @configureImdbOpts ;
  % minival annotations are only used for 2014
  dataOpts.miniValPath = fullfile(opts.dataRoot, ... 
                            'mscoco/annotations/instances_minival2014.json') ;
  dataOpts.labelMapFile = fullfile(vl_rootnn, ... 
                                 'contrib/mcnFasterRCNN/coco/label_map.txt') ;

  % select imdb based on year
  imdbName = sprintf('imdb%d.mat', opts.year) ;
  dataOpts.imdbPath = fullfile(vl_rootnn, 'data/coco/standard_imdb/', imdbName) ;
  imdbOpts.conserveSpace = true ; imdbOpts.includeTest = true  ;

  % configure paths
  expDir = fullfile(vl_rootnn, 'data/evaluations', dataOpts.name, opts.modelName) ;
  resultsFile = sprintf('%s-%s-results.mat', opts.modelName, opts.testset) ;
  evalCacheDir = fullfile(expDir, 'eval_cache') ;
  cacheOpts.resultsCache = fullfile(evalCacheDir, resultsFile) ;
  cacheOpts.evalCacheDir = evalCacheDir ;

  if ~exist(evalCacheDir, 'dir')
      mkdir(evalCacheDir) ;
      mkdir(fullfile(evalCacheDir, 'cache')) ;
  end

  % configure meta options
  opts.dataOpts = dataOpts ;
  opts.modelOpts = modelOpts ;
  opts.batchOpts = batchOpts ;
  opts.cacheOpts = cacheOpts ;
  opts.imdbOpts = imdbOpts ;

  % re-use the faster-rcnn eval code
  results = faster_rcnn_evaluation(expDir, net, opts) ;

% -----------------------------------------------------------
function [opts, imdb] = configureImdbOpts(~, opts, imdb)
% -----------------------------------------------------------
% split images according to the popular "trainval35k" split commonly
% used for ablation experiments
  switch opts.dataOpts.year
    case 2014
      if opts.useMiniVal
        annotations = gason(fileread(opts.dataOpts.miniValPath)) ;
        miniValIds = [annotations.images.id] ;
        fullValIms = find(imdb.images.set == 2) ;
        keep = ismember(imdb.images.id(fullValIms), miniValIds) ;
        imdb.images.set(fullValIms(~keep)) = 1 ;
      end
    case 2015
      if opts.benchmarkTiming 
        keep = 200 ; testIdx = find(imdb.images.set == 4) ;
        imdb.images.set(testIdx(keep+1:end)) = 5 ;
      end
  end

% ------------------------------------------------------------------
function aps = coco_eval_func(~, decoded, imdb, opts)
% ------------------------------------------------------------------
  aps = {} ; % maintain interface
  numClasses = numel(imdb.meta.classes) - 1 ;  % exclude background
  image_id = vertcat(decoded.imageIds{:}) ;
  category_id = arrayfun(@(x) {x*ones(1,numel(decoded.imageIds{x}))}, 1:numClasses) ;
  category_id = [category_id{:}]' ;

  [labelMap,~] = getCocoLabelMap('labelMapFile', opts.dataOpts.labelMapFile) ;
  category_id = arrayfun(@(x) labelMap(x), category_id) ;
  bbox = vertcat(decoded.bboxes{:}) ; score = vertcat(decoded.scores{:}) ;
  table_ = table(image_id, category_id, bbox, score) ; res = table2struct(table_) ;

  % encode as json (this may take a little while...., the gason func adds to storage)
  cocoJason = jsonencode(res) ;  template = 'detections_%s%d_%s%d.mat' ; 
  resFile = sprintf(template, opts.testset, opts.dataOpts.year, opts.modelName) ; 
  resPath = fullfile(opts.cacheOpts.evalCacheDir, resFile) ;
  fid = fopen(resPath, 'w') ; fprintf(fid, cocoJason) ; fclose(fid) ;
  fprintf('detection results have been saved to %s\n', resPath) ;

  if strcmp(opts.testset, 'val')
    %% initialize COCO ground truth api
    if opts.useMiniVal, mini = 'mini' ; else, mini = '' ; end 
    dataType = sprintf('%s%s%d', mini, opts.testset, opts.dataOpts.year) ;
    dataDir = fullfile(opts.dataOpts.dataRoot, 'mscoco') ;
    annFile = sprintf('%s/annotations/instances_%s.json',dataDir,dataType) ;
    cocoGt = CocoApi(annFile) ; % load ground truth
    cocoDt = cocoGt.loadRes(resPath) ; % load detections
    cocoEval = CocoEval(cocoGt, cocoDt, 'bbox') ;
    imgIds = sort(cocoGt.getImgIds());  cocoEval.params.imgIds = imgIds ;
    cocoEval.evaluate() ; cocoEval.accumulate() ; cocoEval.summarize() ;
    aps = cocoEval.eval ;
    if opts.debug, visualizeRes(res, cocoEval, imdb) ; end % for debugging
  end

% ------------------------------------------------------------------------
function visualizeRes(res, cocoEval,imdb)
% ------------------------------------------------------------------------
  res = res([res.score] > 0.6) ; % restrict to confident preds
  sampleSize = 100 ; sample = randi(numel(res), 1, sampleSize) ;
  [revLabelMap,labels] = getCocoLabelMap('reverse', true) ;
  for ii = 1:numel(sample)
    rec = res(sample(ii)) ;
    id = find(imdb.images.id == rec.image_id) ;
    imName = imdb.images.name{id} ; template = imdb.images.paths{id} ;
    imPath = sprintf(template, imName) ; im = single(imread(imPath)) ;
    label = labels{revLabelMap(rec.category_id)} ;
    box = rec.bbox ; score = rec.score ;  
    im_ids = [cocoEval.cocoGt.data.annotations.image_id] ;
    gt = cocoEval.cocoGt.data.annotations(im_ids == rec.image_id) ;
    drawCocoBoxes(im, box, score, label, 'format', 'MinWH', 'gt', gt) ;
  end

% ---------------------------------------------------------------------------
function displayCocoResults(~, aps, opts)
% ---------------------------------------------------------------------------
  if ~strcmp(opts.testset, 'val'), return ; end 
  [~,labels] = getCocoLabelMap('labelMapFile', opts.dataOpts.labelMapFile) ;
  iou=0.5:0.05:0.95 ; areas='asml' ; mdets=[1 10 100] ;
  t = 1 ; a = 1 ; m = 3 ; 
  fprintf('AP @0.5 IoU, areas=%s, max dets/img=%d\n',iou(t),areas(a),mdets(m)) ;
  for ii = 1:numel(aps.params.catIds)
    s = aps.precision(1,:,ii,a,m) ; s=mean(s(s>=0)) * 100 ; 
    fprintf('%.1f: %s \n', s, labels{ii}) ;
  end

% ------------------------------------
function net = configureNet(net, opts)
% ------------------------------------
%CONFIGURENET - update net to optimise performance and fix names
%  CONFIGURENET(NET, OPTS) updates NMS to run on GPU (if specified),
%  and ensures that variables name are consistent after converting 
%  to autonn

  dnet = Layer.fromCompiledNet(net) ; % decompile
  cls_head = dnet{1} ; bbox_head = dnet{2} ;

  % update NMS
  prev = cls_head.find(@vl_nnproposalrpn, 1) ;
  in = [ prev.inputs {'nms', opts.nms}] ; % update nms option
  proposals = Layer.create(@vl_nnproposalrpn, in) ;
  proposals.name = prev.name ;
  cls_head.find(@vl_nnpsroipool, 1).inputs{2} = proposals ; % reattach 
  bbox_head.find(@vl_nnpsroipool, 1).inputs{2} = proposals ; % reattach 

  % use standard output variable names for detection
  cls_head.name = 'cls_prob' ; bbox_head.name = 'bbox_pred' ;
  net = Net(cls_head, bbox_head) ; % recompile
