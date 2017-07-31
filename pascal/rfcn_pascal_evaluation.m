function [aps, speed] = rfcn_pascal_evaluation(varargin)
%RFCN_PASCAL_EVALUATION Evaluate an R-FCN model on VOC 2007
%   RFCN_PASCAL_EVALUATION computes and evaluates a set of detections
%   for a given R-FCN detector on the Pascal VOC 2007 test set.
%
%   RFCN_PASCAL_EVALUATION(..'name', value) accepts the following 
%   options:
%
%   `net` :: []
%    The `autonn` network object to be evaluated.  If not supplied, a network
%    will be loaded instead by name from the detector zoo.
%
%   `gpus` :: []
%    If provided, the gpu ids to be used for processing.
%
%   `evalVersion` :: 'fast'
%    The type of VOC evaluation code to be run.  The options are 'official', 
%    which runs the original (slow) pascal evaluation code, or 'fast', which
%    runs an optimised version which is useful during development.
%
%   `dataRoot` :: fullfile(vl_rootnn, 'data/datasets')
%    The path to the directory containing the pascal data
%
%   `nms` :: 'cpu'
%    NMS can be run on either the gpu if the dependency has been installed
%    (see README.md for details), or on the cpu (slower).
%
%   `modelName` :: 'rfcn-res50-pascal'
%    The name of the detector to be evaluated (used to generate output
%    file names, caches etc.)
%
%   `refreshCache` :: false
%    If true, overwrite previous predictions by any detector sharing the 
%    same model name, otherwise, load results directly from cache.
%
% Copyright (C) 2017 Samuel Albanie 
% All rights reserved.

  opts.net = [] ;
  opts.gpus = 4 ;
  opts.nms = 'gpu' ;  
  opts.refreshCache = true ;
  opts.evalVersion = 'fast' ;
  opts.modelName = 'rfcn-res50-pascal' ;
  opts.dataRoot = fullfile(vl_rootnn, 'data/datasets') ;
  opts = vl_argparse(opts, varargin) ;

  % if needed, load network and convert to autonn
  if isempty(opts.net)
    opts.net = rfcn_zoo(opts.modelName) ; 
    layers = Layer.fromDagNN(opts.net, @rfcn_autonn_custom_fn) ;
    net = Net(layers{:}) ;
  else
    net = opts.net ;
  end

  net = configureNet(net, opts) ; % configure NMS/varNames if required

  % evaluation options
  opts.testset = 'test' ; 
  opts.prefetch = false ;

  % configure batch opts
  batchOpts.scale = 600 ;
  batchOpts.maxScale = 1000 ;
  batchOpts.use_vl_imreadjpeg = 1 ; 
  batchOpts.batchSize = numel(opts.gpus) * 1 ;
  batchOpts.numThreads = numel(opts.gpus) * 4 ;
  batchOpts.averageImage = net.meta.normalization.averageImage ;

  % configure model options
  modelOpts.maxPreds = 300 ; % the maximum number of total preds/img
  modelOpts.nmsThresh = 0.3 ;
  modelOpts.numClasses = 21 ; % includes background for pascal
  modelOpts.confThresh = 0.05 ;
  modelOpts.maxPredsPerImage = 100 ; 
  modelOpts.classAgnosticReg = true ; 
  modelOpts.get_eval_batch = @faster_rcnn_eval_get_batch ; % re-use function

  % configure dataset options
  dataOpts.name = 'pascal' ;
  dataOpts.resultsFormat = 'minMax' ; 
  dataOpts.getImdb = @getPascalImdb ;
  dataOpts.dataRoot = opts.dataRoot ;
  dataOpts.eval_func = @pascal_eval_func ;
  dataOpts.evalVersion = opts.evalVersion ;
  dataOpts.displayResults = @displayPascalResults ;
  dataOpts.configureImdbOpts = @configureImdbOpts ;
  dataOpts.imdbPath = fullfile(vl_rootnn, 'data/pascal/standard_imdb/imdb.mat') ;

  % configure paths and cache 
  expDir = fullfile(vl_rootnn, 'data/evaluations', dataOpts.name, opts.modelName) ;
  resultsFile = sprintf('%s-%s-results.mat', opts.modelName, opts.testset) ;
  evalCacheDir = fullfile(expDir, 'eval_cache') ;
  cacheOpts.resultsCache = fullfile(evalCacheDir, resultsFile) ;
  cacheOpts.evalCacheDir = evalCacheDir ;
  cacheOpts.refreshCache = opts.refreshCache ;
  if ~exist(evalCacheDir, 'dir'), mkdir(evalCacheDir) ; end

  % configure meta options
  opts.dataOpts = dataOpts ;
  opts.modelOpts = modelOpts ;
  opts.batchOpts = batchOpts ;
  opts.cacheOpts = cacheOpts ;

  % make use of the faster-rcnn driver
  faster_rcnn_evaluation(expDir, net, opts) ;

% ------------------------------------------------------------------
function aps = pascal_eval_func(modelName, decodedPreds, imdb, opts)
% ------------------------------------------------------------------
  fprintf('evaluating %s \n', modelName) ;
  numClasses = numel(imdb.meta.classes) - 1 ;  % exclude background
  aps = zeros(numClasses, 1) ;

  for c = 1:numClasses
      className = imdb.meta.classes{c + 1} ; % offset for background
      results = eval_voc(className, ...
                         decodedPreds.imageIds{c}, ...
                         decodedPreds.bboxes{c}, ...
                         decodedPreds.scores{c}, ...
                         opts.dataOpts.VOCopts, ...
                         'evalVersion', opts.dataOpts.evalVersion) ;
      fprintf('%s %.1\n', className, 100 * results.ap_auc) ;
      aps(c) = results.ap_auc ; 
  end
  save(opts.cacheOpts.resultsCache, 'aps') ;

% -----------------------------------------------------------
function [opts, imdb] = configureImdbOpts(expDir, opts, imdb)
% -----------------------------------------------------------
% configure VOC options 
% (must be done after the imdb is in place since evaluation
% paths are set relative to data locations)

  BENCHMARK = 0 ;
  if BENCHMARK  % benchmark
    keep = 100 ; testIdx = find(imdb.images.set == 3) ;
    imdb.images.set(testIdx(keep+1:end)) = 4 ;
  end
  opts.dataOpts = configureVOC(expDir, opts.dataOpts, 'test') ;

%-----------------------------------------------------------
function dataOpts = configureVOC(expDir, dataOpts, testset) 
%-----------------------------------------------------------
% LOADPASCALOPTS Load the pascal VOC database options
%
% NOTE: The Pascal VOC dataset has a number of directories 
% and attributes. The paths to these directories are 
% set using the VOCdevkit code. The VOCdevkit initialization 
% code assumes it is being run from the devkit root folder, 
% so we make a note of our current directory, change to the 
% devkit root, initialize the pascal options and then change
% back into our original directory 

  VOCRoot = fullfile(dataOpts.dataRoot, 'VOCdevkit2007') ;
  VOCopts.devkitCode = fullfile(VOCRoot, 'VOCcode') ;

  % check the existence of the required folders
  assert(logical(exist(VOCRoot, 'dir')), 'VOC root directory not found') ;
  assert(logical(exist(VOCopts.devkitCode, 'dir')), 'devkit code not found') ;

  currentDir = pwd ;
  cd(VOCRoot) ;
  addpath(VOCopts.devkitCode) ;

  % VOCinit loads database options into a variable called VOCopts
  VOCinit ; 

  dataDir = fullfile(VOCRoot, '2007') ;
  VOCopts.localdir = fullfile(dataDir, 'local') ;
  VOCopts.imgsetpath = fullfile(dataDir, 'ImageSets/Main/%s.txt') ;
  VOCopts.imgpath = fullfile(dataDir, 'ImageSets/Main/%s.txt') ;
  VOCopts.annopath = fullfile(dataDir, 'Annotations/%s.xml') ;
  VOCopts.cacheDir = fullfile(expDir, '2007/Results/Cache') ;
  VOCopts.drawAPCurve = false ;
  VOCopts.testset = testset ;
  detDir = fullfile(expDir, 'VOCdetections') ;

  % create detection and cache directories if required
  requiredDirs = {VOCopts.localdir, VOCopts.cacheDir, detDir} ;
  for i = 1:numel(requiredDirs)
      reqDir = requiredDirs{i} ;
      if ~exist(reqDir, 'dir') 
          mkdir(reqDir) ;
      end
  end

  VOCopts.detrespath = fullfile(detDir, sprintf('%%s_det_%s_%%s.txt', 'test')) ;
  dataOpts.VOCopts = VOCopts ;

  % return to original directory
  cd(currentDir) ;

% ---------------------------------------------------------------------------
function displayPascalResults(modelName, aps, opts)
% ---------------------------------------------------------------------------
  fprintf('============\n') ;
  fprintf(sprintf('%s set performance of %s:', opts.testset, modelName)) ;
  fprintf('%.1f (mean ap) \n', 100 * mean(aps)) ;
  fprintf('============\n') ;

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

  % use standard output variable names for detection
  cls_head.name = 'cls_prob' ;
  bbox_head.name = 'bbox_pred' ;

  net = Net(cls_head, bbox_head) ; % recompile
