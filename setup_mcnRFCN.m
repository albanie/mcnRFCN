function setup_mcnRFCN()
%SETUP_MCNRFCN Sets up mcnRFCN, by adding its folders 
% to the Matlab path, as well as setting up mcnFasterRCNN as a dependency

  root = fileparts(mfilename('fullpath')) ;
  addpath(root) ;
  addpath(root, [root '/matlab'], [root '/pascal'], [root '/core']) ;
  addpath( [root '/matlab/mex'], [root '/misc'], [root '/coco']) ;
  run(fullfile(vl_rootnn, 'contrib/mcnFasterRCNN/setup_mcnFasterRCNN')) ;
