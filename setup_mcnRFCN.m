function setup_mcnRFCN()
%SETUP_MCNRFCN Sets up mcnRFCN, by adding its folders 
% to the Matlab path

  root = fileparts(mfilename('fullpath')) ;
  addpath(root) ;
  addpath(root, [root '/matlab'], [root '/pascal'], [root '/core']) ;
  addpath(root, [root '/misc']) ;
  addpath([vl_rootnn '/examples/fast_rcnn/bbox_functions']) ;
