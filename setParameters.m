function param = setParameters()

param.numTrainImages = 1545;
param.numTestImages = 396;
param.decRange = 1996:8:2019;

param.initPatchSize = 50;
param.numPatchesPerScale = 4;
param.scales = 0.5.^([0 1 2]);
param.sBin = 10;
param.normalizeFeats = 1;
param.numTopMatches = 50;
param.numClustersPerDecade = 10;

param.normalizeDet = 0;
param.patchSize = 70;
param.scalesDet = (sqrt(2).^[0 1 2 3 4]).^(-1);

% debug flag
param.DEBUG_FLAG = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHANGE THESE DIRECTORIES APPROPRIATELY
% input directories
param.trainimgdir = ['./train/'];
param.testimgdir = ['./testt/'];
param.bgDir = ['./randBgImgs/'];

% output directories
param.sampledir = ['./randomSamples/'];
param.matchdir = ['./nearestNeighbors/'];
param.clusterdir = ['./clusters/'];
param.detectordir = ['./detectors/'];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~exist(param.sampledir,'dir'); mkdir(param.sampledir); end;
if ~exist(param.matchdir,'dir'); mkdir(param.matchdir); end;
if ~exist(param.clusterdir,'dir'); mkdir(param.clusterdir); end;
if ~exist(param.detectordir,'dir'); mkdir(param.detectordir); end;

