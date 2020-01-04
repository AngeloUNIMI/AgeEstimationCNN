%%
clc
close all
clear variables


%--------------------------------------------------------------------------
%params
plotta = 1;
savefile = 1;
stepDisp = 50;
numFeatures = 8192; %8192


%--------------------------------------------------------------------------
%paths
dirMatConvNet = './matconvnet-1.0-beta17/';

%addpath
addpath(genpath(dirMatConvNet));
addpath(genpath('./util'));


%--------------------------------------------------------------------------
%img directory
dbName = 'AgeDB';
dirImgs = ['.\' dbName '\'];
ext = 'jpg';

%dir results
dirResults = ['./results/' dbName '/'];
mkdir_pers(dirResults, savefile);

%pre-trained CNN
netFilename_vggface = './nets/vgg-face.mat';
netFilename_alexnet = './nets/imagenet-caffe-alex.mat';

if exist(netFilename_vggface, 'file') ~= 2
    fprintf(1, 'Downloading vgg-face...\n');
    out = websave(netFilename_vggface, 'http://www.vlfeat.org/matconvnet/models/vgg-face.mat');
end %if exist

if exist(netFilename_alexnet, 'file') ~= 2
    fprintf(1, 'Downloading imagenet-caffe-alex...\n');
    out = websave(netFilename_alexnet, 'http://www.vlfeat.org/matconvnet/models/imagenet-caffe-alex.mat');
end %if exist
    


%--------------------------------------------------------------------------
%setup matconvnet
% run('vl_compilenn');
run('vl_setupnn');


%--------------------------------------------------------------------------
%file list
filesImg = dir([dirImgs '*.' ext]);

%load net
fprintf(1, 'Loading nets...\n');
net_vggface = load(netFilename_vggface);
net_vggface = vl_simplenn_tidy(net_vggface);
net_alexnet = load(netFilename_alexnet);
net_alexnet = vl_simplenn_tidy(net_alexnet);

%init feature vector
featAll = -1 .* ones(numFeatures, numel(filesImg));


%if not exist, compute

%file results
fileLabels = [dirResults 'labels.mat'];
fileFeatures = [dirResults 'features.mat'];
filePca = [dirResults 'pca.mat'];
if exist(fileLabels, 'file') ~= 2 || exist(fileFeatures, 'file') ~= 2 || exist(fileFeatures, 'file') ~= 2
    
    fprintf(1, 'Computing features and PCA coefficients...\n');
    
    %loop on images
    labels = zeros(numel(filesImg), 1);
    
    for i = 1 : numel(filesImg)

        % for i = 1 : numFeatures
        
        %get label
        [C, ind] = strsplit(filesImg(i).name, '_');
        labels(i) = str2double(C{3});
                
        %display
        if mod(i, stepDisp) == 0
            fprintf(1, ['\tFile n. ' num2str(i) '/' num2str(numel(filesImg)) ': ' filesImg(i).name '\n']);
        end %if mod
        
        %load image
        imFilename = [dirImgs filesImg(i).name];
        im = imread(imFilename);
        im_ = single(im);
        
        %VGG-Face
        %We used the second fully connected layer for feature extraction,
        %obtaining 4096 dimensional feature sets.
        im_ = imresize(im_, net_vggface.meta.normalization.imageSize(1:2)) ;
        im_ = bsxfun(@minus,im_,net_vggface.meta.normalization.averageImage) ;
        res_vggface = vl_simplenn(net_vggface, im_);
        feat_vggface = squeeze(res_vggface(34).x);
        
        %AlexNet
        %We used the second fully connected layer for feature extraction,
        %obtaining 4096 dimensional feature sets.
        im_ = imresize(im_, net_alexnet.meta.normalization.imageSize(1:2)) ;
        im_ = im_ - net_alexnet.meta.normalization.averageImage ;
        res_alexnet = vl_simplenn(net_alexnet, im_);
        feat_alexnet = squeeze(res_alexnet(18).x);
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %ONLY FOR TESTING - CROP
        %     feat_vggface = feat_vggface(1 : numFeatures/2);
        %     feat_alexnet = feat_alexnet(1 : numFeatures/2);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        %combine feature vector
        feat = [feat_vggface; feat_alexnet];
        
        %add to global feature vector
        featAll(:, i) = feat;
        
        clear res_vggface feat_vggface res_alexnet feat_alexnet feat
        
        %pause
        
    end %for i
    
    %
    save(fileLabels, 'labels');
    save(fileFeatures, 'featAll');
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %ONLY FOR TESTING - CROP
    %crop feature vector
    % featAll(:, i+1 : end) = [];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    %--------------------------------------------------------------------------
    %coeff = pca(X) returns the principal component coefficients,
    %also known as loadings, for the n-by-p data matrix X.
    %Rows of X correspond to observations and columns correspond to variables.
    featAll = featAll';
    
    %apply PCA
    %De-mean
    X = bsxfun(@minus,featAll, mean(featAll));
    
    %Do the PCA
    [coeff,score,latent] = pca(X);
    
    % %Calculate eigenvalues and eigenvectors of the covariance matrix
    covarianceMatrix = cov(X);
    [V,D] = eig(covarianceMatrix);
    dataInPrincipalComponentSpace = X*coeff;
    
    %select
    
    %save pca transformation
    save(filePca, 'coeff', 'dataInPrincipalComponentSpace');
    
else %if exist
    
    fprintf(1, 'Features and PCA coefficients already computed...\n');
    
end %if exist

pause








%%
close all
clear variables


%--------------------------------------------------------------------------
%img directory
dbName = 'AgeDB';
dirImgs = ['.\' dbName '\'];

%dir results
dirResults = ['./results/' dbName '/'];


%--------------------------------------------------------------------------
%load
fprintf(1, 'Loading features...\n');
load([dirResults 'features.mat']);
load([dirResults 'pca.mat']);
load([dirResults 'labels.mat']);

%check if exist
fileNet = [dirResults 'PCA30DB_Results_CONCAT.mat'];

if exist(fileNet, 'file') ~= 2
    
    fprintf(1, 'Training FFNN...\n');

%--------------------------------------------------------------------------
allFeatures = double(dataInPrincipalComponentSpace(:, 1:30));
allLabels = labels;
% X = isnan(trainLabels1);
% trainLabels = trainLabels1(~X);
% trainFeatures = trainFeatures1(:,~X);
% [q,q1] = size(trainFeatures);
% trainFeatures = rand(q,q1);
trainFeatureName = 'agedb_concatenated';

fid = fopen([dirResults 'PCA30DB_Results_CONCAT.csv'],'w');
fprintf(fid,'\n');
fprintf(fid, 'TrainFeature TestFeature Classname LearnAlgo #N MAE  Epochs\n\n');
TR = [];
TC = [];

%10-fold
kF = 5;
indices = crossvalind('kfold', numel(allLabels), kF);


%fprintf(1, 'Training FFNN...\n');

Neurons = [15]; %15? 20?

for ii = 1:length(Neurons)
    numLayers = Neurons(ii);
    
    %for iter = 1 : kF
    for iter = 1
        
        testIndices = (indices == iter);
        trainIndices = ~testIndices;
        trainFeatures = allFeatures(trainIndices, :);
        testFeatures = allFeatures(testIndices, :);
        trainLabels = allLabels(trainIndices);
        testLabels = allLabels(testIndices);
        
        % train the final neural network
        net = feedforwardnet(numLayers,'trainscg');
        %     net.divideParam.trainRatio = 1;
        net.divideParam.trainRatio = 0.9;
        net.divideParam.testRatio  = 0;
        net.divideParam.valRatio   = 0.1;
        net.trainParam.showWindow = false;
        net.trainParam.showCommandLine = true;
        net.trainParam.epochs = 2000;
        net.trainParam.goal = 0.000001;
        neuronTopology{1} = 'tansig';
        
        for iL = 1: size(numLayers,2)
            net.layers{iL}.transferFcn = neuronTopology{iL};
        end
        
        % train a neural network
        [net,tr,Y,E] = train(net, trainFeatures', trainLabels', 'useGPU', 'only');
        % test
        testResultKDouble = net(testFeatures');
        testResultK = (testResultKDouble);
        %     indMin = find(testResultK < min(trainLabels));
        %     testResultK(indMin) = min(trainLabels);
        %     indMax = find(testResultK > max(trainLabels));
        %     testResultK(indMax) = max(trainLabels);
        %
        % concatenate the results and ground truth for each fold
        %     TR = [TR, testLabels];
        %     TC = [TC, testResultK];
        
        % results in each iteration of FFNN
        mae = sum(abs(testLabels'-testResultK))/length(testLabels);
        %     CMStructure = confusionmat(TR,TC);
        
        %  SAVE THE NUMERICAL RESULTS
        LearningMethod = tr.trainFcn;
        Epochs = net.trainParam.epochs;
        numOfNeurons = numLayers;
        className = 'FFNN';
        
        
        fprintf(fid, '%s\t', className);
        fprintf(fid, '%s\t', LearningMethod);
        fprintf(fid, '%3d\t', numOfNeurons);
        fprintf(fid, '%f\t', mae);
        fprintf(fid, '%3d\t', Epochs);
        fprintf(fid,'\n');
        
        MAE(1,iter) = mae;
        actualAge(:,iter) = testLabels;
        estimatedAge(:,iter) = testResultK;
        hiddenNeurons(1,iter) = numOfNeurons;
        trainedNET{iter} = net;
        trainedTR{iter} = tr;
        trainedEstimatedAge(:,iter) = Y;
        trainedActualAge(:,iter) = trainLabels;
        
    end
    
    % results of average FFNN
    actualAge = mean(actualAge,2);
    estimatedAge = mean(estimatedAge,2);
    
    mean_MAE = mean(MAE);
    std_MAE = std(MAE);
    CMStructure = confusionmat(actualAge,estimatedAge);
    
    Results_features{ii,1} =  mean_MAE;
    Results_features{ii,2} = trainedActualAge;
    Results_features{ii,3} = trainedEstimatedAge;
    Results_features{ii,4} = MAE;
    Results_features{ii,5} = '';
    Results_features{ii,6} = CMStructure;
    Results_features{ii,7} = actualAge;
    Results_features{ii,8} = estimatedAge;
end

fclose(fid);

save(fileNet, 'Results_features', 'net');

else %if exist
    
    fprintf(1, 'FFNN already trained...\n');
    
end %if exist

pause










%%
close all
clear variables


%--------------------------------------------------------------------------
%params
plotta = 1;
savefile = 1;
stepDisp = 50;
numFeatures = 100;
numEigPCA = 30;

%--------------------------------------------------------------------------
%paths
dirMatConvNet = './matconvnet-1.0-beta17/';

%addpath
addpath(genpath(dirMatConvNet));
addpath(genpath('./util'));


%--------------------------------------------------------------------------
%img directory
dbName = 'AgeDB';
dirImgs = ['.\' dbName '\'];
ext = 'jpg';

%dir results
dirResults = ['./results/' dbName '/'];
mkdir_pers(dirResults, savefile);

%pre-trained CNN
netFilename_vggface = './nets/vgg-face.mat';
netFilename_alexnet = './nets/imagenet-caffe-alex';

%pca transformation
filenamePCA = [dirResults 'pca.mat'];

%preTrained FFNN
filenameFFNN = [dirResults 'PCA30DB_Results_CONCAT.mat'];


%--------------------------------------------------------------------------
%setup matconvnet
% run('vl_compilenn');
run('vl_setupnn');


%--------------------------------------------------------------------------
%file list
filesImg = dir([dirImgs '*.' ext]);

%load net
fprintf(1, 'Loading nets...\n');
net_vggface = load(netFilename_vggface);
net_vggface = vl_simplenn_tidy(net_vggface);
net_alexnet = load(netFilename_alexnet);
net_alexnet = vl_simplenn_tidy(net_alexnet);

%load pca
load(filenamePCA);

%load ffnn
load(filenameFFNN);

%init
result(numel(filesImg)).name = '';
result(numel(filesImg)).age = 0;


%--------------------------------------------------------------------------
%loop on images
for i = 1 : numel(filesImg)
    
    %display
    fprintf(1, ['File n. ' num2str(i) '/' num2str(numel(filesImg)) ': ' filesImg(i).name '\n']);
    
    %load image
    imFilename = [dirImgs filesImg(i).name];
    
    %get label
    [C, ind] = strsplit(imFilename, '_');
    realAge = str2double(C{3});
    
    %imFilename = 'D:\Google Drive\Shared AIEIE\Immagini\Avatars\Angelo Genovese - Photo.jpg';
    
    im = imread(imFilename);
    im_ = single(im);
    
    %VGG-Face
    %We used the second fully connected layer for feature extraction,
    %obtaining 4096 dimensional feature sets.
    im_ = imresize(im_, net_vggface.meta.normalization.imageSize(1:2)) ;
    im_ = bsxfun(@minus,im_,net_vggface.meta.normalization.averageImage) ;
    res_vggface = vl_simplenn(net_vggface, im_);
    feat_vggface = squeeze(res_vggface(34).x);
        
    %AlexNet
    %We used the second fully connected layer for feature extraction,
    %obtaining 4096 dimensional feature sets.    
    im_ = imresize(im_, net_alexnet.meta.normalization.imageSize(1:2)) ;
    im_ = im_ - net_alexnet.meta.normalization.averageImage ;
    res_alexnet = vl_simplenn(net_alexnet, im_);
    feat_alexnet = squeeze(res_alexnet(18).x);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %ONLY FOR TESTING - CROP
    %feat_vggface = feat_vggface(1 : numFeatures/2);
    %feat_alexnet = feat_alexnet(1 : numFeatures/2);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    
    
    %combine feature vector
    feat = [feat_vggface; feat_alexnet];
    
    clear res_vggface feat_vggface res_alexnet feat_alexnet

    %de - mean
    feat = bsxfun(@minus, feat, mean(feat));
    
    %apply pca
    featPCA = feat' * coeff;
    
    %select best features
    featPCA = featPCA(1 : numEigPCA);
    
    %apply FFNN
    age = net(featPCA');
    
    %round
    age = round(age);
    
    %display
    fprintf(1, ['\tReal age: ' num2str(realAge) '; Est. age: ' num2str(age) '\n']);

    %save results
    result(i).name = filesImg(i).name;
    result(i).age = age;
    
    %plot
    figure(1)
    imshow(imresize(im, [500 500]), [])
    title([filesImg(i).name, '; Real age: ' num2str(realAge) '; Est. age: ' num2str(age)], 'Interpreter', 'none')
    
    pause
    
end %for i








