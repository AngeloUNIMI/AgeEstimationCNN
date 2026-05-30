clc
close all
clear variables


%--------------------------------------------------------------------------
%params
plotta = 1;
savefile = 1;
numEigPCA = 30;

%path
dirMatConvNet = './matconvnet-1.0-beta17/';
%addpath
addpath(genpath(dirMatConvNet));
addpath(genpath('./util'));

%dir
dirNets = './nets/';
dbName = 'AgeDB';
dirResults = ['./results/' dbName '/'];

%pre-trained CNN
netFilename_vggface = [dirNets 'vgg-face.mat'];
netFilename_alexnet = [dirNets 'imagenet-caffe-alex.mat'];

if exist(netFilename_vggface, 'file') ~= 2
    fprintf(1, 'Downloading vgg-face...\n');
    out = websave(netFilename_vggface, 'http://www.vlfeat.org/matconvnet/models/vgg-face.mat');
end %if exist

if exist(netFilename_alexnet, 'file') ~= 2
    fprintf(1, 'Downloading imagenet-caffe-alex...\n');
    out = websave(netFilename_alexnet, 'http://www.vlfeat.org/matconvnet/models/imagenet-caffe-alex.mat');
end %if exist

%pca transformation
filenamePCA = [dirResults 'pca.mat'];

%preTrained FFNN
filenameFFNN = [dirResults 'PCA30DB_Results_CONCAT.mat'];


%--------------------------------------------------------------------------
%setup matconvnet
% run('vl_compilenn');
run('vl_setupnn');

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


%--------------------------------------------------------------------------
%face detector
faceDetector = vision.CascadeObjectDetector('FrontalFaceCART');


%--------------------------------------------------------------------------
%cam
cam = webcam(1);
cam.Resolution = cam.AvailableResolutions{end};


%--------------------------------------------------------------------------
%loop

if plotta
    figure(1)
end %if plotta

while (1)
    
    im_orig = snapshot(cam);
    
    %detect face
    bbox = step(faceDetector, im_orig);
    
    if numel(bbox) == 0
        fprintf(1, 'Cannot detect face...\n');
        pause(1)
        continue
    end %f numel
    
    imPlot = im_orig;
    
    for b = 1 : size(bbox, 1)
        bbox_p = bbox(b, :);
        
        im_crop = imcrop(im_orig, bbox_p);
        
        im_ = single(im_crop);
        
        %VGG-Face
        %We used the second fully connected layer for feature extraction,
        %obtaining 4096 dimensional feature sets.
        im1_ = imresize(im_, net_vggface.meta.normalization.imageSize(1:2)) ;
        im1_ = bsxfun(@minus,im1_,net_vggface.meta.normalization.averageImage) ;
        res_vggface = vl_simplenn(net_vggface, im1_);
        feat_vggface = squeeze(res_vggface(34).x);
        
        %AlexNet
        %We used the second fully connected layer for feature extraction,
        %obtaining 4096 dimensional feature sets.
        im2_ = imresize(im_, net_alexnet.meta.normalization.imageSize(1:2)) ;
        im2_ = im2_ - net_alexnet.meta.normalization.averageImage ;
        res_alexnet = vl_simplenn(net_alexnet, im2_);
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
        
        labelStr = num2str(age);
        
        if plotta
            figure(1)
            imPlot = insertObjectAnnotation(imPlot, 'rectangle', bbox_p, labelStr, 'FontSize', 36, 'LineWidth', 3);
            imshow(imPlot, []);
            title('Estimated age', 'Interpreter', 'none')
            pause(0.01);
        end %if plotta
        
        %pause
        
        
        
    end %for bbox
    
    
end %while









