%% Advance Neuroscience - HW09 - Ali Ghavampour 97102293

%% Simulate sparse basis function of the natural image
clc; clear all; close all;
rng shuffle
addpath('C:\Users\aligh\Desktop\Sharif FUT\Semester 6\Advance Neuro\HWs\HW09\sparsenet')
load IMAGES.mat
A = rand(256) - 0.5;
A = A*diag(1./sqrt(sum(A.*A)));

for i1 = 1:size(IMAGES,3)
    x{i1} = [ones(size(IMAGES,1),1),IMAGES(:,:,i1),ones(size(IMAGES,1),1)];
end
figure;
montage(x,'size',[1,10])
title("Unnatural Wihtened Images")


%% Run sparsenet
figure(1), colormap(gray)
sparsenet


%% Load and Crop Images
clear all; close all; clc

% Put the images in the temp folder
path = 'C:\Users\aligh\Desktop\Sharif FUT\Semester 6\Advance Neuro\HWs\HW09\Temp\imgFolder';
addpath(path)
files = struct2cell(dir(path));
names = files(1,3:end);
imgNum = length(names);

% finding minimum image size
for i = 1:length(names)
    I = imread(names{i});
    if (length(size(I)) > 2)
        I = rgb2gray(I);
    end
    I = im2double(I);
    tmpSize(i) = min(size(I));
end
minSize = min(tmpSize);

% Cropping to the min size
croppedImages = zeros(minSize,minSize,imgNum);
rect = [0,0,minSize,minSize];
for i = 1:imgNum
    I = imread(names{i});
    if (length(size(I)) > 2)
        I = rgb2gray(I);
    end
    I = im2double(I);
    Icropped = imcrop(I,rect);
    croppedImages(:,:,i) = Icropped;
end

for i1 = 1:size(croppedImages,3)
    x{i1} = [ones(size(croppedImages,1),10),croppedImages(:,:,i1),ones(size(croppedImages,1),10)];
end
figure;
montage(x,'size',[1,10])
title("Paper Original Images")



%% Whitening 
clc;

N = size(croppedImages,1);
imgNum = size(croppedImages,3);
M = imgNum;

[fx, fy]=meshgrid(-N/2:N/2-1,-N/2:N/2-1);
rho=sqrt(fx.*fx+fy.*fy);
f_0=0.4*N;
filt=rho.*exp(-(rho/f_0).^4);

for i=1:M
    image=croppedImages(:,:,i);
    If=fft2(image);
    imagew=real(ifft2(If.*fftshift(filt)));
%     IMAGES(:,i)=reshape(imagew,N^2,1);
    IMAGES(:,:,i) = imagew;
end
IMAGES=sqrt(0.1)*IMAGES./sqrt(mean(var(IMAGES)));
save MY_IMAGES IMAGES



%% Question03 - Bird video
clear all; close all; clc;

obj = VideoReader('BIRD.avi');
vid = read(obj);
frames = obj.NumberOfFrames;
for i = 1 : frames
    tmp = rgb2gray(vid(:, :, :, i));
    tmp = im2double(tmp);
    crop = imcrop(tmp,[0,0,288,288]);
    Vid(:,:,i) = crop;
end

% makeing IMAGES
IMAGES_holder = {};
for i = 1:10:110
    tmp = Vid(:,:,i:i+9);
    tmp = whitening(tmp);
    IMAGES_holder{round(i/10+1)} = tmp;
end
IMAGES_holder{12} = Vid(:,:,111:118);

%% Sparsenet for First 10 frames
clear all; close all; clc;
addpath('C:\Users\aligh\Desktop\Sharif FUT\Semester 6\Advance Neuro\HWs\HW09\sparsenet')
load IMAGES_holder.mat
IMAGES = IMAGES_holder{1};
A = rand(64) - 0.5;
A = A*diag(1./sqrt(sum(A.*A)));
figure(1), colormap(gray)
sparsenet

%% Finding coefficients in time
clear all; close all; clc;
load A.mat
load IMAGES_holder.mat

patchSize = 8;
noise_var = 0.01;
beta = 2.2;
sigma = 0.316;
tol = 0.01;

for f = 1:12
    images = IMAGES_holder{f};
    X = [];
    if (f<12)
        for j = 1:10
            img = images(:,:,j);
            X = [X,patcher(img,patchSize)];
        end
        S{f} = cgf_fitS(A,X,noise_var,beta,sigma,tol);
    else
        for j = 1:8
            img = images(:,:,j);
            X = [X,patcher(img,patchSize)];
        end
        S{f} = cgf_fitS(A,X,noise_var,beta,sigma,tol);
    end
end

%% Processing the coefficients
clear all; close all; clc;
load S.mat

goodP = [4061 3668 711 9672 5752 6943 4722 3149 5965 6981 8226 2317 6157 8938 9623];
for j = 1:4
    subplot(2,2,j)
    x = [];
    p = randi(9800);
    for i = 1:12
        tmp = S{i};
        x = [x,abs(tmp(:,p))];
        plot(abs(tmp(:,p))+(i)*1.5);
        yline((i)*1.5,'--r');
        hold on
    end
    plot(mean(x,2)+(i+1)*1.6,'k','linewidth',2)
    yline((i+1)*1.6,'--r');
    title(sprintf("Patch Number %d",p))
    ylabel("Time(rising from bottom to top)")
    xlabel("Coefficient")
end

% good patches
verygoodP = [4061 3149 9623 6157];
figure;
for j = 1:4
    subplot(2,2,j)
    x = [];
    p = verygoodP(j);
    for i = 1:12
        tmp = S{i};
        x = [x,abs(tmp(:,p))];
        plot(abs(tmp(:,p))+(i)*1.5);
        yline((i)*1.5,'--r');
        hold on
    end
    plot(mean(x,2)+(i+1)*1.6,'k','linewidth',2)
    yline((i+1)*1.6,'--r');
    title(sprintf("Patch Number %d",p))
    ylabel("Time(rising from bottom to top)")
    xlabel("Coefficient")
end



figure;
for i = 1:12
    tmp = S{i};
    avg = mean(abs(tmp),2);
    x = [x,avg];
    plot(avg+i*1.5,'linewidth',1.5);
    yline((i)*1.5,'--r');
    hold on
end
plot(mean(x,2)+(i+1)*1.6,'k','linewidth',2)
yline((i+1)*1.6,'--r');
ylabel("Time(rising from bottom to top)")
xlabel("Coefficient")
title("Mean of absolute of coefficients over all patches")

%Histograms
figure;
for i = 1:12
    tmp = S{i};
    x = reshape(tmp,size(tmp,1)*size(tmp,2),1);
%     x = abs(x);
    subplot(3,4,i);
    hist(x,5000)
    ylim([0,1.4*10^4])
    xlim([-6 6])
    title(sprintf("Frame set %d",i))
end


%% Optional Part - Saliency maps of paper images
clear all; close all; clc;
addpath('C:\Users\aligh\Desktop\Sharif FUT\Semester 6\Advance Neuro\HWs\HW08\matlabPyrTools-master\matlabPyrTools-master');
addpath('C:\Users\aligh\Desktop\Sharif FUT\Semester 6\Advance Neuro\HWs\HW08\SaliencyToolbox2.3\SaliencyToolbox')
addpath('C:\Users\aligh\Desktop\Sharif FUT\Semester 6\Advance Neuro\HWs\HW08\voc-release-3.1-win-master\voc-release-3.1-win-master')
addpath('C:\Users\aligh\Desktop\Sharif FUT\Semester 6\Advance Neuro\HWs\HW08\FaceDetect')
addpath('C:\Users\aligh\Desktop\Sharif FUT\Semester 6\Advance Neuro\HWs\HW08\LabelMeToolbox-master\LabelMeToolbox-master')
addpath('C:\Users\aligh\Desktop\Sharif FUT\Semester 6\Advance Neuro\HWs\HW08\JuddSaliencyModel\JuddSaliencyModel')
addpath('C:\Users\aligh\Desktop\Sharif FUT\Semester 6\Advance Neuro\HWs\HW08\JuddSaliencyModel\JuddSaliencyModel\horizon code')
addpath('C:\Users\aligh\Desktop\Sharif FUT\Semester 6\Advance Neuro\HWs\HW08\LabelMeToolbox-master\LabelMeToolbox-master\features')
addpath('C:\Users\aligh\Desktop\Sharif FUT\Semester 6\Advance Neuro\HWs\HW08\LabelMeToolbox-master\LabelMeToolbox-master\imagemanipulation')
addpath('C:\Users\aligh\Desktop\Sharif FUT\Semester 6\Advance Neuro\HWs\HW08\JuddSaliencyModel\JuddSaliencyModel')
addpath('C:\Users\aligh\Desktop\Sharif FUT\Semester 6\Advance Neuro\HWs\HW08\JuddSaliencyModel\JuddSaliencyModel\FelzenszwalbDetectors')
addpath('C:\Users\aligh\Desktop\Sharif FUT\Semester 6\Advance Neuro\HWs\HW08\matlabPyrTools-master\matlabPyrTools-master\MEX')

load IMAGES_RAW.mat;
map = {};
for i = 1:10
    img = IMAGESr(:,:,i);
    img = repmat(img,1,1,3);
    [saliencyMap,FEATURES] = saliency(img,10);
    map{i} = saliencyMap;
end

%% Patching
clear all; close all; clc;
load map.mat
load IMAGES.mat
pSize = 8;
p = [];
th = (pSize*pSize)/2;
th = 20;
cnt = 1;
for i = 1:10
    img = IMAGES(:,:,i);
    salMap = map{i};
    step = pSize - pSize/2;
    for r = 1:step:size(img,1)-step
        disp(r);
        for c = 1:step:size(img,2)-step
            pTmp = img(r:r+pSize-1,c:c+pSize-1);
            pTmp = reshape(pTmp,pSize^2,1);
            sc = sum(salMap(r:r+pSize-1,c:c+pSize-1),'all');
            if (sc <= th)
                p(:,cnt) = pTmp;
                cnt = cnt + 1;
            end
        end
    end
end
save p


%% Run Sparsenet
clear all; close all; clc;
rng shuffle
load IMAGES.mat
A = rand(64) - 0.5;
A = A*diag(1./sqrt(sum(A.*A)));
figure(1), colormap(gray)
mySparsenet




%% Functions
function [X] = patcher(img,patchSize)
    sz = size(img);
    p = {};
    inds1 = 1:patchSize:sz(1)-patchSize;
    inds2 = 1:patchSize:sz(2)-patchSize;
    for i = 1:length(inds1)
        i1 = inds1(i);
        for j = 1:length(inds2)
            j1 = inds2(j);
            p{(i-1)*length(inds2) + j} = img(i1:i1+patchSize-1,j1:j1+patchSize-1);
            X(:,(i-1)*length(inds2) + j) = reshape(img(i1:i1+patchSize-1,j1:j1+patchSize-1),patchSize^2,1);
        end
    end
    
end

function IMAGES = whitening(images)
    N = size(images,1);
    imgNum = size(images,3);
    M = imgNum;
    [fx, fy]=meshgrid(-N/2:N/2-1,-N/2:N/2-1);
    rho=sqrt(fx.*fx+fy.*fy);
    f_0=0.4*N;
    filt=rho.*exp(-(rho/f_0).^4);
    for i=1:M
        image=images(:,:,i);
        If=fft2(image);
        imagew=real(ifft2(If.*fftshift(filt)));
        IMAGES(:,:,i) = imagew;
    end
    IMAGES=sqrt(0.1)*IMAGES./sqrt(mean(var(IMAGES)));
end



