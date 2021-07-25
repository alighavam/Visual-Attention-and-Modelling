%% Advance Neuro HW08 - Ali Ghavampour - 97102293

%% Part01 =================================================================
clear all; close all; clc;

datafolder = 'C:\Users\aligh\Desktop\Sharif FUT\Semester 6\Advance Neuro\HWs\HW08\Eye tracking database\Eye tracking database\DATA\hp';
stimfolder = 'C:\Users\aligh\Desktop\Sharif FUT\Semester 6\Advance Neuro\HWs\HW08\Eye tracking database\Eye tracking database\ALLSTIMULI';
addpath('C:\Users\aligh\Desktop\Sharif FUT\Semester 6\Advance Neuro\HWs\HW08\DatabaseCode')
showEyeData(datafolder, stimfolder)

% numFix = 3;
% showEyeDataAcrossUsers(stimfolder, numFix)


%% Part02 =================================================================
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
addpath('C:\Users\aligh\Desktop\Sharif FUT\Semester 6\Advance Neuro\HWs\HW08\Eye tracking database\Eye tracking database\ALLSTIMULI')
addpath('C:\Users\aligh\Desktop\Sharif FUT\Semester 6\Advance Neuro\HWs\HW08\FaceDetect\mine')
addpath('C:\Users\aligh\Desktop\Sharif FUT\Semester 6\Advance Neuro\HWs\HW08\matlabPyrTools-master\matlabPyrTools-master\MEX')

imgFile = 'i2238477160.jpeg';
[~,FEATURES] = saliency(imgFile,10);

load model
img = imread(imgFile);
lengthF = size(FEATURES,2);
[win, h, c] = size(img);
dims = [200, 200];
for f = 1:10
    w = model.w;
    if (f==1) %subband
        w(14:end-1) = 0;
    elseif (f==2) %Itti
        w(1:13) = 0;
        w(17:end-1) = 0; 
    elseif (f==3) %Color
        w(1:16) = 0;
        w(28:end-1) = 0;
    elseif f==4 %Torralba
        w(1:27) = 0;
        w(29:end-1) = 0;
    elseif f==5 %Horizon
        w(1:28) = 0;
        w(30:end-1) = 0;
    elseif f==6 %Object
        w(1:29) = 0;
        w(32:end-1) = 0;
    elseif f==7 %Center
        w(1:31) = 0;
        w(33) = 0;
    elseif f==8 %low-level
        w(29:end-1) = 0;
    elseif f==9 %mid&high-level
        w(1:28) = 0;
    elseif f==10 %all
        w = model.w;
    end
    meanVec = model.whiteningParams(1, lengthF);
    stdVec = model.whiteningParams(2, lengthF);
    FEATURESTmp=FEATURES-repmat(meanVec, [size(FEATURES, 1), 1]);
    FEATURESTmp=FEATURESTmp./repmat(stdVec, [size(FEATURESTmp, 1), 1]);
    
    % find the saliency map given the features and the model
    saliencyMapTmp = (FEATURESTmp*w(1:end-1)') + w(end);
    saliencyMapTmp = (saliencyMapTmp-min(saliencyMapTmp))/(max(saliencyMapTmp)-min(saliencyMapTmp));
    saliencyMapTmp = reshape(saliencyMapTmp, dims);
    saliencyMap{f} = imresize(saliencyMapTmp, [win, h]);
end

map{1} = img;
map(2:8) = saliencyMap(1:7);
figure;
montage(map,'size',[2,4])
title("From top left: 1)image  2)subband  3)Itti  4)color  5)Torralba  6)Horizon  7)Objects  8)Center")

figure;
montage({img,saliencyMap{10}})
title("Total saliency map")

%% Section 3
clc; close all; clear all;

addpath('C:\Users\aligh\Desktop\Sharif FUT\Semester 6\Advance Neuro\HWs\HW08\gbvs\gbvs\util')
addpath('C:\Users\aligh\Desktop\Sharif FUT\Semester 6\Advance Neuro\HWs\HW08\Eye tracking database\Eye tracking database\DATA\hp')
addpath('C:\Users\aligh\Desktop\Sharif FUT\Semester 6\Advance Neuro\HWs\HW08\Eye tracking database\Eye tracking database\ALLSTIMULI')
addpath('C:\Users\aligh\Desktop\Sharif FUT\Semester 6\Advance Neuro\HWs\HW08\LabelMeToolbox-master\LabelMeToolbox-master\objectdetection')
addpath('C:\Users\aligh\Desktop\Sharif FUT\Semester 6\Advance Neuro\HWs\HW08\Eye tracking database\Eye tracking database\ALLFIXATIONMAPS\ALLFIXATIONMAPS')
addpath('C:\Users\aligh\Desktop\Sharif FUT\Semester 6\Advance Neuro\HWs\HW08\DatabaseCode')

name = 'i1799873859';
filename = sprintf("%s.mat",name);
imgname = sprintf("%s.jpeg",name);
salName = sprintf("%s_fixMap.jpg",name);
imgdisp = name;

[saliencyMap,~] = saliency(imgname,10);

data = load(filename);
data = struct2cell(data);
data = data{1};
data = data.DATA.eyeData;
[eyeData Fix Sac] = checkFixations(data);
fixs = find(eyeData(:,3)==0);

img = imread(imgname);
imgSal = imread(salName);
origimgsize = size(img);
x = eyeData(:,1);
y = eyeData(:,2);
x1 = x(1:362);
x2 = x(363:end);
y1 = y(1:362);
y2 = y(363:end);

%% SaliencyMap along with eye tracking data
close all;
montage({img,imgSal,saliencyMap},'Size',[1,3])
title("1)Image  2)Eye tracking data  3)Saliency map")

%% First half vs Second hald data
close all; clc;
nameTmp = 'i1799873859';
imgnameTmp = sprintf("%s.jpeg",nameTmp);
filenameTmp = sprintf("%s.mat",nameTmp);
img = imread(imgnameTmp);
data = load(filenameTmp);
data = struct2cell(data);
data = data{1};
data = data.DATA.eyeData;
[eyeData,~,~] = checkFixations(data);

x = eyeData(:,1);
y = eyeData(:,2);
dis = 360;
x1 = x(1:dis);
x2 = x(725-dis+1:end);
y1 = y(1:dis);
y2 = y(725-dis+1:end);

% subplot(1,2,1)
imshow(img);
hold on
plot(x1,y1,'r.','MarkerSize',14);
hold on
plot(x2,y2,'g.','MarkerSize',14);
title("Red: Early saccads    Green: Late saccads    Black Circle: Middle Point")
hold on
scatter(round(size(img,2)/2),round(size(img,1)/2),2000,'k','LineWidth',4)

%% Score Histograms
clear all; close all; clc;

datafolder = 'C:\Users\aligh\Desktop\Sharif FUT\Semester 6\Advance Neuro\HWs\HW08\Eye tracking database\Eye tracking database\DATA\hp';
stimfolder = 'C:\Users\aligh\Desktop\Sharif FUT\Semester 6\Advance Neuro\HWs\HW08\Eye tracking database\Eye tracking database\ALLSTIMULI';
files=dir(fullfile(datafolder,'*.mat'));
[filenames{1:size(files,1)}] = deal(files.name);
Nstimuli = size(filenames,2);

score1 = [];
score2 = [];
dis = 180;

tic
for i = 1:100
    fprintf("i = %d\n",i)
    
    % load eye data and get eye tracking positions
    load(fullfile(datafolder,filenames{i}))
    stimFile = eval([filenames{i}(1:end-4)]);
    eyeData = stimFile.DATA.eyeData;
    [eyeData Fix Sac] = checkFixations(eyeData);
    fixs = find(eyeData(:,3)==0);

    % load image and get saliency maps
    imgName = stimFile.imgName;
    img = imread(imgName);
    map = {};
    for f = 8:9 %saliency map for low-level and high-level features
        [map{f-7},~] = saliency(imgName,f);
    end

    % first fourth and last fourth eye positions
    x = eyeData(:,1);
    y = eyeData(:,2);
    x1 = x(10:dis);
    y1 = y(10:dis);
    x2 = x(end-dis+1:end);
    y2 = y(end-dis+1:end);
    origimgsize = size(img);

    % calculating low-level scores
    score1(1,i) = rocScoreSaliencyVsFixations(map{1},x1,y1,origimgsize);
    score1(2,i) = rocScoreSaliencyVsFixations(map{1},x2,y2,origimgsize);

    % calculating high-level scores
    score2(1,i) = rocScoreSaliencyVsFixations(map{2},x1,y1,origimgsize);
    score2(2,i) = rocScoreSaliencyVsFixations(map{2},x2,y2,origimgsize);
end
toc

%% For scores1 and scores2
nbins = 10;
[counts1,centers1] = hist(score2(1,:),nbins);
[counts2,centers2] = hist(score2(2,:),nbins);
plot(centers1,counts1,'k','LineWidth',1.5);
hold on
plot(centers2,counts2,'r','LineWidth',1.5);
title("Low-Level features - hp subject")
ylabel("counts")
xlabel("Score")
legend("First Fourth","Last Fourth",'location','northwest')
xlim([0,1])

%% Score Histograms - All features
clear all; close all; clc;

stimfolder = 'C:\Users\aligh\Desktop\Sharif FUT\Semester 6\Advance Neuro\HWs\HW08\Eye tracking database\Eye tracking database\ALLSTIMULI';
datafolder = "C:\Users\aligh\Desktop\Sharif FUT\Semester 6\Advance Neuro\HWs\HW08\Eye tracking database\Eye tracking database\DATA\ajs";
files=dir(fullfile(datafolder,'*.mat'));
[filenames{1:size(files,1)}] = deal(files.name);
Nstimuli = size(filenames,2);
subName = {'ajs', 'CNG', 'emb', 'ems', 'ff', 'hp', 'jcw', ...
    'jw', 'kae', 'krl', 'po', 'tmj', 'tu', 'ya', 'zb'};

scores = [];
dis = 100;
numImg = 50;
tic
for i = 1:numImg
    fprintf("i = %d\n",i)
    % load image and get saliency maps
    datafolder = sprintf("C:\\Users\\aligh\\Desktop\\Sharif FUT\\Semester 6\\Advance Neuro\\HWs\\HW08\\Eye tracking database\\Eye tracking database\\DATA\\ajs");
    load(fullfile(datafolder,filenames{i}))
    stimFile = eval([filenames{i}(1:end-4)]);
    imgName = stimFile.imgName;
    img = imread(imgName);
    [~,FEATURES] = saliency(imgName,10);
    map = {};
    load model
    img = imread(imgName);
    lengthF = size(FEATURES,2);
    [win, h, c] = size(img);
    dims = [200, 200];
    for f = 8:9
        w = model.w;
        if (f==1) %subband
            w(14:end-1) = 0;
        elseif (f==2) %Itti
            w(1:13) = 0;
            w(17:end-1) = 0; 
        elseif (f==3) %Color
            w(1:16) = 0;
            w(28:end-1) = 0;
        elseif f==4 %Torralba
            w(1:27) = 0;
            w(29:end-1) = 0;
        elseif f==5 %Horizon
            w(1:28) = 0;
            w(30:end-1) = 0;
        elseif f==6 %Object
            w(1:29) = 0;
            w(32:end-1) = 0;
        elseif f==7 %Center
            w(1:31) = 0;
            w(33) = 0;
        elseif f==8 %low-level
            w(29:end-1) = 0;
        elseif f==9 %mid&high-level
            w(1:28) = 0;
        elseif f==10 %all
            w = model.w;
        end
        meanVec = model.whiteningParams(1, lengthF);
        stdVec = model.whiteningParams(2, lengthF);
        FEATURESTmp=FEATURES-repmat(meanVec, [size(FEATURES, 1), 1]);
        FEATURESTmp=FEATURESTmp./repmat(stdVec, [size(FEATURESTmp, 1), 1]);

        % find the saliency map given the features and the model
        saliencyMapTmp = (FEATURESTmp*w(1:end-1)') + w(end);
        saliencyMapTmp = (saliencyMapTmp-min(saliencyMapTmp))/(max(saliencyMapTmp)-min(saliencyMapTmp));
        saliencyMapTmp = reshape(saliencyMapTmp, dims);
        map{f-7} = imresize(saliencyMapTmp, [win, h]);
    end
    
    % first fourth and last fourth eye positions
    for sub = 1:14
        fprintf("subject %d\n",sub)
        % load eye data and get eye tracking positions
        datafolder = sprintf("C:\\Users\\aligh\\Desktop\\Sharif FUT\\Semester 6\\Advance Neuro\\HWs\\HW08\\Eye tracking database\\Eye tracking database\\DATA\\%s",subName{sub});
        load(fullfile(datafolder,filenames{i}))
        stimFile = eval([filenames{i}(1:end-4)]);
        eyeData = stimFile.DATA.eyeData;
        [eyeData Fix Sac] = checkFixations(eyeData);
        fixs = find(eyeData(:,3)==0);
        
        x = eyeData(:,1);
        y = eyeData(:,2);
        x1 = x(100:dis+9);
        x1(find(isnan(x1))) = [];
        y1 = y(100:dis+9);
        y1(find(isnan(y1))) = [];
        x2 = x(end-dis+1:end);
        x2(find(isnan(x2))) = [];
        y2 = y(end-dis+1:end);
        y2(find(isnan(y2))) = [];
        origimgsize = size(img);

        % calculating scores    
        for f = 1:length(map)
            scores(i,sub,f,1) = rocScoreSaliencyVsFixations(map{f},x1,y1,origimgsize);
            scores(i,sub,f,2) = rocScoreSaliencyVsFixations(map{f},x2,y2,origimgsize);
        end
    end
end
toc


%% scores plot
clc; close all;
flag = {'Subband','Itti','Color','Torralba','Horizon','Object','Center'};
for f = 1:7
    a1 = scores(:,f,1);
    a2 = scores(:,f,2);
    nbins = 10;
    [counts1,centers1] = hist(a1,nbins);
    [counts2,centers2] = hist(a2,nbins);
    subplot(2,4,f)
    plot(centers1,counts1,'k','linewidth',1.5);
    hold on
    plot(centers2,counts2,'r','linewidth',1.5);
    title(sprintf("%s",flag{f}))
    ylabel("Counts")
    xlabel("Score")
    legend("First 0.2 second","Last 0.2 second",'location','northwest')
    xlim([0,1])
end
sgtitle("Scores for hp subject")

%% Scores All plot
clc; close all;

sc = zeros(14*numImg,7,2);
for f = 1:7
    tmp1 = [];
    tmp2 = [];
    for sub = 1:14
        tmp1 = [tmp1,scores(:,sub,f,1)'];
        tmp2 = [tmp2,scores(:,sub,f,2)'];
    end
    sc(:,f,1) = tmp1;
    sc(:,f,2) = tmp2;
end

flag = {'Subband','Itti','Color','Torralba','Horizon','Object','Center'};
for f = 1:7
    a1 = sc(:,f,1);
    a2 = sc(:,f,2);
    nbins = 20;
    [counts1,centers1] = hist(a1,nbins);
    [counts2,centers2] = hist(a2,nbins);
    subplot(2,4,f)
    plot(centers1,counts1,'k','LineWidth',1.5);
    hold on
    plot(centers2,counts2,'r','LineWidth',1.5);
    title(sprintf("%s",flag{f}))
    ylabel("Counts")
    xlabel("Score")
    legend("First 0.2 second","Last 0.2 second",'location','northwest')
    xlim([0,1])
end
sgtitle("Scores for all subjects for 50 images")

%% scores Low-High All
clc; close all;

sc = zeros(14*50,2,2);
for f = 1:2
    tmp1 = [];
    tmp2 = [];
    for sub = 1:14
        tmp1 = [tmp1,scores(:,sub,f,1)'];
        tmp2 = [tmp2,scores(:,sub,f,2)'];
    end
    sc(:,f,1) = tmp1;
    sc(:,f,2) = tmp2;
end

flag = {'Low-Level','High-Level'};
for f = 1:2
    a1 = sc(:,f,1);
    a2 = sc(:,f,2);
    nbins = 10;
    [counts1,centers1] = hist(a1,nbins);
    [counts2,centers2] = hist(a2,nbins);
    subplot(1,2,f)
    plot(centers1,counts1,'k','LineWidth',1.5);
    hold on
    plot(centers2,counts2,'r','LineWidth',1.5);
    title(sprintf("%s",flag{f}))
    ylabel("Counts")
    xlabel("Score")
    legend("First 0.2 second","Last 0.2 second",'location','northwest')
    xlim([0,1])
end
sgtitle("Scores for all subjects for 50 images")



