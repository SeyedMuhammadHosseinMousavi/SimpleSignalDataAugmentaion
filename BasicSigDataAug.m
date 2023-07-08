%% Basic Signal Augmentation
% Developed by Seyed Muhammad Hossein Mousavi - July 2023
% Augmentations include Time shifting, Amplitude scaling, and Random Gaussian noise It 
% applies to a simple self-made dataset with 22 samples and 5 features in three classes
% which you can change it. KNN is used as a classifier for both original and (augmented) 
% synthetic data. For the original dataset, there is cross-validation by 30-70 
% train test data split, and for augmented data, train in on whole augmented data and test
% is on the whole original dataset.

clear;
clc;
close all;
warning('off')

%% Load the original dataset
% Each row represents a data sample, and each column represents a feature
dataset=load('SimpleDataset.mat');
dataset=dataset.dataset;
% Splitting data into train and test
SizeData=size(dataset,1);
cv = cvpartition(SizeData,'HoldOut',0.3); % Cross varidation 
dataTrain = dataset(cv.training,:);
dataTest = dataset(cv.test,:);
% Storing dataset without label
AllDataforAug=dataset(:,1:end-1);
% Label vectors
lbl=dataset(:,end);
lbltrain=dataTrain(:,end);
lbltest=dataTest(:,end);


%% Parameters
TimeShiftRange = 0.2; % Maximum time shift range (in seconds)
ScaleRange = 0.3; % Maximum amplitude scaling range
NoiseMagnitude = 0.1; % Adjust the noise magnitude as needed
%-------------------------------------------------------------

%% The desired number of augmented samples
NoAugSam = 100;
%------------------------------------------------------------

% Empty matrix creation
% +1 is for preventing error in adding lbl
AugmentedDataset = zeros(NoAugSam, size(AllDataforAug, 2)+1);
for i = 1:NoAugSam
% Select a random data sample from the original dataset
randomSample = randi(size(dataset, 1));
% Keeping the record of selected samples
StoreRandSampIndex(i)=randomSample;
% Temporary storing the selected sample
originalSample = AllDataforAug(randomSample, :);
%%--------------------------------------------------------
% Apply data augmentation operations to the selected sample
augmentedSample = ApplyDataAug(originalSample,TimeShiftRange,ScaleRange,NoiseMagnitude);
%--------------------------------------------------------
% Adding related lbl at the end of each augmented sample
augmentedSample(1,end+1)=lbl(StoreRandSampIndex(i));
% Store the augmented sample in the augmented dataset
AugmentedDataset(i, :) = augmentedSample;
end

Augmented= AugmentedDataset (:,1:end-1);
lbl2= AugmentedDataset (:,end);
% Save the augmented dataset
% save('Augmented_dataset.mat', 'augmentedDataset');

%% Plot the comparison result
figure('units','normalized','outerposition',[0 0 1 1])
subplot(1,2,1)
plot(AllDataforAug, 'linewidth',1); title('Original Data');xlim([0 size(dataset, 1)]);
subplot(1,2,2)
plot(AugmentedDataset(:,1:end-1), 'linewidth',1); title('Augmented Data');xlim([0 size(AugmentedDataset, 1)]);

%% Plot data and classes
Feature1=1;
Feature2=2;
% Original
f1=dataset(:,Feature1); % feature1
f2=dataset(:,Feature2); % feature 2
% Augmented 
ff1=AugmentedDataset(:,Feature1); % feature1
ff2=AugmentedDataset(:,Feature2); % feature 2
lbl2=AugmentedDataset(:,end); % labels
figure('units','normalized','outerposition',[0 0 1 1])
subplot(1,2,1)
gscatter(f1,f2,lbl,'rkgb','*',12); title('Original');
subplot(1,2,2)
gscatter(ff1,ff2,lbl2,'rkgb','*',12); title('Augmented');

%% Training
% Training original dataset by KNN
DataTrainNoLbl=dataTrain(:,1:end-1);
Mdl = fitcknn(DataTrainNoLbl,lbltrain,'Distance', 'Euclidean','NumNeighbors',1,'Standardize',1);
rng(1); % For reproducibility
KNNmodel = crossval(Mdl,'KFold', 5);
KNNError = kfoldLoss(KNNmodel);
KNNpredict = kfoldPredict (KNNmodel);
KNNAccOrgTrain = (1 - kfoldLoss(KNNmodel, 'LossFun', 'ClassifError'))*100

% Training augmented dataset by KNN
Mdl2 = fitcknn(Augmented,lbl2,'Distance', 'Euclidean','NumNeighbors',1,'Standardize',1);
rng(1); % For reproducibility
KNNmodel2 = crossval(Mdl2,'KFold', 5);
KNNError2 = kfoldLoss(KNNmodel2);
KNNpredict2 = kfoldPredict (KNNmodel2);
KNNAccAugTrain = (1 - kfoldLoss(KNNmodel2, 'LossFun', 'ClassifError'))*100

% Getting label sizes
sizlbl=size(lbl);
sizlbl=sizlbl(1,1);
sizefeature=size(dataset);
sizefeature=sizefeature(1,2);
sizlbl1=size(lbltest);
sizlbl1=sizlbl1(1,1);

%% Predict new data
% Predict for original dataset - on 30 % left
DataTestNoLbl=dataTest(:,1:end-1);
[label3,score3,cost3] = predict(Mdl,DataTestNoLbl);
% Predict for augmented dataset - on whole original dataset
[label2,score2,cost2] = predict(Mdl2,AllDataforAug);

%% Misclassifications
% for original data
counter1=0; % Misclassifications places
misindex1=0; % Misclassifications indexes
for i=1:sizlbl1
if lbltest(i)~=label3(i)
misindex1(i)=i;
counter1=counter1+1;
end
end

% For augmented data
counter=0; % Misclassifications places
misindex=0; % Misclassifications indexes
for i=1:sizlbl
if lbl(i)~=label2(i)
misindex(i)=i;
counter=counter+1;
end
end

% Test accuracy original
TestErrOrg = counter1*100/sizlbl1;
KNNAccOrgTest = 100 - TestErrOrg
% Test accuracy augmented
TestErrAug = counter*100/sizlbl;
KNNAccAugTest = 100 - TestErrAug

% Results
OrgRes = [' Original Train "',num2str(KNNAccOrgTrain),'" Original Test "', num2str(KNNAccOrgTest),'"'];
AugRes = [' Augmented Train "',num2str(KNNAccAugTrain),'" Augmented Test "', num2str(KNNAccAugTest),'"'];
disp(OrgRes);
disp(AugRes);



%% Hint/Help
%Some common techniques used for signal data augmentation include:
%Time and Frequency Shifts: Shifting a signal in time or frequency domains by a certain amount can simulate
%changes in timing or pitch. For example, audio signals can be time-stretched or compressed, or their frequencies can be shifted up or down.
%Noise Injection: Adding different types of noise to signals can simulate real-world noise or measurement errors.
%This helps the model become more robust to noise in practical applications.
%Scaling and Amplitude Variations: Changing the amplitude or scaling of signals can simulate variations in signal
%strength or intensity. It can be applied to various types of signals, such as images, audio, or sensor data.
%Filtering and Smoothing: Applying different filters or smoothing techniques to signals can alter their frequency 
%content or remove noise. This can help the model learn to handle different levels of signal smoothness or filtering effects.
%Time Warping: Distorting the time scale of signals by stretching or compressing different parts can simulate temporal
%variations. It is commonly used in applications involving time series data.
%Data Mixing: Combining multiple signals or mixing different parts of signals can create new samples with different characteristics. 
%This technique is often used in audio processing or speech recognition tasks.
%By employing signal data augmentation, models can benefit from a more diverse and representative training dataset,
%leading to improved generalization and performance when applied to real-world scenarios. It helps prevent overfitting 
%and enhances the model's ability to handle variations, noise, and uncertainties present in the signals it encounters during inference.
