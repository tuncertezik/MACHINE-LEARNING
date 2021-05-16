function MachineLearningAssessmentItem1
%% Machine Learning Assessment
clc, clear all, close all, warning off

disp('Project started...')

%% Load Data
disp('Loading data...')
arffData = importdata('examples_2.arff');
testData = importdata('final_test.arff');
data = arffData.data;
finalData = testData.data;

%% Normalize Data
disp('Data normalizing...')
for i = 1 : size(data, 2) - 1
    mn = min(data(:, i));
    mx = max(data(:, i));
    mn1 = min(finalData(:, i));
    mx1 = max(finalData(:, i));
    data(:, i) = (data(:, i) - mn) / (mx - mn);
    finalData(:, i) = (finalData(:, i) - mn1) / (mx1 - mn1);
end

%% Feature Reduction
disp('Feature reduction...')
[coeff, score, latent, tsquared, explained] = pca(data);
features = 1 : max(find(cumsum(explained) < 99)) + 1;
reducedData = data(:, features);

[coeff1, score1, latent1, tsquared1, explained1] = pca(finalData);
features1 = 1 : max(find(cumsum(explained1) < 99)) + 1;
reducedFinalData = finalData(:, features1);

%% Train and test dataset
disp('Splitting train and test dataset...')
numData = size(data, 1);
numTrainData = round(numData * 0.7);
trainIdx = randperm(numData, numTrainData);
xTrainFull = data(trainIdx, 1 : 11);
yTrainFull = data(trainIdx, 12);
xTrainReduced = reducedData(trainIdx, :);
yTrainReduced = data(trainIdx, 12);

xTrainFull1 = data(:, 1 : 11);
yTrainFull1 = data(:, 12);
xTrainReduced1 = reducedData(:, :);
yTrainReduced1 = data(:, 12); 

testIdx = setdiff(1 : numData, trainIdx);
xTestFull = data(testIdx, 1 : 11);
yTestFull = data(testIdx, 12);
xTestReduced = reducedData(testIdx, :);
yTestReduced = data(testIdx, 12);

xTestFull1 = finalData(:, 1 : 11);
yTestFull1 = finalData(:, 12);
xTestReduced1 = reducedFinalData(:, :);
yTestReduced1 = finalData(:, 12);

%% Train and classification
disp('Train and classification...')
knnResFull = knnclassify(xTestFull, xTrainFull, yTrainFull);
knnResReduced = knnclassify(xTestReduced, xTrainReduced, yTrainReduced);

knnResFull1 = knnclassify(xTestFull1, xTrainFull1, yTrainFull1);
knnResReduced1 = knnclassify(xTestReduced1, xTrainReduced1, yTrainReduced1);

SVMStructFull = svmtrain(xTrainFull, yTrainFull);
SVMStructReduced = svmtrain(xTrainReduced, yTrainReduced);
svmResFull = svmclassify(SVMStructFull, xTestFull);
svmResReduced = svmclassify(SVMStructReduced, xTestReduced);

SVMStructFull1 = svmtrain(xTrainFull1, yTrainFull1);
SVMStructReduced1 = svmtrain(xTrainReduced1, yTrainReduced1);
svmResFull1 = svmclassify(SVMStructFull1, xTestFull1);
svmResReduced1 = svmclassify(SVMStructReduced1, xTestReduced1);

NBStructFull = NaiveBayes.fit(xTrainFull, yTrainFull);
NBStructReduced = NaiveBayes.fit(xTrainReduced, yTrainReduced);
nbResFull = predict(NBStructFull, xTestFull);
nbResReduced = predict(NBStructReduced, xTestReduced);

NBStructFull1 = NaiveBayes.fit(xTrainFull1, yTrainFull1);
NBStructReduced1 = NaiveBayes.fit(xTrainReduced1, yTrainReduced1);
nbResFull1 = predict(NBStructFull1, xTestFull1);
nbResReduced1 = predict(NBStructReduced1, xTestReduced1);

%% Evaluation
disp('Evaluation started...')
disp('Full Feature result');
disp('Support Vector Machine Classification...');
[confusionMat, precision11, recall11, fscore11] = confusionStat(yTestFull, svmResFull)
disp('Naive Bayes...');
[confusionMat, precision12, recall12, fscore12] = confusionStat(yTestFull, nbResFull)
disp('K Nearest Neighbors...');
[confusionMat, precision13, recall13, fscore13] = confusionStat(yTestFull, knnResFull)

disp('Reduced Feature result');
disp('Support Vector Machine Classification...');
[confusionMat, precision21, recall21, fscore21] = confusionStat(yTestReduced, svmResReduced)
disp('Naive Bayes...');
[confusionMat, precision22, recall22, fscore22] = confusionStat(yTestReduced, nbResReduced)
disp('K Nearest Neighbors...');
[confusionMat, precision23, recall23, fscore23] = confusionStat(yTestReduced, knnResReduced)

%% Plot Graph
figure;
bar([precision11, recall11, fscore11; precision12, recall12, fscore12;...
     precision13, recall13, fscore13], 'Group');
set(gca,'XTickLabel',{'SVM';'NB';'KNN'});
title('MATLAB Result for Full Data');

figure;
bar([precision21, recall21, fscore21; precision22, recall22, fscore22;...
     precision23, recall23, fscore23], 'Group');
set(gca,'XTickLabel',{'SVM';'NB';'KNN'});
title('MATLAB Result for Reduced Data');

%% Testing
disp('Testing started...')
disp('Full Feature result');
disp('Support Vector Machine Classification...');
[confusionMat, precision11, recall11, fscore11] = confusionStat(yTestFull1, svmResFull1)
disp('Naive Bayes...');
[confusionMat, precision12, recall12, fscore12] = confusionStat(yTestFull1, nbResFull1)
disp('K Nearest Neighbors...');
[confusionMat, precision13, recall13, fscore13] = confusionStat(yTestFull1, knnResFull1)

disp('Reduced Feature result');
disp('Support Vector Machine Classification...');
[confusionMat, precision21, recall21, fscore21] = confusionStat(yTestReduced1, svmResReduced1)
disp('Naive Bayes...');
[confusionMat, precision22, recall22, fscore22] = confusionStat(yTestReduced1, nbResReduced1)
disp('K Nearest Neighbors...');
[confusionMat, precision23, recall23, fscore23] = confusionStat(yTestReduced1, knnResReduced1)
disp('Project ended...')

function [value, precision, recall, fscore] = confusionStat(group, grouphat)
group = group + 1;
grouphat = grouphat + 1;
value = confusionmat(group, grouphat);
cp = classperf(group, grouphat);
precision = cp.CorrectRate;
recall = cp.Sensitivity;
fscore = 2 * (recall * precision) / (recall + precision);