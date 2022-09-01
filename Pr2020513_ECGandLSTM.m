%% load data
s = load("ArrhythmiasTrain_5sec_one.mat");
XTrain={}; YTrain={};
XTrain=[s.XTrain(1:150); s.XTrain(201:350); XTrain];
YTrain=[s.YTrain(1:150); s.YTrain(201:350); YTrain];

XVal={}; YVal={};
XVal=[s.XTrain(151:200); s.XTrain(351:400); XVal];
YVal=[s.YTrain(151:200); s.YTrain(351:400); YVal];
% [XTrain,YTrain] = japaneseVowelsTrainData;
numObservations = numel(XTrain);
classes = categories(YTrain);
numClasses = numel(classes);

%% Build LSTM
numFeatures = 1;
numHiddenUnits1 = 64;
numHiddenUnits2 = 32;

layers = [ ...
    sequenceInputLayer(numFeatures)
    bilstmLayer(numHiddenUnits1,'OutputMode','last')
    dropoutLayer(0.1)
    bilstmLayer(numHiddenUnits2,'OutputMode','last')
    dropoutLayer(0.1)
    flattenLayer
    fullyConnectedLayer(128)
    reluLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
    ];

%% Define options
options = trainingOptions('adam', ...
    'MaxEpochs',50, ...
    'MiniBatchSize',10, ...
    'Shuffle','every-epoch', ...
    'InitialLearnRate',0.001, ...
    'GradientThreshold',1, ...
    'Verbose',1, ...
    'Plots','training-progress');

%% Train network
net = trainNetwork(XTrain,YTrain,layers,options);

%%
% t = load("ArrhythmiasTest_5sec_one.mat");
% XTest = t.XTest;
% YTest = t.YTest;
[XTest,YTest] = japaneseVowelsTestData;
testPred = classify(net,XTest);
LSTMAccuracy = sum(testPred == YTest)/numel(YTest)*100
figure
confusionchart(YTest,testPred,'Title','Confusion Chart for LSTM');
