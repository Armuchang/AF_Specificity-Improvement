close all;
clear;
clc;

%% load data
cd E:\Project\mat\mat_tcn\TCN_samples\Train\2cats_NPA
s = load("ArrhythmiasTrain_30sec_2cats_NPA_CutAF_600seg.mat");
XTrain = s.XTrain;
YTrain = s.YTrain;

numObservations = numel(XTrain);
classes = categories(YTrain{1});
numClasses = numel(classes);

%% define parameters in residual blocks
numBlocks = 10;
numFilters = 175;
filterSize = 18;
dropoutFactor = 0.05;

%% create a struct containing the model parameters
hyperparameters = struct;
hyperparameters.NumBlocks = numBlocks;
hyperparameters.DropoutFactor = dropoutFactor;

%% Create a struct containing dlarray objects for all the learnable parameters
numInputChannels = 1;

parameters = struct;
numChannels = numInputChannels;
cd E:\Project\mat\mat_tcn
for k = 1:numBlocks
    parametersBlock = struct;
    blockName = "Block"+k;
    
    weights = initializeGaussian([filterSize, numChannels, numFilters]);
    bias = zeros(numFilters, 1, 'single');
    parametersBlock.Conv1.Weights = dlarray(weights);
    parametersBlock.Conv1.Bias = dlarray(bias);
    
    weights = initializeGaussian([filterSize, numFilters, numFilters]);
    bias = zeros(numFilters, 1, 'single');
    parametersBlock.Conv2.Weights = dlarray(weights);
    parametersBlock.Conv2.Bias = dlarray(bias);
    
    % If the input and output of the block have different numbers of
    % channels, then add a convolution with filter size 1.
    if numChannels ~= numFilters
        weights = initializeGaussian([1, numChannels, numFilters]);
        bias = zeros(numFilters, 1, 'single');
        parametersBlock.Conv3.Weights = dlarray(weights);
        parametersBlock.Conv3.Bias = dlarray(bias);
    end
    numChannels = numFilters;
    
    parameters.(blockName) = parametersBlock;
end

weights = initializeGaussian([numClasses,numChannels]);
bias = zeros(numClasses,1,'single');

parameters.FC.Weights = dlarray(weights);
parameters.FC.Bias = dlarray(bias);

%% Specify Training Options
maxEpochs = 6;
miniBatchSize = 10;
initialLearnRate = 0.001;
learnRateDropFactor = 0.1;
learnRateDropPeriod = 12;
gradientThreshold = 1;

executionEnvironment = "auto";

plots = "training-progress";

%% Initialize the learning rate
learnRate = initialLearnRate;

%% Initialize the moving average of the parameter gradients and ...
% the element-wise squares of the gradients used by the Adam optimizer.
trailingAvg = [];
trailingAvgSq = [];

%% Initialize a plot showing the training progress.
if plots == "training-progress"
    figure
    lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
    ylim([0 inf])
    xlabel("Iteration")
    ylabel("Loss")
    grid on
end

%% Train the model.
iteration = 0;
numIterationsPerEpoch = floor(numObservations./miniBatchSize);

start = tic;

% Loop over epochs.
for epoch = 1:maxEpochs
    
    % Shuffle the data.
    idx = randperm(numObservations);
    XTrain = XTrain(idx);
    YTrain = YTrain(idx);
    
    % Loop over mini-batches.
    for i = 1:numIterationsPerEpoch
        iteration = iteration + 1;
        
        % Read mini-batch of data and apply the transformSequences
        % preprocessing function.
        idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        [X,Y,numTimeSteps] = transformSequences(XTrain(idx),YTrain(idx));
        
        % Convert to dlarray.
        dlX = dlarray(X);
        
        % If training on a GPU, convert data to gpuArray.
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
        end
        
        % Evaluate the model gradients and loss using dlfeval.
        [gradients, loss] = dlfeval(@modelGradients,dlX,Y,parameters,hyperparameters,numTimeSteps);
        
        % Clip the gradients.
        gradients = dlupdate(@(g) thresholdL2Norm(g,gradientThreshold),gradients);
        
        % Update the network parameters using the Adam optimizer.
        [parameters,trailingAvg,trailingAvgSq] = adamupdate(parameters,gradients, ...
            trailingAvg, trailingAvgSq, iteration, learnRate);
        
        if plots == "training-progress"
            % Plot training progress.
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            
            % Normalize the loss over the sequence lengths            
            loss = mean(loss ./ numTimeSteps);
            loss = double(gather(extractdata(loss)));
            loss = mean(loss);
            
            addpoints(lineLossTrain,iteration, mean(loss));

            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow
        end
    end
    
    % Reduce the learning rate after learnRateDropPeriod epochs
    if mod(epoch,learnRateDropPeriod) == 0
        learnRate = learnRate*learnRateDropFactor;
    end
end

% Save parameter and hyperparameter
cd E:\Project\mat\mat_tcn\TCN_parameter_train\parameter\2cats_NPA
fileName="parameters_2cats_NPA_CutAF_"+numFilters+"_"+numBlocks+"_"+filterSize+".mat";
fileName=char(fileName);
save(fileName,'parameters');
cd E:\Project\mat\mat_tcn\TCN_parameter_train\hyperparameter\2cats_NPA
fileName1="hyperparameters_2cats_NPA_CutAF_"+numFilters+"_"+numBlocks+"_"+filterSize+".mat";
fileName1=char(fileName1);
save(fileName1,'hyperparameters');

% save figure
cd E:\Project\mat\mat_tcn\TCN_results\2cats_NPA
fileName2="TCN_2cats_NPA_CutAF_"+numFilters+"_"+numBlocks+"_"+filterSize+".fig";
fileName2=char(fileName2);
savefig(fileName2);

%% Load parameter and hyperparameter
cd E:\Project\mat\mat_tcn\TCN_parameter_train
load('parameters_2cats_N2PA_175_12_8.mat');
load('hyperparameters_2cats_N2PA_175_12_8.mat');

%% Test model
cd E:\Project\mat\mat_tcn\TCN_samples\Test\2cats_NPA
t = load("ArrhythmiasTest_30sec_2cats_NPA_nonCutAF_200seg_V3.mat");
XTest = t.XTest;
YTest = t.YTest;
classes = categories(YTest{1});
numObservationsTest = numel(XTest);

cd E:\Project\mat\mat_tcn
[X1,Y1] = transformSequences(XTest,YTest);
dlXTest = dlarray(X1);

doTraining = false;
dlYPred = model(dlXTest,parameters,hyperparameters,doTraining);

YPred = gather(extractdata(dlYPred));

labelsPred = categorical(zeros(numObservationsTest,size(dlYPred,3)));
accuracy = zeros(1,numObservationsTest);

for i = 1:numObservationsTest
    [~,idxPred] = max(YPred(:,i,:),[],1);
    [~,idxTest] = max(Y1(:,i,:),[],1);
    
    labelsPred(i,:) = classes(idxPred)';
    accuracy(i) = mean(idxPred == idxTest);
end

mean(accuracy);
pred_nsr=find(accuracy(1:100)<0.5)
pred_af=find(accuracy(101:300)<0.5)
pred_pac=find(accuracy(301:400)<0.5)
sen=(200-length(pred_af))/200
spec=(200-length(pred_nsr)-length(pred_pac))/200
% pred_nsr=find(accuracy(1:100)<0.5)
% pred_af=find(accuracy(101:400)<0.5)
% pred_pac=find(accuracy(401:500)<0.5)
% pred_pvc=find(accuracy(501:600)<0.5)
% sen=(300-length(pred_af))/300
% spec=(300-length(pred_nsr)-length(pred_pac)-length(pred_pvc))/300
% pred_lessthan1=find(accuracy~=1 & accuracy~=0);



%%
% figure,
% idx = 13;
% plot(categorical(labelsPred(idx,:)),'.-')
% hold on
% plot(YTest{13})
% hold off
% 
% xlabel("Time Step")
% ylabel("Arrhythmia")
% title("Predicted Activities")
% legend(["Predicted" "Test Data"])
% 
% %%
% pred_lessthan1_1=pred_lessthan1(4:6);
% figure;
% % :length(pred_lessthan1)
% for num_lessthan1=1:3
%     subplot(3,1,num_lessthan1),plot(XTest{pred_lessthan1(num_lessthan1),1},...
%         'Color',[.8 .8 .8]), hold on
%     title("Segment: " + pred_lessthan1(num_lessthan1))
%     for num_data=1:length(XTest{1,:})
%         if (labelsPred(pred_lessthan1(num_lessthan1),num_d+ata)=="NSR")
%             plot(num_data,XTest{pred_lessthan1(num_lessthan1),1}(1,num_data),...
%                 '.','Color','r','MarkerSize',5)
%         elseif (labelsPred(pred_lessthan1(num_lessthan1),num_data)=="AF")
%             plot(num_data,XTest{pred_lessthan1(num_lessthan1),1}(1,num_data),...
%                 '.','Color','b','MarkerSize',5)
%         elseif (labelsPred(pred_lessthan1(num_lessthan1),num_data)=="PAC")
%             plot(num_data,XTest{pred_lessthan1(num_lessthan1),1}(1,num_data),...
%                 '.','Color','g','MarkerSize',5)
%         elseif (labelsPred(pred_lessthan1(num_lessthan1),num_data)=="PVC")
%             plot(num_data,XTest{pred_lessthan1(num_lessthan1),1}(1,num_data),...
%                 '.','Color','m','MarkerSize',5)
%         end
%     end
% end
% 
% %%
% % pred_lessthan1_1=pred_lessthan1(4:6);
% figure;
% % :length(pred_lessthan1)
% for num_lessthan1_1=1:3
%     subplot(3,1,num_lessthan1_1),plot(XTest{pred_lessthan1_1(num_lessthan1_1),1},...
%         'Color',[.8 .8 .8]), hold on
%     title("Segment: " + pred_lessthan1_1(num_lessthan1_1))
%     for num_data=1:length(XTest{1,:})
%         if (labelsPred(pred_lessthan1_1(num_lessthan1_1),num_data)=="Non-AF")
%             plot(num_data,XTest{pred_lessthan1_1(num_lessthan1_1),1}(1,num_data),...
%                 '.','Color','r','MarkerSize',5)
%         elseif (labelsPred(pred_lessthan1_1(num_lessthan1_1),num_data)=="AF")
%             plot(num_data,XTest{pred_lessthan1_1(num_lessthan1_1),1}(1,num_data),...
%                 '.','Color','b','MarkerSize',5)
%         end
%     end
% end