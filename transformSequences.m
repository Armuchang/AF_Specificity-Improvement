function [XTransformed, YTransformed, numTimeSteps] = transformSequences(X,Y)

numTimeSteps = cellfun(@(sequence) size(sequence,2),X);

miniBatchSize = numel(X);
numFeatures = size(X{1},1);
sequenceLength = max(cellfun(@(sequence) size(sequence,2),X));
classes = categories(Y{1});
numClasses = numel(classes);

sz = [numFeatures miniBatchSize sequenceLength];
XTransformed = zeros(sz,'single');

sz = [numClasses miniBatchSize sequenceLength];
YTransformed = zeros(sz,'single');

for i = 1:miniBatchSize
    predictors = X{i};
    
    % Create dummy labels.
    responses = zeros(numClasses, numTimeSteps(i), 'single'); 
    for c = 1:numClasses
        responses(c,Y{i}==classes(c)) = 1;
    end
    
    % Left pad.
    XTransformed(:,i,:) = leftPad(predictors,sequenceLength);
    YTransformed(:,i,:) = leftPad(responses,sequenceLength);
end

end