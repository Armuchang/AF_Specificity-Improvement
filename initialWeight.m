function initialParameters = initialWeight(dimension)
% Use this function for initializing weight and bias of model
% Numeric type is single (32 bit).
initialParameters = randn(dimension,'single').*0.01;
end

