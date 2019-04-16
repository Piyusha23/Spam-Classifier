function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

C = 1;
sigma = 0.3;

ct = [0.01,0.03,0.1,0.3,1,3,10,30];
s = [0.01,0.03,0.1,0.3,1,3,10,30];

% ct = [0.01,0.03,0.1];
% s = [0.01,0.03,0.1];

min_error = 1;

for i = 1:length(ct)
    for j = 1:length(s)
        model_train = svmTrain(X, y, ct(i), @(x1, x2) gaussianKernel(x1, x2, s(j))); 
        predictions = svmPredict(model_train, Xval);
        error =  mean(double(predictions ~= yval));
        if error < min_error;
            C = ct(i);
            sigma = s(j);
            min_error = error;
     end
end

end
