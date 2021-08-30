function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1); %it initialize J to be an zeros matrix of n rows and 1 column where n is the number of times we want to iterate gradient descent

for iter = 1:num_iters
h=X*theta;
theta = theta - (alpha/m)*( (h-y)'*X)';

    




    J_history(iter) = computeCost(X, y, theta);

end

end
