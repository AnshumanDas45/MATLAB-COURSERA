function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X*theta;
theta1 = [0 ; theta(2:end, :)]; %as we know we don't regularize theta(0) hence in the operations we want to remove the term theta(0) hence we make the theta matrix as theta(2:end,:) which means theta is the matrix with rows from 2 to end (as theta(0) is in the first row) and all the columns 
p = lambda*(theta1'*theta1)*(1/(2*m));
J = (1/(2*m))*(sum((h-y).^2)) + p;
%Gradient
grad = (1/m)*(X'*(h-y)+lambda*theta1);












% =========================================================================

grad = grad(:);

end
