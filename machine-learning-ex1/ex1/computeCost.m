function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

for i = 1:1:m%1:m
    %disp("i="+i)
    ytheta = theta(1) * X(i,1) + theta(2) * X(i,2);%theta(1) + theta(2)*X(i)
    J = J + (ytheta-y(i))*(ytheta-y(i))/(m*2)

% disp(X)
% disp(theta)

% h = X*theta - y;
% J = 1/(2*m) * sum(h.^2);

% =========================================================================

end