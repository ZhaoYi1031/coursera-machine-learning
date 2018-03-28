function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

row = size(X, 2);
for i = 1:1:m%1:m
    %disp("i="+i)
    ytheta = 0;
    for j = 1:row
        ytheta = ytheta + theta(j) * X(i,j);%theta(1) + theta(2)*X(i)
    end
    J = J + (ytheta-y(i))*(ytheta-y(i))/(m*2);
end

% =========================================================================

end
