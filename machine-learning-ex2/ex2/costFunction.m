function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % m=100% number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));%theta: 3*1 initial=[0;0;0]

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

for i=1:m
    h = 1/(1+exp(-X(i,:)*theta));
    J = J-y(i)*log(h)-(1-y(i))*log(1-h);
end
% X=100*3
J=J/m;
n = length(theta); %n=3, 3 features
for i=1:n
    tot = 0;
    for j=1:m
        h = 1/(1+exp(-X(j,:)*theta));
        tot = tot + (h-y(j)) * X(j,i);
    end
%     disp("i=",i," tot=",tot,"gradi=",grad(i));
    res = tot/m;
    disp(res);
    grad(i) = res;%tot/m;
end




% =============================================================

end
