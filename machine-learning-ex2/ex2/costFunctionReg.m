function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

disp("$$$$$$$$$");
disp(theta);
disp("$$$$$$$$$");

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

n = length(theta); %n=3, 3 features
for i=1:m
    h = 1/(1+exp(-X(i,:)*theta));
    J = J-y(i)*log(h)-(1-y(i))*log(1-h);
end

for j=2:n
    J = J+lambda*theta(j,1)*theta(j,1)/2;
end
% X=100*3
J=J/m;

for i=1:n
    tot = 0;
    for j=1:m
        h = 1/(1+exp(-X(j,:)*theta));
        tot = tot + (h-y(j)) * X(j,i);
    end
%     disp("i=",i," tot=",tot,"gradi=",grad(i));
    res = tot/m;
%     disp(res);
    grad(i) = res;%tot/m;
end

% for i=2:n
%      res = grad(i)+lambda*theta(i,1)/m;
%      grad(i) = res;
% end
% =============================================================

end
