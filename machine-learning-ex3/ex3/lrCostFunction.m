function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 
% disp(y);
% disp(X);
% disp(theta); %4*1
% disp(size(theta)); %4*1
% disp(size(X)); %5*4
% disp(size(y)); %5*1
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


% %J = mean(-y*log(1/(1+exp(-x(i,:)*theta)-(1-y
% for i = 1:m
% %     disp(y(i,:));
% %     disp(X(i,:));
%     h = 1/(1+exp(-X(i,:)*theta));
%     J = J-y(i)*log(h)-(1-y(i))*log(1-h);
% end
% 
% n = size(X,2);
% 
% for i = 2:n
% %     disp(theta(i));
% %     disp("!!!");
%     J = J + theta(i)*theta(i)/2*lambda;
% end
% J = J / m;
% 
% 
% for j = 1:n
%     tot = 0;
%     for i = 1:m
%         h = 1/(1+exp(-X(i,:)*theta));
%         val = (h - y(i))*X(i,j);        
%         tot = tot+val;
%     end
%     grad(j) = tot/m;
% end
% for j = 2:n
%     grad(j) = grad(j) + lambda/m*theta(j);
% end
% 
% %grad = 1/m* X' * (1/(1+exp(-X*theta))-y) + lambda/m*thet;




T = theta;
T(1) = 0;
S = sigmoid(X * theta);
J = ( (-y' * log(S)) - ((1 - y') * log(1-S)) ) / m + lambda / (2 * m) * sum(T .^ 2);
grad = (S - y)' * X / m + lambda / m * T';

% =============================================================

grad = grad(:);

end
