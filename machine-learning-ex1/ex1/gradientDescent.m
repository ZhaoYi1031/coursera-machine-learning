function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
% disp(J_history)



for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    d1 = 0;
    d2 = 0;
    for i = 1:m
        d1 = d1 + 2*theta(1) + 2*(theta(2)*X(i,2) - y(i));
        d2 = d2 + 2*theta(2)*X(i,2)*X(i,2) + 2*(theta(1)-y(i))*X(i,2);
    end
    d1 = d1/(2*m);
    d2 = d2/(2*m);
    tmp1 = theta(1) - alpha*d1;%theta(1) - alpha*theta(1);
    tmp2 = theta(2) - alpha*d2;%theta(2) - alpha*theta(2);

    theta = [tmp1; tmp2];

%     H = X*theta-y;
%     theta(1)=theta(1)-alpha*(1/m)*sum(H.*X(:,1));
%     theta(2)=theta(2)-alpha*(1/m)*sum(H.*X(:,2));
    % ============================================================

    % Save the cost J in every iteration  
    disp(X)
    c = computeCost(X, y, theta);
    J_history(iter) = c;
   
    
end

end
