function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters


    % ====================== YOUR CODE HERE ======================

    h = X * theta;
    err = h - y;

    theta = theta - alpha * (1/m) * sum(err .* X)';

    % theta = theta - alpha * (1/m) * (X' * err);


    % ============================================================
    cost = computeCost(X, y, theta);

    % Save the cost J in every iteration
    J_history(iter) = cost;

    % fprintf('\n%f  \n%f', theta, cost);

end

end
