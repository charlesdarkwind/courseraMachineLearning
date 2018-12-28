function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

m = length(y);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    h = X * theta;
    err = h - y;
    theta = theta - alpha * (1/m) * (X' * err);
    cost = computeCost(X, y, theta);
    J_history(iter) = cost;
end

end
