function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.


X = [ones(m, 1) X]; % Add bias col
yArr = (y == 1:num_labels); % destructure y into logical arrays, output like nn


% Forward propagation method 1
a1 = X;
z2 = X*Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2, 1), 1) a2]; % add bias col
z3 = a2*Theta2';
a3 = sigmoid(z3);
h = a3;


for t = 1:m

    % Forward propagation method 2
    a_1 = X(t,:); %  1 x 401
    z_2 = a_1*Theta1'; % 1 x 25
    a_2 = sigmoid(z_2); % 1 x 25
    a_2 = [ones(size(a_2, 1), 1) a_2]; % 1 x 26
    z_3 = a_2*Theta2';
    a_3 = sigmoid(z_3); % 1 x 10

    % Back propagation
    d3 = zeros(size(num_labels)); % 1 x 10

    for k = 1:num_labels
      d3(k) = a_3(k) - yArr(t, k);
    end

    d2 = (Theta2' * d3')' .* a_2 .* (1-a_2); % 26x1
    d2 = d2(2:end); % 1 x 25
                              % 10x1 * 1x26 = 10x26
    Theta2_grad = Theta2_grad + (d3' * a_2);
                              % 25x1 * 1x401 = 25x401
    Theta1_grad = Theta1_grad + (d2' * a_1);
end

% Regularize non-bias grads
Theta2_grad = [Theta2_grad(:, 1)    Theta2_grad(:, 2:end) + lambda * Theta2(:, 2:end)] / m;
Theta1_grad = [Theta1_grad(:, 1)    Theta1_grad(:, 2:end) + lambda * Theta1(:, 2:end)] / m;

% cost function

% Regularization term
regTheta1 = sum(Theta1(:, 2:end).^2); % For each theta matrices, simply sum all their squared values
regTheta2 = sum(Theta2(:, 2:end).^2);
reg = (lambda/(2*m)) * (sum(regTheta1) + sum(regTheta2));

for i = 1:m % num of samples
    for k = 1:num_labels % num of classes
        J = J + (-yArr(i, k)*log(h(i, k))-(1-yArr(i, k))*log(1-h(i, k)));
    end

end

J = (J / m) + reg;


% -------------------------------------------------------------

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
