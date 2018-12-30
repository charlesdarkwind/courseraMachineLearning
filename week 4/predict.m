function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%

X = [ones(m, 1) X];
z2 = X*Theta1';
a2 = sigmoid(z2); % activation layer is  5000 x 25
% Then add bias unit col to preceding layer for 5000 x 26
a2 = [ones(size(a2, 1), 1) a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);
h = a3;
[~, index] = max(h, [], 2);
p = index;

% =========================================================================


end
