% first hidden layer computation
% generating the NN variables
% let J be the number of neurons
J = 32;
% generating weights and bias for the first hidden layer
W1 = randn(J,5);
b1 = randn(J,1);
% generating weights and bias for the second hidden layer
W2 = randn(1,J);
b2 = randn;