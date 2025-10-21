%function from Charbel's code
function [W1, W2,b1,b2] = Backpropag(W1, W2, b1, b2, inputs, Price)
 alpha = 0.5;                   %learning rate
 N = length(inputs(:,1));            %dataset size
 iterations = 1;
 for j = 1:iterations
    for k = 1:N
        x = inputs(k, :)';           %inputs
        d = Price(k);               %true price
        v1 = W1*x+b1;           
        y1 = Sigmoid(v1);
        v = W2*y1+b2;
        y = Sigmoid(v);         %y_hat, predicted price
        e = d - y;              %loss from 1-norm
        delta = y.*(1-y).*e;    %e*dsig(y)
        e1 = W2'*delta;
        delta1 = y1.*(1-y1).*e1;
        dW1 = alpha*delta1*x';
        W1 = W1 + dW1;
        db1 = alpha*delta1;
        b1 = b1 + db1;
        dW2 = alpha*delta*y1';
        W2 = W2 + dW2;
        db2 = alpha*delta;
        b2 = b2 + db2;
    end
 end
end