%project 3_02
%try to make everything from scratch
%NN option pricing project
%Cj Destefani, project and presentation due 12-5-22.
% Dr. Liu's Topics in Financial Mathematics, UD Financial Math.

%% Basic set up
SGD=true; %if true, uses stochastic GD, otherwise uses full GD.
smallSetSize=false; %which dataset to use

if smallSetSize
    load('Training01.mat', 'Price', 'inputs');  %2,000 option prices and inputs
else
    load('Training02.mat', 'Price', 'inputs'); %20,000 option prices and inputs
end

%Set hyperparameters
J=30; %width of the network
alpha=2*10^-3; %learning rate
M=20; %batch size
epoch_max=2*10^4; %training automatically stops after this many iterations
epsilon=5*10^-6; %training also stops if two sets of parameters are this close to each other

%activation functions
relu= @(x) max(0,x); %rectified linear unit activation function
drelu= @(x) (x>0)*1; %derivative of relu, is activation function: 1 if x>0, 0 otherwise.

% split data into training and testing
N=size(inputs,1); %total dataset size
L=zeros(epoch_max,1); %saves loss function at each epoch
ntheta=zeros(epoch_max,1); %saves norms of the difference between parameters at step l and l+1. 
epoch=0; %store epoch number
X=inputs; %rename so these can be played with without damaging original dataset
Y=Price;
Ntrain=floor(N*.8); %number of datapoints in training set ~80% of data to train, and 20% to test.
Ntest=N-Ntrain;
idx0=randperm(N,Ntrain); %pick out indexes of training data points
x_Tr=X(idx0,:); %use those idices to get input and output data
y_Tr=Y(idx0);
X(idx0,:)=0; %clear the ones set aside for training
Y(idx0)=0;
idx1=find(Y); %finds the index of the nonzero elements
x_Te=X(idx1,:); %put the rest in testing sets
y_Te=Y(idx1);
s=sqrt(Ntest)/2; %stretch or squish distribution of starting weights and biases

%% Run NN
%set initial weights
W0=randn(J,5)/s; %randomly initialize weights. W0 connects inputs and hidden layer
W1=randn(1,J)/s; %W1 is weights connecting hidden layer and output
b0=randn(J,1)/s; % biases for hidden layer
b1=randn(1,1)/s+1; % bias for output
dLdw1=zeros(J,1); %declare derivative vectors
dLdb0=zeros(J,1);
dLdw0=zeros(J,5);

while true
epoch=epoch+1; %increment counter

% Feed forward through network
u = W0*x_Tr' + b0; %first function
v = relu(u); %first activation
w = W1*v + b1; %final layer
y_hat = relu(w); %final activation

%calculate loss
prod0 = y_Tr'-y_hat; %this is not actually a product, but it is used in multiple places and will take less memory if given a name
L(epoch) = sum(prod0.^2)/Ntrain; %loss function is MSE, saved at each iteration

% calculate derivatives in order to perform either gradient descent or stochastic gradient descent
if SGD
    %incorporate batch size so that we use stochastic gradient descent:
    idx2=randperm(Ntrain,M); %pick out indexes of training data points
    u_short=u(:,idx2);
    v_short=v(:,idx2); %will be a MxJ matrix
    w_short=w(idx2);
    x_Tr_short=x_Tr(idx2,:);

    %back propagation with SGD--approximate derivatives
    prod1=2*(prod0(idx2)).*drelu(w_short);
    dLdb1=sum(prod1)/M; %same for both useRelu and ~useRelu since final activation is Relu in both cases
    dLdw1=v_short*prod1'/M;  %also same in both activation cases
    prod2=(W1'*prod1).*drelu(u_short);         %store these here to reduce total number of multiplications network performs
    dLdb0=mean(prod2,2);
    dLdw0=prod2*x_Tr_short/M;
    
    %{
    for j=1:J
        dLdw1(j)=mean(prod1.*v_short(j,:));
        prod2=prod1.*W1(j).*drelu(u_short(j,:));
        dLdb0(j)=sum(prod2)/M;
        for k=1:5
            dLdw0(j,k)=sum(prod2.*x_Tr_short(:,k)')/M; %depends on both j and k so is inside a loop for each
        end
    end
    %}
else
    %back propagation--gradient descent--uses 'exact' derivates. does not shorten any vectors, but otherwise operates the same as above.
    prod1=2*(prod0).*drelu(w);
    dLdb1=sum(prod1)/Ntrain;
    dLdw1=v*prod1'/Ntrain;  %dLdw1(j)=sum(prod1.*v(j,:))/Ntrain;
    prod2=(W1'*prod1).*drelu(u);
    for j=1:J
        %prod2=prod1.*(W1(j).*drelu(u(j,:)));
        dLdb0(j)=sum(prod2)/Ntrain;
        for k=1:5
            dLdw0(j,k)=sum(prod2.*x_Tr(:,k)')/Ntrain;
        end
    end
end

% iterate parameters with either gradient descent or Stochastic gradient descent
theta=[W0(:); b0; W1'; b1]; % 7J+1 parameters long
gradL=-1*[dLdw0(:); dLdb0; dLdw1; dLdb1]; %all derivatives of parameters
theta2 = theta- alpha*gradL;  %gradient descent step

%repackage parameters and use in next iteration
b1 = theta2(7*J+1); %last entry
W1 = theta2(6*J+1:7*J)'; % J entries before that
b0 = theta2(5*J+1:6*J); % J entries before that
W0 = reshape(theta2(1:5*J),[J,5]); %first 5J entries

%terminal conditions
ntheta(epoch)=norm(theta2-theta);
if and(ntheta(epoch)<epsilon, epoch>100)
    L(epoch+1:epoch_max)=[]; %remove extra empty entries if it ends early
    ntheta(epoch+1:epoch_max)=[]; 
    fprintf("Solution converged in %d iterations.\n",epoch)
    break;
end
if epoch>=epoch_max
    fprintf("Solution did not converge after %d iterations.\n", epoch)
    break;
end
end

%% evaluate test data set and plot results
y_fin=relu(W1*relu(W0*x_Te' + b0) + b1); %create predicted y values with test data
mseTest = sum((y_Te' - y_fin).^2)/Ntest;
fprintf("Final error: %2.6f\n",L(epoch))
fprintf("MSE on test data: %2.6f\n",mseTest)

figure
histogram(y_Te' - y_fin, 50)
xlabel('Error Distribution')
ylabel('Counts')
title('Error histogram: Test data - predictions')

figure
plot(y_Te,y_fin,'.',[min(y_Te),max(y_Te)],[min(y_Te),max(y_Te)],'r')
xlabel("True normalized option prices");
ylabel("Predicted normalized option prices");
title('Testing set')

figure
plot(log(1:epoch),log(L))
xlabel("Log Epoch");
ylabel("Log MSE Loss");
title('Loss diagram')

figure
plot(1:epoch,log(ntheta))
hold on
yline(log(epsilon),'r')
xlabel("Epoch");
ylabel("Log norm of diff between thetas at adjacent time steps");
title('Learning Rate and Convergence')