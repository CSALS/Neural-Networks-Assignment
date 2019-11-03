%Removing any previous code related variables,commands%
clc;
clear all;
close all;
%Loading the dataset%
load data.mat;
X = x(:,1:72);
X = (X - mean(X,1))/std(X,1);
Y = x(:,73);
%Converting single column output into its binarized version%
for i=1:length(Y)
     if(Y(i) == 0)
        z(i,:) = [1 0];
     else if(Y(i) == 1)
       z(i,:) = [0 1];
    end
    end
end
%Holdout approach 70-30
trainInput = [];
trainOutput = [];
testInput = [];
testOutput = [];
p = randperm(size(X,1)); %returns random permutation of numbers from 1 to size(X,1)
p = randperm(size(X,1));
p = randperm(size(X,1));
trainSize = uint64((7*size(X,1))/10);
for j = 1:trainSize
    trainInput = [trainInput;X(p(j),:)];
    trainOutput = [trainOutput;z(p(j),:)];
end
for j = trainSize+1:size(X,1)
    testInput = [testInput;X(p(j),:)];
    testOutput = [testOutput;z(p(j),:)];
end

fprintf('70-30 hold-out approach results :- \n') 
[W1,b1,W2,b2,W3,b3] = train(trainInput,trainOutput);
accuracy = test(testInput,testOutput,W1,b1,W2,b2,W3,b3)
%5-fold cross validation%
c = cvpartition(size(X,1),'kfold',5);
for partitionIndex = 1:5
    A = training(c,partitionIndex);
    trainInput = [];
    trainOutput = [];
    testInput  = [];
    testOutput = [];
    for dataIndex = 1:size(A,1)
        if(A(dataIndex) == 1)
            trainInput = [trainInput;X(dataIndex,:);];
            trainOutput = [trainOutput;z(dataIndex,:);];
        else if(A(dataIndex) == 0)
            testInput = [testInput;X(dataIndex,:);];
            testOutput = [testOutput;z(dataIndex,:);];    
        end
        end
    end
    fprintf('Fold %d results :- \n',partitionIndex)
    [W1,b1,W2,b2,W3,b3] = train(trainInput,trainOutput);
    accuracy = test(testInput,testOutput,W1,b1,W2,b2,W3,b3)
end
%Activation Function
function A = sigmoid(Z)
    A = 1./(1 + exp(-Z));
end
function Aderivative = sigmoidDerivative(A)
    Aderivative = A.*(1 - A);
end
%
function [W1,b1,W2,b2,W3,b3] = train(train_X,train_Y)
    %Parameters%
    inputNeurons = size(train_X,2);
    hiddenNeurons1 = 10; %neurons of hidden layer 1
    hiddenNeurons2 = 5; %neurons of hidden layer 2
    instances = size(train_X,1);
    features = size(train_X,2);
    classes = size(train_Y,2);
    alpha = [0.2,0.2,0.2];
    num_iterations = 2000;
    %initialization of weight and bias%
    W1 = rand(hiddenNeurons1,features);
    b1 = ones(hiddenNeurons1,1);
    W2 = rand(hiddenNeurons2,hiddenNeurons1);
    b2 = ones(hiddenNeurons2,1);
    W3 = rand(classes,hiddenNeurons2);
    b3 = ones(classes,1);
    X = train_X';
    Y = train_Y';
    %Iterations%
    for iteration = 1:num_iterations
        %Forward propagation%
        Z1 = W1*X + b1;
        A1 = sigmoid(Z1);
        Z2 = W2*A1 + b2;
        A2 = sigmoid(Z2);
        Z3 = W3*A2 + b3;
        A3 = sigmoid(Z3);
        %Backpropagation%
        %1. Calculate Deltas
        delta3 = (Y - A3).*sigmoidDerivative(A3);
        delta2 = (W3'*delta3).*sigmoidDerivative(A2);
        delta1 = (W2'*delta2).*sigmoidDerivative(A1);
        %2. Update weights and bias
        W3 = W3 - alpha(3)*(delta3*A2');
        W2 = W2 - alpha(2)*(delta2*A1');
        W1 = W1 - alpha(1)*(delta1*X');
        b3 = b3 - alpha(3)*(sum(delta3,2));
        b2 = b2 - alpha(2)*(sum(delta2,2));
        b1 = b1 - alpha(1)*(sum(delta1,2));
    end
    %plot(iterations,costs)
end

function accuracy = test(test_X,test_Y,W1,b1,W2,b2,W3,b3)
    %Do forward propagation using test data and A3 is our prediction%
    X = test_X';
    Z1 = W1*X + b1;
    A1 = sigmoid(Z1);
    Z2 = W2*A1 + b2;
    A2 = sigmoid(Z2);
    Z3 = W3*A2 + b3;
    A3 = sigmoid(Z3);
    Y_predicted = A3';
    instances = size(test_X,1);
    %Compare our prediction with actual outputs%
    count = 0;
    for testIndex = 1:instances
            [~,predictedMaxIndex] = max(Y_predicted(testIndex,:));
            [~,testMaxIndex] = max(test_Y(testIndex,:));
            if(predictedMaxIndex == testMaxIndex)
                count = count + 1;
            end
    end
    accuracy = (count*100)/instances;
end