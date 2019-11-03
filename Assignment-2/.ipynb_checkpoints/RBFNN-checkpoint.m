%Removing any previous code related variables,commands%
clc;
clear all;
close all;
%Loading the dataset%
load data.mat;
X = x(:,1:72);
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
[W,k,mu,sigma,name] = train(trainInput,trainOutput,"Gaussian");
accuracy = test(testInput,testOutput,W,k,mu,sigma,name);
fprintf('1. Gaussian Function gives accuracy = %.10f \n',accuracy)
[W,k,mu,sigma,name] = train(trainInput,trainOutput,"MultiQuadric");
accuracy = test(testInput,testOutput,W,k,mu,sigma,name);
fprintf('2. MultiQuadric Function gives accuracy = %.10f \n',accuracy)
[W,k,mu,sigma,name] = train(trainInput,trainOutput,"Linear");
accuracy = test(testInput,testOutput,W,k,mu,sigma,name);
fprintf('3. Linear Function gives accuracy = %.10f \n',accuracy)
[W,k,mu,sigma,name] = train(trainInput,trainOutput,"Cubic");
accuracy = test(testInput,testOutput,W,k,mu,sigma,name);
fprintf('4. Cubic Function gives accuracy = %.10f \n',accuracy)

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
    [W,k,mu,sigma,name] = train(trainInput,trainOutput,"Gaussian");
    accuracy = test(testInput,testOutput,W,k,mu,sigma,name);
    fprintf('1. Gaussian Function gives accuracy = %.10f \n',accuracy)
    [W,k,mu,sigma,name] = train(trainInput,trainOutput,"MultiQuadric");
    accuracy = test(testInput,testOutput,W,k,mu,sigma,name);
    fprintf('2. MultiQuadric Function gives accuracy = %.10f \n',accuracy)
    [W,k,mu,sigma,name] = train(trainInput,trainOutput,"Linear");
    accuracy = test(testInput,testOutput,W,k,mu,sigma,name);
    fprintf('3. Linear Function gives accuracy = %.10f \n',accuracy)
    [W,k,mu,sigma,name] = train(trainInput,trainOutput,"Cubic");
    accuracy = test(testInput,testOutput,W,k,mu,sigma,name);
    fprintf('4. Cubic Function gives accuracy = %.10f \n',accuracy)

end

function [W,k,mu,sigma,name] = train(train_X,train_Y,name)
    %Parameters%
    k = 10; %hidden neurons
    instances = size(train_X,1);
    %Using K means for defining hidden neurons%
    [l,mu] = kmeans(train_X,k); %l contains class of each vector in X,mu contains k cluster centers
    %Evaluate sigma of each hidden neuron%
    sigma = zeros(k,1);
    count = zeros(k,1); %number of instances in each of the K cluster
    for trainIndex = 1:size(l,1)
        clusterIndex = l(trainIndex);
        count(clusterIndex) = count(clusterIndex) + 1;
        sigma(clusterIndex) = sigma(clusterIndex) + norm(train_X(trainIndex),mu(clusterIndex));
    end
    for clusterIndex = 1:k
        sigma(clusterIndex) = sigma(clusterIndex)/count(clusterIndex);
    end
    %Evaluate Hidden Layer Matrix%
    H = zeros(instances,k);
    for trainIndex = 1:instances
        for clusterIndex = 1:k
            H(trainIndex,clusterIndex) = kernelFunction(train_X(trainIndex,:),mu(clusterIndex,:),sigma(clusterIndex),name);
        end
    end
    %Evalute Weight Matrix%
    W = pinv(H)*train_Y;
end


function accuracy = test(test_X,test_Y,W,k,mu,sigma,name)
    instances = size(test_X,1);
    H = zeros(instances,k);
    %evaluate hidden layer matrix for test input%
    for testIndex = 1:instances
        for clusterIndex = 1:k
            H(testIndex,clusterIndex) = kernelFunction(test_X(testIndex,:),mu(clusterIndex,:),sigma(clusterIndex),name);
        end
    end
    %evaluate output%
    count = 0;
    Y_predicted = H*W;
    for testIndex = 1:size(Y_predicted,1)
            [~,predictedMaxIndex] = max(Y_predicted(testIndex,:));
            [~,testMaxIndex] = max(test_Y(testIndex,:));
            if(predictedMaxIndex == testMaxIndex)
                count = count + 1;
            end
    end
    accuracy = (count*100)/instances;
end

function phi = kernelFunction(X,mu,sigma,name)
    if(name == "Gaussian")
        phi = exp(-norm(X - mu)^2/(sigma^2 * 2));
    else if(name == "MultiQuadric")
        phi = (norm(X - mu)^2 + sigma^2)^0.5;
    else if(name == "Linear")  
        phi = norm(X - mu);
    else if(name == "Cubic")
        phi = norm(X - mu)^3;
    end
    end
    end
    end
end
