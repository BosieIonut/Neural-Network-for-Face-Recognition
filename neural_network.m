%% TRAIN
clc; clear; close all;
load('poze.mat');
W = rand(128,64*64) - 0.5;
V = rand(64,128) - 0.5;
P = rand(1,64) - 0.5;
c = rand(128,1) - 0.5;
b = rand(64,1) - 0.5;
d = rand(1,1) - 0.5;

alfa = 0.1;
epoch = 100000;
error = zeros(1,epoch/1000);
for k=1:epoch
    for i=1:n
        x = X(:,i);
        y = Y(i);
        o1 = W*x + c;
        o1a = sigmoid(o1);
        o2 = V*o1a+b;
        o2a = sigmoid(o2);
        o3 = P*o2a+d;
        o3a = sigmoid(o3);
        if(mod(k,1000) < 0.001)
            error(k/1000) = error(k/1000) + loss(y, o3a);
        end
        %%Layer 3 (activation layer)
        gradient_E = loss_derivat(y,o3a) .* sigmoid_derivat(o3);
        %%LAYER 3 (hidden layer)
        weights_gradient = gradient_E * o2a';
        P = P - alfa*weights_gradient;
        d = d - alfa*gradient_E;
        gradient_E = P' * gradient_E;
        %%Layer 2 (activation layer)
        gradient_E = gradient_E .* sigmoid_derivat(o2);
        %%Layer 2 (hidden layer)
        weights_gradient = gradient_E * o1a';
        V = V - alfa*weights_gradient;
        b = b - alfa*gradient_E;
        gradient_E = V' * gradient_E;
        %%Layer 1(activation layer)
        gradient_E = gradient_E .* sigmoid_derivat(o1);
        %%Layer 1 (hidden layer)
        weights_gradient = gradient_E * x';
        W = W - alfa*weights_gradient;
        c = c - alfa*gradient_E;
        
    end
     if(mod(k,1000) < 0.001)
         disp(k);
         disp(['eroarea este:' num2str(error(k/1000))]);
     end
end
save('net.mat','W','V','P','c','b','d');
%% 