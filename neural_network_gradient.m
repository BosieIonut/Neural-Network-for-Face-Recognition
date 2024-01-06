%% TRAIN
clc; clear; close all;
load('poze.mat');
W = rand(128,64*64) - 0.5;
V = rand(64,128) - 0.5;
P = rand(1,64) - 0.5;
c = rand(128,1) - 0.5;
b = rand(64,1) - 0.5;
d = rand(1,1) - 0.5;

alfa = 0.01;
epoch = 50000;
error = zeros(1,epoch/1000+1);

   % for i=1:n

for k=1:epoch
        o1 = W*X + c;
        o1a = sigmoid(o1);
        o2 = V*o1a+b;
        o2a = sigmoid(o2);
        o3 = P*o2a+d;
        o3a = sigmoid(o3);
        if(mod(k,1000) < 0.001 || k == 1)
            error(floor(k/1000) + 1) = error(floor(k/1000)+1) + sum(loss(Y', o3a));
            disp(k);
            disp(error(floor(k/1000) + 1));
        end
        %%Layer 3 (activation layer)
        gradient_E = loss_derivat(Y',o3a) .* sigmoid_derivat(o3);
        %%LAYER 3 (hidden layer)
        P_gradient = gradient_E * o2a';
        d_grad = gradient_E;
        gradient_E = P' * gradient_E;
        %%Layer 2 (activation layer)
        gradient_E = gradient_E .* sigmoid_derivat(o2);
        %%Layer 2 (hidden layer)
        V_gradient = gradient_E * o1a';
        %V = V - alfa*weights_gradient;
        %b = b - alfa*gradient_E;
        b_grad = gradient_E;
        gradient_E = V' * gradient_E;
        %%Layer 1(activation layer)
        gradient_E = gradient_E .* sigmoid_derivat(o1);
        %%Layer 1 (hidden layer)
        W_gradient = gradient_E * X';
        c_grad = gradient_E;
        
        P = P-alfa*P_gradient;
        V = V-alfa*V_gradient;
        W = W - alfa*W_gradient;
        b = b - alfa*sum(b_grad,2) ;
        c = c - alfa*sum(c_grad,2) ;
        d = d - alfa*sum(d_grad,2);
    
   % end
end
save('net_grad.mat','W','V','P','c','b','d');
%% 