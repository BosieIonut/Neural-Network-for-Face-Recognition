function [rez] = predict(W,V,P,c,b,d,x)
rez = sigmoid(P*sigmoid(V*sigmoid(W*x + c) + b)+d);
end