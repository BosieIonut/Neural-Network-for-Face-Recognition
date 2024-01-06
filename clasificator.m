load('poze.mat');
epsilon = 0.1;
maxIter = 100000;
X = [X; ones(1,n)];
[w,vect,nr] = gradient(X(:,1:n),n,Y,epsilon,maxIter);
plot(vect);
save('LR.mat','w');
disp('Finished');