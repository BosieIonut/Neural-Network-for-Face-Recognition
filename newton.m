function [w,vect,nr_pasi] = newton(x,n,y,epsilon,maxIter)
w = zeros(1,64*64+1)';
i=1;
    plotmax = 50;
      alfa = 0.01;
    vect = zeros(plotmax,1);
while true
    h = sigmoid(x'*w);
    q = h.*(1-h);
    Q = diag(q);
    gradF = (1/n)*x*(h-y);
    hesianaF = (1/n)* x*Q*x';
  
    w = w - alfa* hesianaF\gradF;
     norm(gradF,2)
    if(norm(gradF) < epsilon || i>maxIter)
            break;
    end
    if i<plotmax
       vect(i) = norm(gradF,2);
    end
    i = i+1;
end
nr_pasi = i;
end

