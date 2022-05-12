function [w,vect,nr_pasi] = newton_ideal(x,n,y,epsilon,maxIter,F)
w = [0,0,0]';
i=1;
    plotmax = 10;
    vect = zeros(plotmax,1);
while true
    h = sigmoid(x'*w);
    q = h.*(1-h);
    Q = diag(q);
    gradF = (1/n)*x*(h-y);
    hesianaF = (1/n)* x*Q*x';
    arg = @(Alpha) F(w - Alpha *inv(hesianaF)* gradF);
    alpha = fminbnd(arg,0,2);
    w = w - alpha* inv(hesianaF)*gradF;
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


