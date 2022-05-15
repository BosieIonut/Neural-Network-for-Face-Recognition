function [w,vect,nr_pasi] = gradient_ideal(x,n,y,epsilon,maxIter,F)
   w = zeros(1,64*64+1)';
    plotmax = 500;
    vect = zeros(1,plotmax);
    i = 1;
    while true
        h = sigmoid(x'*w);
        gradF = (1/n)*x*(h-y);
       
        arg = @(Alpha) F(w - Alpha * gradF);
        alpha = fminbnd(arg,0,1);
         w = w - alpha * gradF;
        if(norm(gradF,2) < epsilon || i > maxIter)
            break;
        end
        if(i<plotmax)
            vect(i) = norm(gradF,2);
        end
        i=i+1;
    end
    nr_pasi = gradF;
end
