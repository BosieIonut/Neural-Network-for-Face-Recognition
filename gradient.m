function [w,vect,nr_pasi] = gradient(x,n,y,epsilon,maxIter)
    w = zeros(1,64*64+1);
    w = w';
    L = 0;
    for i=1:n
        L = L + norm(x(:,i),2)^2;
    end
    L = L / 4;
    L
    alpha = 1/L;
    plotmax = 500;
    vect = zeros(1,plotmax);
    i = 1; 
    while true
        h = sigmoid(x'*w);
        gradF = (1/n)*x*(h-y);
        w = w - alpha * gradF;
       
        if(norm(gradF,2) < epsilon || i > maxIter)
            break;
        end
        if(i<plotmax)
        vect(i) = norm(gradF,2);
        
        end
    i=i+1;
    end
    nr_pasi = i;
end

