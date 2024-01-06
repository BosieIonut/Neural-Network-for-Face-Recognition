    load net_grad.mat
    load LR.mat
    close all;
    faceDetector = vision.CascadeObjectDetector();
    imagine2 = imread('test19.jpg');
    [width, height] = size(imagine2);
    if width>320
       imagine2 = imresize(imagine2,[320 NaN]);
    end
    imagine = rgb2gray(imagine2);
    
    face_Location = step(faceDetector, imagine);
    if isempty(face_Location) || size(face_Location,1) > 1
        disp('Nu s-a gasit o fata');
    end
    bias = int64(face_Location(3) * 10 / 100);
    %bias = 0;
    face = imagine(face_Location(2) - bias:face_Location(2)+ ...
        face_Location(4)+bias,face_Location(1) - bias:face_Location(1)+ ...
        face_Location(3) + bias);
    face_Location(1) = face_Location(1) - bias;
    face_Location(2) = face_Location(2) - bias;
    face_Location(3) = face_Location(3) + bias;
    face_Location(4) = face_Location(3);
    imagine2 = insertShape(imagine2, 'Rectangle', face_Location);
    face = imresize(face,[64 64]);
    t(1:64*64) = double(reshape(face',1,[]));
  
    rasp = predict(W,V,P,c,b,d,t(1:64*64)');
    figure
    subplot(1,2,1);
    imshow(imagine2)
    if rasp < 0.5
        rasp = num2str(100*(1-rasp)^2);
        disp( ['Sunt ' rasp ' % sigur ca este corect'] ) ;
        title('(NN)clar e ionut asta');
    else
        rasp = num2str(100*rasp^2);
        disp( ['Sunt ' rasp ' % sigur ca este corect'] ) ;
        title('(NN)clar e dumitru asta');
    end
    %figure
    t(64*64+1) = 1;
    rasp = sigmoid(w'*t');
    subplot(1,2,2);
        imshow(imagine2)
    if rasp < 0.5
         rasp = num2str(100*(1-rasp)^2);
        disp( ['Sunt ' rasp ' % sigur ca este corect'] ) ;
        title('(LR)clar e ionut asta');
    else
        rasp = num2str(100*rasp^2);
        disp( ['Sunt ' rasp ' % sigur ca este corect'] ) ;
        title('(LR)clar e dumitru asta');
    end