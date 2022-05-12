w_g_c;
    faceDetector = vision.CascadeObjectDetector();
    imagine2 = imread('11.jpg');
    [width, height] = size(imagine);
    if width>320
       imagine2 = imresize(imagine2,[320 NaN]);
    end
    imagine = rgb2gray(imagine2);
    
    face_Location = step(faceDetector, imagine);
    if isempty(face_Location) || size(face_Location,1) > 1
        disp(nume);
    end
    face = imagine(face_Location(2):face_Location(2)+face_Location(4),face_Location(1):face_Location(1)+face_Location(3));
    imagine2 = insertShape(imagine2, 'Rectangle', face_Location);
    face = imresize(face,[64 64]);
    t(1:64*64) = double(reshape(face',1,[]));
    t(64*64+1) = 1;
  
    rasp = sigmoid(w_g_c'*t')
    imshow(imagine2)
    if rasp < 0.5 
        title('clar e ionut asta');
    else
        
        title('clar e dumitru asta');
    end