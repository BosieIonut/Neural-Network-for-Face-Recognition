w_g_c;
    faceDetector = vision.CascadeObjectDetector();
    imagine2 = imread('test13.jpg');
    [width, height] = size(imagine);
    if width>320
       imagine2 = imresize(imagine2,[320 NaN]);
    end
    imagine = rgb2gray(imagine2);
    
    face_Location = step(faceDetector, imagine);
    if isempty(face_Location) || size(face_Location,1) > 1
        disp(nume);
    end
    byass = int64(face_Location(3) * 10 / 100);
    face = imagine(face_Location(2) - byass:face_Location(2)+face_Location(4)+byass,face_Location(1) - byass:face_Location(1)+face_Location(3) + byass);
    face_Location(1) = face_Location(1) - byass;
    face_Location(2) = face_Location(2) - byass;
    face_Location(3) =face_Location(3) + byass;
    face_Location(4) = face_Location(3);
    imagine2 = insertShape(imagine2, 'Rectangle', face_Location);
    face = imresize(face,[64 64]);
    t(1:64*64) = double(reshape(face',1,[]));
    t(64*64+1) = 1;
  
    rasp = sigmoid(w_g_c'*t');
    figure
    imshow(imagine2)
    if rasp < 0.5
        rasp = num2str(100-100*rasp);
        disp( ['Sunt ' rasp ' % sigur ca este corect'] ) ;
        title('clar e ionut asta');
    else
        rasp = num2str(100*rasp);
        disp( ['Sunt ' rasp ' % sigur ca este corect'] ) ;
        title('clar e dumitru asta');
    end