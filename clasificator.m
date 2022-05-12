clc;clear;close all
N_dum = 22;
N_ionut = 20;

epsilon = 0.001; % marja eroarea 
maxIter = 100000; % nr maxim de iteratii
faceDetector = vision.CascadeObjectDetector();
Ionut_found_faces = 0;
Dumi_found_faces = 0;

x = zeros(64*64+1,N_ionut + N_dum);


for i=1:N_dum
    nume = ['dum/' num2str(i) '.jpg'];
    
    imagine = rgb2gray(imread(nume));
    [width, height] = size(imagine);
    if width>320
       imagine = imresize(imagine,[320 NaN]);
    end

    face_Location = step(faceDetector, imagine);


    if isempty(face_Location) || size(face_Location,1) > 1
        disp(nume);
        continue; 
    end
    Dumi_found_faces = Dumi_found_faces + 1;
    face = imagine(face_Location(2):face_Location(2)+face_Location(4),face_Location(1):face_Location(1)+face_Location(3));
     imagine = insertShape(imagine, 'Rectangle', face_Location);
     %imshow(imagine)
     
     figure;
     
     face = imresize(face,[64 64]);
     imshow(face);
     x(1:64*64,Dumi_found_faces) = reshape(face',1,[])';
     x(64*64+1,Dumi_found_faces) = 1;
end
close all
for i=1:N_ionut
    nume = ['ion/' num2str(i) '.jpg'];
    
    imagine = rgb2gray(imread(nume));
    [width, height] = size(imagine);
    if width>320
       imagine = imresize(imagine,[320 NaN]);
    end

    face_Location = step(faceDetector, imagine);


    if isempty(face_Location) || size(face_Location,1) > 1
        disp(nume);
        continue; 
    end
    Ionut_found_faces = Ionut_found_faces + 1;
    face = imagine(face_Location(2):face_Location(2)+face_Location(4),face_Location(1):face_Location(1)+face_Location(3));
     imagine = insertShape(imagine, 'Rectangle', face_Location);
     %imshow(imagine)
     
     figure;
     
     face = imresize(face,[64 64]);
     imshow(face);
    
    x(1:64*64,Ionut_found_faces + Dumi_found_faces) = reshape(face',1,[])';
    x(64*64+1,Ionut_found_faces + Dumi_found_faces) = 1;
end
if(N_dum > Dumi_found_faces || N_ionut > Ionut_found_faces)
    disp('In unele poze nu au fost gasite fete sau au fost gasite mai mult de una, deci nu vor fi luate in considerare');
else
    disp('Totul este in regula');
end
n = Ionut_found_faces + Dumi_found_faces;
y = [ones(1,Dumi_found_faces) zeros(1,Ionut_found_faces)]';
F = @(w) 1/n * (-1 * y' * log(sigmoid(x' * w)) - (ones(n, 1) - y)' * log(1 - sigmoid(x' * w)));
[w_g_c,vect_g_c,n_g_c] = gradient(x,n,y,epsilon,maxIter);
plot(vect_g_c);
close all;
disp('Finished');