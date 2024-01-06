%% SETUP
clc; close all;

N_dum = 26;
N_ionut = 26;

faceDetector = vision.CascadeObjectDetector();

Ionut_found_faces = 0;
Dumi_found_faces = 0;

X = zeros(64*64,N_ionut + N_dum);
Y = [zeros(1,N_dum) zeros(1,N_ionut)]';
%% FIND DUMITRU FACES

for i=1:N_dum
    nume = ['dum/' num2str(i) '.jpg'];
    imagine = rgb2gray(imread(nume));
    [width, ~] = size(imagine);
    if width>320
       imagine = imresize(imagine,[320 NaN]);
    end

    face_Location = step(faceDetector, imagine);


    if isempty(face_Location) || size(face_Location,1) > 1
        disp(nume);
        continue; 
    end
    bias = int64(face_Location(3) * 10 / 100);
    %bias = 0;
    Dumi_found_faces = Dumi_found_faces + 1;
    face = imagine(face_Location(2) - bias:face_Location(2)+ ...
        face_Location(4)+bias,face_Location(1) - bias: ...
        face_Location(1)+face_Location(3) + bias);
    %imagine = insertShape(imagine, 'Rectangle', face_Location);
    figure;
     
    face = imresize(face,[64 64]);
    imshow(face);
    X(1:64*64,2*Dumi_found_faces) = reshape(face',1,[])';
    Y(2*Dumi_found_faces) = 1;
end
%%  FIND IONUT FACES
close all
for i=1:N_ionut
    nume = ['ion/' num2str(i) '.jpg'];
    
    imagine = rgb2gray(imread(nume));
    [width, ~] = size(imagine);
    if width>320
       imagine = imresize(imagine,[320 NaN]);
    end

    face_Location = step(faceDetector, imagine);


    if isempty(face_Location) || size(face_Location,1) > 1
        disp(nume);
        continue; 
    end
    Ionut_found_faces = Ionut_found_faces + 1;
    bias = int64(face_Location(3) * 10 / 100);
    %bias = 0;
    face = imagine(face_Location(2) - bias:face_Location(2)+ ...
        face_Location(4)+bias,face_Location(1) - bias: ...
        face_Location(1)+face_Location(3) + bias);
    %imagine = insertShape(imagine, 'Rectangle', face_Location);
    figure;
     
    face = imresize(face,[64 64]);
    imshow(face);
    
    X(1:64*64,2*Ionut_found_faces-1) = reshape(face',1,[])';
    Y(2*Ionut_found_faces) = 1;
end

%% VERIFICARE
if(N_dum > Dumi_found_faces || N_ionut > Ionut_found_faces)
    disp(['In unele poze nu au fost gasite fete sau au fost gasite mai mult' ...
        ' de una, deci nu vor fi luate in considerare']);
else
    disp('Totul este in regula');
end
close all;
n = Ionut_found_faces + Dumi_found_faces;

%% FIN
 clear bias Dumi_found_faces Ionut_found_faces face_Location faceDetector...
       face width N_ionut N_dum nume imagine i
save('poze.mat','X','Y','n');