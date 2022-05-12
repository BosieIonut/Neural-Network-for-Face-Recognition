%loading the video
the_Image      = rgb2gray(imread('f_rec.jpg'));
[width, height] = size(the_Image);

if width>320
the_Image = imresize(the_Image,[320 NaN]);
end

% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();

%finding the bounding box that encloses the face on video frame
face_Location = step(faceDetector, the_Image);

% Draw the returned bounding box around the detected face.
the_Image = rgb2gray(insertShape(the_Image, 'Rectangle', face_Location));
figure; 
imshow(the_Image); 
title('Detected face');
face = the_Image(face_Location(2):face_Location(2)+face_Location(4),face_Location(1):face_Location(1)+face_Location(3));
figure
imshow(face);