load('myparams.mat');

a=imread('ges316.bmp');

a=double(a);
b=(a-mean(a(:)))/std(a(:));
b=b(:)';
imshow(reshape(b,size(a,1),size(a,2)));
pred=predict(Theta1,Theta2,b)
% fprintf('Program paused. Press enter to continue.\n');
% pause;
