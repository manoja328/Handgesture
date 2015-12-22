load('myparams.mat');

mylist = ls ('testim\*.bmp');
for k=1:size(mylist,1)
a=imread(['testim\' mylist(k,:)]);
a=double(a);
b=(a-mean(a(:)))/std(a(:));
b=b(:)';
imshow(reshape(b,size(a,1),size(a,2)));
pred=predict(Theta1,Theta2,b)
fprintf('Program paused. Press enter to continue.\n');
pause;
end



% load('myparams.mat');
% a=imread('ges81.bmp');
% a=double(a);
% b=(a-mean(a(:)))/std(a(:));
% b=b(:)';
% imshow(reshape(b,size(a,1),size(a,2)));
% pred=predict(Theta1,Theta2,b)

