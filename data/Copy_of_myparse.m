l=1;
for i=1:5
mylist = ls ([int2str(i) '\*.bmp']);
for k=1:size(mylist,1)
        I=imread([int2str(i) '\' mylist(k,:)]);
        Xa(l,:)=I(:);        
        y(l,1)=i;
        l=l+1;
end
end


for i=1:size(Xa,1)
a=double(Xa(i,:));
b=(a-mean(a))/std(a);
X(i,:)=b(:);
%imshow(reshape(b,32,24));
%fprintf('Program paused. Press enter to continue.\n');
%pause;
end


% in terminal have only 2 data X and Y and do
% save ('mydata.mat')
% clear
% load ('mydata.mat') 
% same as in neural coursera to check
% imshow(reshape(X(1,:),32,24)) 