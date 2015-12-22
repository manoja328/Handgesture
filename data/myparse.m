l=1;
for i=1:7
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
%imshow(reshape(b,30,30));
%fprintf('Program paused. Press enter to continue.\n');
%pause;
end


save('mydata.mat','X','y');