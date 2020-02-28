
function [Data,labels,ijindex,count1,count2,count3,count4,count5,count6,count7,count8,count9,c1,c2,c3,c4,c5,c6,c7,c8,c9]=buildLabeledData()
%This function organizes the total labeled pixels in the input image and their
%corresponding class labels in matrices Data and labels, respectively.
%counti:Stores number of samples available for class i.

load 'PaviaU';
load 'PaviaU_gt';
[r,c,lam]=size(paviaU);
numClasses=9

%%
counter=0;
for i=1:r
    for j=1:c
       if paviaU_gt(i,j)~=0
          counter=counter+1; 
    end
end
end
Data=zeros(counter,lam);
labels=zeros(counter,1);
ijindex=zeros(counter,2);

count1=0;
for i=1:r
    for j=1:c
        if  paviaU_gt(i,j)==1
            count1=count1+1;
         Data( count1,:)=paviaU(i,j,:);
         labels(1:count1,1)=1;
         ijindex(count1,1)=i;
         ijindex(count1,2)=j;
        end
    end
end
c1=count1;

count2=0;
for i=1:r
    for j=1:c
        if  paviaU_gt(i,j)==2
            count2=count2+1;
         Data( c1+count2,:)=paviaU(i,j,:);
         labels(c1+1:c1+count2,1)=2;
          ijindex(c1+count2,1)=i;
         ijindex(c1+count2,2)=j;
        end
    end
end
c2=count1+count2;


count3=0;
for i=1:r
    for j=1:c
        if  paviaU_gt(i,j)==3
            count3=count3+1;
         Data( c2+count3,:)=paviaU(i,j,:);
         labels(c2+1:c2+count3,1)=3;
           ijindex(c2+count3,1)=i;
         ijindex(c2+count3,2)=j;
        end
    end
end

c3=count1+count2+count3;


count4=0;
for i=1:r
    for j=1:c
        if  paviaU_gt(i,j)==4
            count4=count4+1;
         Data( c3+count4,:)=paviaU(i,j,:);
          labels(c3+1:c3+count4,1)=4;
           ijindex(c3+count4,1)=i;
         ijindex(c3+count4,2)=j;
        end
    end
end
c4=c3+count4;

count5=0;
for i=1:r
    for j=1:c
        if  paviaU_gt(i,j)==5
            count5=count5+1;
         Data( c4+count5,:)=paviaU(i,j,:);
         labels(c4+1:c4+count5,1)=5;
          ijindex(c4+count5,1)=i;
         ijindex(c4+count5,2)=j;
        end
    end
end
c5=c4+count5;


count6=0;
for i=1:r
    for j=1:c
        if  paviaU_gt(i,j)==6
            count6=count6+1;
         Data( c5+count6,:)=paviaU(i,j,:);
          labels(c5+1:c5+count6,1)=6;
          ijindex(c5+count6,1)=i;
         ijindex(c5+count6,2)=j;
        end
    end
end
c6=c5+count6;

count7=0;
for i=1:r
    for j=1:c
        if  paviaU_gt(i,j)==7
            count7=count7+1;
         Data( c6+count7,:)=paviaU(i,j,:);
         labels(c6+1:c6+count7,1)=7;
          ijindex(c6+count7,1)=i;
         ijindex(c6+count7,2)=j;
        end
    end
end
c7=c6+count7;

count8=0;
for i=1:r
    for j=1:c
        if  paviaU_gt(i,j)==8
            count8=count8+1;
         Data( c7+count8,:)=paviaU(i,j,:);
             labels(c7+1:c7+count8,1)=8;
               ijindex(c7+count8,1)=i;
         ijindex(c7+count8,2)=j;
        end
    end
end
c8=c7+count8;

count9=0;
for i=1:r
    for j=1:c
        if  paviaU_gt(i,j)==9
            count9=count9+1;
         Data( c8+count9,:)=paviaU(i,j,:);
         labels(c8+1:c8+count9,1)=9;
         ijindex(c8+count9,1)=i;
         ijindex(c8+count9,2)=j;
        end
    end
end 
c9=c8+count9;




