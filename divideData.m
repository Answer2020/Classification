
function [trainDataIndices,testDataIndices]=divideData(percen,count1,count2,count3,count4,count5,count6,count7,count8,count9,c1,c2,c3,c4,c5,c6,c7,c8,c9)

%This function randomly divides the data in each class into two groups of 
%training and testing based on the provided percentage. 
    
valRatio=0;
p = 1:count1;

if count1<=100
    newPercen=0.5;
   [P1trainInd,P1testInd,valInd] = dividerand(p,newPercen,1-newPercen,valRatio); 
else
[P1trainInd,P1testInd,valInd] = dividerand(p,percen,1-percen,valRatio);
end

p=1:count2;
p=p+c1;
if count2<=100
    newPercen=0.5;
    [P2trainInd,P2testInd,valInd] = dividerand(p,newPercen,1-newPercen,valRatio); 
else
[P2trainInd,P2testInd,valInd] = dividerand(p,percen,1-percen,valRatio);
end

p=1:count3;
p=p+c2;
if count3<=100
    newPercen=0.5;
    [P3trainInd,P3testInd,valInd] = dividerand(p,newPercen,1-newPercen,valRatio); 
else
[P3trainInd,P3testInd,valInd] = dividerand(p,percen,1-percen,valRatio);
end

p=1:count4;
p=p+c3;
if count4<=100
    newPercen=0.5;
    [P4trainInd,P4testInd,valInd] = dividerand(p,newPercen,1-newPercen,valRatio); 
else
[P4trainInd,P4testInd,valInd] = dividerand(p,percen,1-percen,valRatio);
end


p=1:count5;
p=p+c4;
if count5<=100
    newPercen=0.5;
    [P5trainInd,P5testInd,valInd] = dividerand(p,newPercen,1-newPercen,valRatio); 
else
[P5trainInd,P5testInd,valInd] = dividerand(p,percen,1-percen,valRatio);
end

p=1:count6;
p=p+c5;
if count6<=100
    newPercen=0.5;
    [P6trainInd,P6testInd,valInd] = dividerand(p,newPercen,1-newPercen,valRatio); 
else
[P6trainInd,P6testInd,valInd] = dividerand(p,percen,1-percen,valRatio);
end

p=1:count7;
p=p+c6;
if count7<=100
    newPercen=0.5;
    [P7trainInd,P7testInd,valInd] = dividerand(p,newPercen,1-newPercen,valRatio); 
else
[P7trainInd,P7testInd,valInd] = dividerand(p,percen,1-percen,valRatio);
end


p=1:count8;
p=p+c7;
if count8<=100
    newPercen=0.5;
    [P8trainInd,P8testInd,valInd] = dividerand(p,newPercen,1-newPercen,valRatio); 
else
[P8trainInd,P8testInd,valInd] = dividerand(p,percen,1-percen,valRatio);
end

p=1:count9;
p=p+c8;
if count9<=100
    newPercen=0.5;
    [P9trainInd,P9testInd,valInd] = dividerand(p,newPercen,1-newPercen,valRatio); 
else
[P9trainInd,P9testInd,valInd] = dividerand(p,percen,1-percen,valRatio);
end



trainDataIndices=horzcat(P1trainInd,P2trainInd,P3trainInd,P4trainInd,P5trainInd,P6trainInd,P7trainInd,P8trainInd,P9trainInd);
testDataIndices=horzcat(P1testInd,P2testInd,P3testInd,P4testInd,P5testInd,P6testInd,P7testInd,P8testInd,P9testInd);






