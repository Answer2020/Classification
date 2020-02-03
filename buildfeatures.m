function [trainData,tTrain,ijTrain,trainLabels,TestData, tTest,ijTest,testlabels,r,c]=buildfeatures(neighbourSize,numComp,LDAcomp,Data,labels,ijindex,trainInd,testInd)
%This function builds the PCDA features of the input spectra

load 'PaviaU';
load 'PaviaU_gt';

numClasses=9;

x=floor(neighbourSize/2);
[r,c,lam]=size(paviaU);


%%
%PCDA
paviaUReshaped=reshape(paviaU,[r*c,lam]);
[coeff,score,latent] = pca(paviaUReshaped,'NumComponents',lam);

m1=numComp;
A1=coeff(:,1:m1);
y1=score(:,1:m1);
y1=y1';

A2=coeff(:,m1+1:end);
y2=paviaUReshaped*A2;


Y=reshape(paviaU_gt,[r*c,1]);
Y=Y';
m2=LDAcomp;
y2=y2';
[scoreLDA,W] = LDA(y2,Y',m2);
y3=scoreLDA;

yt = vertcat(y1,y3);

M=m1+m2;

yt=yt';
paviaPCA=reshape(yt,[r,c,size(yt,2)]);

for q=1:M
    pddedPaviaPCA(:,:,q)=padarray(paviaPCA(:,:,q),[x,x]) ;
end


%%
%BUILDING TRAIN FEATURES
for k1=1:size(trainInd,1)
    spatial=[];
    spectral=Data(trainInd(k1,1),:);
    for ii=ijindex(trainInd(k1,1))-x:ijindex(trainInd(k1,1))+x
        for jj=ijindex(trainInd(k1,1),2)-x:ijindex(trainInd(k1,1),2)+x
            spatial=[spatial reshape(pddedPaviaPCA(ii+x,jj+x,:),[1,M])];
        end
    end
    
    
    trainData(k1,:)=[spectral spatial];
    trainLabels(k1,:)=labels(trainInd(k1,1),1);
    ijTrain(k1,1)=ijindex(trainInd(k1,1),1);
    ijTrain(k1,2)=ijindex(trainInd(k1,1),2);
end


trainData=trainData';
tTrain=zeros(numClasses,size(trainData,2));
for k=1:size(trainData,2)
    tTrain(trainLabels(k,1),k)=1;
end

%%
%BUILDING TEST FEATURES

for k3=1:size(testInd,1)
    spatial=[];
    spectral=Data(testInd(k3,1),:);
    for ii=ijindex(testInd(k3,1))-x:ijindex(testInd(k3,1))+x
        for jj=ijindex(testInd(k3,1),2)-x:ijindex(testInd(k3,1),2)+x
            spatial=[spatial reshape(pddedPaviaPCA(ii+x,jj+x,:),[1,M])];
        end
    end
  
    TestData(k3,:)=[spectral spatial];
    testlabels(k3,:)=labels(testInd(k3,1),1);
    ijTest(k3,1)=ijindex(testInd(k3,1),1);
    ijTest(k3,2)=ijindex(testInd(k3,1),2);
    
end

TestData=TestData';
tTest=zeros(numClasses,size(TestData,2));
tTest=zeros(numClasses,size(TestData,2));
for kkk=1:size(TestData,2)
    tTest(testlabels(kkk,1),kkk)=1;
end

%%