function [kappaValue]= ComputeKappa(C, ObservedAccuracy,N);

%C:Confusion matrix
%N: Number of test samples
KappaVector=zeros(1,size(C,1));

for i=1:size(C,1)
    v1=sum(C(i,:));
    v2=sum(C(:,i));
    KappaVector(i)=(v1*v2)/N;
    
end


expectedAccuracy=(sum(KappaVector))/N;
ObservedAccuracy=ObservedAccuracy/100;
kappaValue=(ObservedAccuracy-expectedAccuracy)/(1-expectedAccuracy)
