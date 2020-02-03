function [output,transMat] = LDA(X,Y,n2)
%This function applies LDA method on the input data matrix X with labels Y
%n2:  Number of eigenvectors to keep
[d,M]=size(X); % M is the number of samples in the dataset and d is their dimensions
output=0;
transMat=0;

classIndex=1;

classes=unique(Y');


NumberOfClasses = size(classes,2)-1;
if(nargin==2)
    n2=NumberOfClasses-1;
end


classMean = cell(1,NumberOfClasses);
Covariance= cell(1,NumberOfClasses);
classFr=zeros(1,NumberOfClasses);

totalMean = mean(X,2);
SB = zeros(d,d);
SW = zeros(d,d);

for k=unique(Y')
    if (k==0)
        disp('k==0');
    else
        Xc=X(:,Y==k);
        classFr(classIndex)=size(Xc,2);
        Covariance(classIndex) = {cov(Xc')};
        classMean(classIndex) = {mean(Xc,2)};
        classIndex=classIndex+1;
    end
end


for j = 1:NumberOfClasses
    SW = SW+Covariance{j};
    SB = SB + classFr(j).*(classMean{j}-totalMean)*(classMean{j}-totalMean)';
end


SwInverse = inv(SW);
targetMatrix = SwInverse * SB;

[V,D] = eig(targetMatrix);
eigval=diag(D);

[sortedEigval,index_sorted]=sort(eigval,'descend');
transMat=V(:,index_sorted(1:n2));
output = transMat'*X;

end