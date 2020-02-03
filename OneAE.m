function deepnet=OneAE(xTrainImages,hiddenUnitNumbers,pretrainingIterations,fineTuningIterations,tTrain1, tTrain)
%(xTrainImages,hiddenUnitNumbers,pretrainingEpochs,fineTuningEpochs,tTrain1, tTrain);

hiddenSize1 = hiddenUnitNumbers;
autoenc1 = trainAutoencoder(xTrainImages,hiddenSize1, ...
    'MaxEpochs',pretrainingIterations, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',2, ...
    'SparsityProportion',0.8, ...
    'ScaleData', true);

feat1 = encode(autoenc1,xTrainImages);

softnet = trainSoftmaxLayer(feat1,tTrain1,'MaxEpochs',fineTuningIterations);

%%
deepnet = stack(autoenc1,softnet);

deepnet = train(deepnet,xTrainImages,tTrain);