clc

% Load the training data into memory
[inTrain, tTrain] = faceTrain_dataset;

% % Display some of the training images
% clf
% for i = 1:20
% subplot(4,5,i);
% imshow(inTrain{i});
% end

rng('default')
hiddenSize1 = 1000;
autoenc1 = trainAutoencoder(inTrain,hiddenSize1,'MaxEpochs',400,'L2WeightRegularization',0.004,'SparsityRegularization',2,'SparsityProportion',0.15,'ScaleData', false);
view(autoenc1)
plotWeights(autoenc1);
feat1 = encode(autoenc1,inTrain);

hiddenSize2 = 500;
autoenc2 = trainAutoencoder(feat1,hiddenSize2,'MaxEpochs',400,'L2WeightRegularization',0.004,'SparsityRegularization',2,'SparsityProportion',0.15,'ScaleData', false);
view(autoenc2)
plotWeights(autoenc2);
feat2 = encode(autoenc2,feat1);


%----------------------------------------------------------------------------------------------------------

% Intraclass Calculation

Intraclass = zeros(1,600);         %((6x5)/2)x40 = 600
z = 1;
for k = 0:6:234
for i = 1:1:6
for j = i:1:6

if (i < j)
Intraclass(1,z) = mse(feat1(:,i+k),feat1(:,j+k));
z = z+1;
end

end
end
end

Intra = mean(Intraclass);

%----------------------------------------------------------------------------------------------------------

% Interclass Calculation
% 
Interclass = zeros(1,28080);     %(240x234)/2 = 28080
z = 1;
for CA = 0:6:228
for i = 1:1:6
for CB = 6:6:234
for j = 1:1:6

if (CA<CB)
Interclass(1,z) = mse(feat1(:,i+CA),feat1(:,j+CB));
z = z+1;
end

end
end
end
end

Inter = mean(Interclass);

%----------------------------------------------------------------------------------------------------------

%Finding Representaion Ratio

Ratio = Inter/Intra;

%----------------------------------------------------------------------------------------------------------

%Plotting the ROC Curve

ezroc3(-1*[Intraclass,Interclass], [ones(1,600),zeros(1,28080)],2,'Hidden size1 = 1000, Sparsity reg = 2, Sparsity prop = 0.15 L2 weight reg = 0.004',1);

%----------------------------------------------------------------------------------------------------------

% Load the test images
[inTest,tTest] = faceTest_dataset;

%----------------------------------------------------------------------------------------------------------

y = zeros(40,160);

for sub = 1:40  
    
    %Training the final softmax layer
    softnet = trainSoftmaxLayer(feat2,tTrain(1,:),'MaxEpochs',400);
    deepnet = stack(autoenc1,autoenc2,softnet);
    view(deepnet);
    
    %----------------------------------------------------------------------------------------------------------

    %To reshape the test images into a matrix

    % Get the number of pixels in each image
    imageWidth = 92;
    imageHeight = 112;
    inputSize1 = imageWidth*imageHeight;

    % Turn the test images into vectors and put them in a matrix
  
    xTest = zeros(inputSize1,numel(inTest));
    for i = 1:numel(inTest)
        xTest(:,i) = inTest{i}(:);
    end
     
    %----------------------------------------------------------------------------------------------------------
    
    %To visualize the results with a confusion matrix
    
    y(sub,:) = deepnet(xTest);
    
    %----------------------------------------------------------------------------------------------------------
    
    %Fine tuning the deep neural network
    
    % Turn the training images into vectors and put them in a matrix
    xTrain = zeros(inputSize1,numel(inTrain));
    for i = 1:numel(inTrain)
        xTrain(:,i) = inTrain{i}(:);
    end
    
    % Perform fine tuning
    deepnet = train(deepnet,xTrain,tTrain(sub,:))
         
    %----------------------------------------------------------------------------------------------------------
   
    %view the results again using a confusion matrix
    y(sub,:) = deepnet(xTest);
      
end
 
ezroc3(y,tTest,2,'ROC after fine tuning',1);
plotconfusion(tTest,y)
 
