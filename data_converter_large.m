% Large File loader

exampleObject = matfile('data/mathworks/train/trainDataNoCar.mat');
varlist = who(exampleObject)
 
[d0,d1,d2,d3] = size(exampleObject,'trainDataNoCar');
lastColB = exampleObject.trainDataNoCar(:,:,:,d3);

labels = exampleObject.trainLabelNoCar;
trainLabelNoCartString = char(labels);
save('data/mathworks/train/trainLabelNoCar.mat','trainLabelNoCartString')

for n = 1:20
    lastColB = zeros(400,144,1000);
    for m = 1:1000
        lastColB(:,:,m) = exampleObject.trainDataNoCar(:,:,:,(1000*(n-1)+(m)));
    end
    save(strcat('data/mathworks/train/trainDataNoCar_',int2str(n),'.mat'), 'lastColB')
    disp(n)
end