load('data/mathworks/test/testDataNoCar.mat')

% load('data/mathworks/trainDataNoCar.mat') % Too big!

testDataNoCar_1 = testDataNoCar(:,:,:,1:1000);
testDataNoCar_2 = testDataNoCar(:,:,:,1001:2000);
testDataNoCar_3 = testDataNoCar(:,:,:,2001:3000);
testDataNoCar_4 = testDataNoCar(:,:,:,3001:4000);
testDataNoCar_5 = testDataNoCar(:,:,:,4001:5000);

save('data/mathworks/test/testDataNoCar_1.mat', 'testDataNoCar_1')
save('data/mathworks/test/testDataNoCar_2.mat', 'testDataNoCar_2')
save('data/mathworks/test/testDataNoCar_3.mat', 'testDataNoCar_3')
save('data/mathworks/test/testDataNoCar_4.mat', 'testDataNoCar_4')
save('data/mathworks/test/testDataNoCar_5.mat', 'testDataNoCar_5')

save('data/mathworks/test/testLabelNoCar.mat', 'testLabelNoCar')