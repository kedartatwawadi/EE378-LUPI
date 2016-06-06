%Ionosphere Test

clear; clc;
addpath('./utils');
% load data
Data = importdata('ring.dat');
Data = (Data.data);
Data(:,end) = Data(:,end)*2 - 1;
shuffle = randperm(size(Data,1));
Data = Data(shuffle,:);

Number_of_attributes = size(Data,2) - 1;
Normal_attributes = 10;
PF_attributes = Number_of_attributes - Normal_attributes;

Fold = 5;

siz_max = 150;
siz = 10:10:siz_max;
error_plus = zeros(length(siz),1);
error = error_plus;
iter_max = 20;

for iteration = 1:iter_max
    Data(1:siz_max,:) = Data(randperm(siz_max),:);
    for i = 1:length(siz)
        % preprocessing data 
        l = siz(i);
        train_features      = Data(1:l, 1:Normal_attributes)';
        test_features       = Data(siz_max+1:end, 1:Normal_attributes)';
        train_PFfeatures    = Data(1:l, Normal_attributes+1:Normal_attributes+...
            PF_attributes)';

        train_labels = Data(1:l,end);
        test_labels = Data(siz_max+1:end,end);


        % calculate kernels
        kparam = struct();
        kparam.kernel_type = 'gaussian';
        [K, train_kparam] = getKernel(train_features, kparam);
        testK       = getKernel(test_features, train_features, train_kparam);

        kparam = struct();
        kparam.kernel_type = 'gaussian';
        tK = getKernel(train_PFfeatures, kparam);

        % ================ train SVM+ ====================
        % parameters could be obtained via validation
        [ C, gamma ] = SVM_plus_cv( train_labels, train_features, train_PFfeatures, Fold );
        svmplus_param.svm_C = C; 
        svmplus_param.gamma = gamma;
        tic;
        model = svm_plus_train(train_labels, K, tK, svmplus_param);
        tt = toc;
        decs    = testK(:, model.SVs) * model.sv_coef - model.rho;
        error_plus(i)     = error_plus(i) + sum((2*(decs>0)-1) ~= test_labels)/length(test_labels);

        % ================ SVM ====================
        C = SVM_cv2( train_labels, train_features, Fold );
        svmplus_param.svm_C = C; 
        tic;
        model = svm_train(train_labels, K, svmplus_param);
        tt = toc;
        decs    = testK(:, model.SVs) * model.sv_coef - model.rho;
        error(i)     = error(i) + sum((2*(decs>0)-1) ~= test_labels)/length(test_labels);


    end
end

error_plus = error_plus / iter_max;
error = error / iter_max;
plot(siz, error_plus, '--rs', siz, error, ':b*')
legend('SVM+','SVM')

