function [ C ] = SVM_cv2( labels, features, Fold )

C_temp = [ 10; 1; .1; .01];
error = zeros(4,1);
n = size(features,2);
m = floor(n/Fold);
param = struct();
param.kernel_type = 'gaussian';

for i = 1:4
    param.svm_C = C_temp(i);
    err = 0;
    for fold = 1:Fold

        X_train = features;
        Y_train = labels;

        X_test = X_train(:,(fold-1)*m+1:fold*m);
        X_train(:, (fold-1)*m+1:fold*m) = [];
        Y_test = Y_train((fold-1)*m+1:fold*m );
        Y_train((fold-1)*m+1:fold*m ) = [];

        kparam = struct();
        kparam.kernel_type = 'gaussian';
        [K, train_kparam] = getKernel(X_train, kparam);
        testK       = getKernel(X_test, X_train, kparam);

        Model = svm_train(Y_train, K, param);

        decs    = testK(:, Model.SVs) * Model.sv_coef - Model.rho;
        acc     = sum((2*(decs>0)-1) == Y_test)/length(Y_test);
        err = err + 1 - acc;
    end

    error(i,1) = err;
end

[M, I] = min(error);

C = C_temp(I);
end

