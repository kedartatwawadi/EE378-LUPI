function [ C, gamma ] = SVM_plus_cv( labels, features, feature_PF, Fold )

C_temp = [ 10; 1; .1; .01];
gamma_temp = [ 10; 1; .1; .01];
error = zeros(4,4);
n = size(features,2);
m = floor(n/Fold);
param = struct();
param.kernel_type = 'gaussian';

for i = 1:4
    for j = 1:4
        param.svm_C = C_temp(i);
        param.gamma = gamma_temp(j);
        err = 0;
        for fold = 1:Fold
            
            X_train = features;
            X_train_PF = feature_PF;
            Y_train = labels;
            
            X_test = X_train(:,(fold-1)*m+1:fold*m);
            X_train(:, (fold-1)*m+1:fold*m) = [];
            X_train_PF(:, (fold-1)*m+1:fold*m) = [];
            Y_test = Y_train((fold-1)*m+1:fold*m );
            Y_train((fold-1)*m+1:fold*m ) = [];
            
            kparam = struct();
            kparam.kernel_type = 'gaussian';
            [K, train_kparam] = getKernel(X_train, kparam);
            testK       = getKernel(X_test, X_train, kparam);
            tK = getKernel(X_train_PF, kparam);
            
            Model = svm_plus_train(Y_train, K, tK, param);
            
            decs    = testK(:, Model.SVs) * Model.sv_coef - Model.rho;
            acc     = sum((2*(decs>0)-1) == Y_test)/length(Y_test);
            err = err + 1 - acc;
        end
        
        error(i,j) = err;
    end
end

[M, I] = min(error);
[MM, II] = min(M);

C = C_temp(I(II));
gamma = gamma_temp(II);
end

