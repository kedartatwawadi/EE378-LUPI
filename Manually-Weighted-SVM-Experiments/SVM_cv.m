function  C = SVM_cv( labels, features, Fold )

C_temp = [100; 10; 1; .1; .01];
error = zeros(5,1);
n = size(features,1);
m = floor(n/Fold);
option = optimset('MaxIter', 1000000);

for i = 1:5
    err = 0;
    for fold = 1:Fold
        boxconstraint = C_temp(i) * ones(n,1);
        X_train = features;
        Y_train = labels;

        X_test = X_train((fold-1)*m+1:fold*m, :);
        X_train( (fold-1)*m+1:fold*m, :) = [];
        Y_test = Y_train((fold-1)*m+1:fold*m );
        Y_train((fold-1)*m+1:fold*m ) = [];
        boxconstraint((fold-1)*m+1:fold*m ) = [];

        SVMStruct = svmtrain(X_train,Y_train,'boxconstraint', ...
            boxconstraint,'options',option);
        Group = svmclassify(SVMStruct,X_test);
        err = err + sum(Group ~= Y_test)/length(Y_test);
    end

    error(i) = err;
end

[M, I] = min(error);

C = C_temp(I);

end