
clc;
clear all;
load('Abalone')
diff = Y == 9;
Fold = 5;

option = optimset('MaxIter', 1000000);
siz_max = 200;
siz = 10:10:150;
error = zeros(length(siz),1);
errorr = zeros(length(siz),1);
iter_max = 10;

for iter = 1:iter_max
    shuffle = randperm(length(Y_train));
    diff = diff(shuffle);
    X_train = X_train(shuffle,:);
    Y_train = Y_train(shuffle,1);
    for i = 1:length(siz)
        %SVMStruct = fitcsvm(X,Y,'KernelFunction','rbf','Standardize',true,...
           % 'Boxconstraint', bc(1:l));
        l = siz(i);  
        [ C_easy, C_hard ] = Weighted_SVM_cv(Y_train(1:l), X_train(1:l,:), diff(1:l), Fold );
        bc = C_easy*ones(l,1);
        bc(diff(1:l) == 1) = C_hard;
        SVMStruct = svmtrain(X_train(1:l,:),Y_train(1:l),'boxconstraint', bc(1:l)...
            ,'options',option);
        Group = svmclassify(SVMStruct,X_test);
        error(i) =  error(i) + sum(Group ~= Y_test)/length(Y_test);
    end


    for i = 1:length(siz)
        l = siz(i);
        [ C ] = SVM_cv( Y_train(1:l), X_train(1:l,:), Fold );
        bc = C*ones(l,1);
        SVMStruct = svmtrain(X_train(1:l,:),Y_train(1:l),'boxconstraint', bc(1:l),'options',option);
        %SVMStruct = fitcsvm(X,Y,'KernelFunction','rbf','Standardize',true,...
         %   'Boxconstraint', bc(1:l));
        Group = svmclassify(SVMStruct,X_test);
        errorr(i) = errorr(i) + sum(Group ~= Y_test)/length(Y_test);
    end
end
error = error/iter_max;
errorr = errorr/iter_max;
plot(siz,error,'--rs', siz,errorr, ':b*')
legend('Manually Weighted SVM','SVM')