
clc;
clear all;
Data = importdata('Wine_Age_Data.txt');
Normal_attributes = 7;
PF_attributes = 13 - Normal_attributes;
siz_max = 60;
siz = 10:5:siz_max;
error = zeros(length(siz),1);
errorr = error;
iter_max = 10;

for iter = 1:iter_max
    Data(1:siz_max,:) = Data(randperm(siz_max),:);

    X_train      = Data(1:siz_max, 1:Normal_attributes);
    X_test       = Data(siz_max+1:end, 1:Normal_attributes);

    Y = Data(:,end);
    Y_train = Y(1:siz_max,end);
    Y_train(Y(1:siz_max,end)<median(Y)) = 0;
    Y_train(Y(1:siz_max,end)>=median(Y)) = 1;
    Y_test = Y(siz_max+1:end,end);
    Y_test(Y(siz_max+1:end,end)<median(Y)) = 0;
    Y_test(Y(siz_max+1:end,end)>=median(Y)) = 1;

    diff = ( Y > median(Y) + sqrt(var(Y))/4 );
    diff = diff + ( Y < median(Y) - sqrt(var(Y))/4 );
    diff = 1 - diff;
    diff = logical(diff);
    Fold = 5;

    option = optimset('MaxIter', 1000000);


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
        error(i) = error(i) + sum(Group ~= Y_test)/length(Y_test);
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

error = error / iter_max;
errorr = errorr / iter_max;
plot(siz,error,'--rs', siz,errorr, ':b*')
legend('Manually Weighted SVM','SVM')