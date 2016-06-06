function models = svm_train(labels, K, param)
% models = svm_plus_train(labels, K, tK, param)
%   Training SVM+.
% Inputs:
%   - labels: n-by-1 vecotr, source labels (support multi-class)
%   - K: n-by-n kernel matrix, feature space
%   - tK: n-by-n kernel matrix, privileged information space
%   - param: 
%       - svm_C: C in SVM
%       - gamma: gamma in SVM+
% Outputs:
%   - models: m-by-1 cell, each is an SVM model, ordered in ascending order
%   of labels.
%
% LI Wen, on July 31, 2013
% -----------------------------------------------
% update the calculation of rho, by Li Wen on Aug 2, 2013
%

% -----------------------------------------------
% parameters
svm_C   = param.svm_C;

cates   = unique(labels);
n_class = length(cates);
n       = length(labels);

DISP    = 1; % if display the messages or not
if(DISP)
    fprintf('Training SVM+...\n'); 
    fprintf('Number of Categoris: %d\n', n_class);
end

if(n_class == 2)
    n_class = 1;
    cates   = [1 -1]; % put postive before, so Label(1) == 1 always hold
end

models  = cell(n_class, 1);
fprintf('In total %d classes:\n', n_class);
for ci = 1:n_class
    y       = (labels == cates(ci))*2 - 1;
    fprintf('Class#%d training...\t', ci);
    % -----------------------------------------------
    % solving in SVM+ form
    tt = tic;
    H       = [K.*(y*y')];
    f       = -ones(n,1);
    A1      = [];
    b1      = [];
    A2      = y'; %a'y = 0; 1'(a + z) = 1'c;
    b2      = 0;
    lb      = zeros(n, 1);
    ub      = [];    
    opt.Display = 'off';
    x       = quadprog(H, f, A1, b1, A2, b2, lb, ub, [], opt);
    
    fprintf('Train Time = %f\n', toc(tt));

    alpha   = x;
    % ------------------ rho --------------------
    % tmp_idx =(alpha>1e-10)&(alpha<svm_C);
    % rho     = mean((K(tmp_idx, :)*(alpha.*y) - y(tmp_idx)));
    %
    % Note: I defaultly use 1e-10 for judging if dual variables great than
    % zeros, but someties, it might be too small/large. Config it for your
    % applications. 
    %     
    
    dec     = K*(alpha.*y);
    dec_2   = -dec + y;
    tmp_idx =(alpha>1e-10);
    if(all(~tmp_idx))    % no alpha > 0        
        lb  = max(dec_2(y>0));
        ub  = min(dec_2(y<0));
        b       = (lb + ub)/2;
    else
        b = mean(dec_2(tmp_idx));
    end
    if(isnan(b))
        error('b is NaN.\n');
    end
    rho = - b;
    
    index   = find(alpha > 1e-10);
    sv_coef = (alpha.*y); 
    
    % save models, similar as libsvm models
    models{ci}.sv_coef  = sv_coef(index);
    models{ci}.SVs      = index;    
    models{ci}.Label(1) = 1;
    models{ci}.rho      = rho;

    % -------------------------------------------    
    % others, for debugging
    models{ci}.x        = x;
    models{ci}.y        = labels;
    models{ci}.param    = param;    
    models{ci}.alpha = alpha;
    models{ci}.b = b;

end
if(n_class == 1)
    models = models{1};
end
