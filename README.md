# LUPI
Course Project for EE378A.

## Intuition
The references 3,4 highlight the fact that one way of thinking about priviledged insformation is separating easy and hard training examples, and peanlizing the student less harshly for an incorrect output on the hard samples and vice-versa. The physical intuition behind this is that: say a teacher is teaching a class 2 student 2+2, 4+5, etc. and suddenly she gives a complicated integral to solve. Of course the student is going to find it very difficult to solve the integral as it is a hard problem for him. However, he should not be penalized significantly and his learning procedure should not be changed because of incorrect answer on the hard training example.

The main contribution in some sense of ref 3 is to obtain these hard/easy labels over the problems and incorporate that in the learning process. The Ref 4 (which I have not read yet) provides a theoretical backing to this idea of assigning weights to training examples and shows that weighted SVM will always perform better than SVM+ (which is exciting from the implementation point of view).
Taking the idea of weighted samples further, I think we can generalize this to even neural networks: 

a) Obtain weights using some way for training examples
b) Use the weights for scaling learning rate in the stochastic descent methods (simple SGD?)

I think this could work.


There is one more intuition for LUPI, which I will confirm later.

## References
1. SVMPy
2. Vapnik's papers
3. [Learning to Transfer Privileged Information](http://ilovevisiondata.wix.com/viktoriia#!projects/cm8a)
4. [Learning Using Privileged Information: SVM+ and Weighted SVM](https://arxiv.org/pdf/1306.3161.pdf)

## Implementations
1. [SVM+](https://github.com/transmatrix-github/svmplus_matlab) : svm_+ implementation in matlab
2. [Margin Transfer](http://ilovevisiondata.wix.com/viktoriia#!projects/cm8a) : A Variant of SVM+ algorithm which can be solved using standard SVM solvers

## Datasets
1. [Adult income dataset](https://archive.ics.uci.edu/ml/datasets/Adult): predicting income level
2. [Age of Wine](http://archive.ics.uci.edu/ml/datasets/Wine): determining age of wine from composition
   ( find first the median and convert to regression problem )
3. [Iris dataset](http://archive.ics.uci.edu/ml/datasets/Iris): Standard Iris dataset
4. [wine quality](http://archive.ics.uci.edu/ml/datasets/Wine+Quality): wine quality

