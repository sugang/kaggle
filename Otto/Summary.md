# Summary for Otto competition

## Our Best Model
The best model we have is from GangSu: 100xg boost + 60 big deepnet

### Important tips to tune those model


## Approaches tried but failed
### CaretEnsemble
What I did for caretEnsemble is both tunning parameters and stack methods. It turns out to be so inefficient! I waited days, and the process don't finish. And I used 8 cpus for parallel computing. I think I need a protocol that let the inefficient methods fail fast, So I could try something else.


## Interesting tips for future modeling


## Things to try in later competition
Las Everyone talks about it, but I am not sure how it works.

## Lession Learned from Forum
### Feature engineering
The reason for feature engineering is that some relations are very hard to learn by a machine learning method, like (y = 1/x, y = max(X)). So, it is necessary to transform the variable to make the data

#### Boosting to distinguish colser probability values
In forum, they provide a way to do this: logit(-log((1 - p) / p))

### Ensemble
There are thousands of possible ways for doing ensemble. And, basically, you can use the prediction of a method as the input for your ensemble method. So, you ensemble method could be the same as your prediction method. A few observations

1. 2-level ensemble is mostly used.
2. In many winning solution, what they do is simply something like 2 * xgboost + 3 * randomForest. I don't know how they choose the parameter. They said they could just use the LB to choose them.
3. For linear combination or *averaging*, one ensemble method is Genetic Algorithm. The package they use is (DEAP)[https://github.com/deap/deap]. With GA, With the GA approach, you don't have to worry about the models going in. Just try to minimize the ensemble loss.
4.


