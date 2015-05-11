# xgboost
# the code is adapted from 
# https://www.kaggle.com/users/32300/tianqi-chen/otto-group-product-classification-challenge/understanding-xgboost-model-on-otto-data

# remove columns
require(xgboost)
require(methods)
require(data.table)
require(magrittr)
train <- fread('./Data/train.csv', header = T, stringsAsFactors = F)
test <- fread('./Data/test.csv', header=TRUE, stringsAsFactors = F)
train[, id := NULL]
test[, id := NULL]

# Convert from classes to numbers
nameLastCol <- names(train)[length(names(train))]
y <- train[, nameLastCol, with = F][[1]] %>% gsub('Class_','',.) %>% {as.integer(.) -1}
train[, nameLastCol:=NULL, with = F]

trainMatrix <- train[,lapply(.SD,as.numeric)] %>% as.matrix
testMatrix <- test[,lapply(.SD,as.numeric)] %>% as.matrix



numberOfClasses <- max(y) + 1

param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = numberOfClasses)

cv.nround <- 500
cv.nfold <- 3

bst.cv = xgb.cv(param=param, data = trainMatrix, label = y, 
                nfold = cv.nfold, nrounds = cv.nround)

# train the real model
nround = 139
bst = xgboost(param=param, data = trainMatrix, label = y, nrounds=nround)

pre_result <- predict(bst, testMatrix)
pre_result_mod <- matrix(pre_result,ncol = 9,byrow = T)
submission <- data.frame(id=test_data$id, Class_1=NA, Class_2=NA, Class_3=NA, Class_4=NA, Class_5=NA, Class_6=NA, Class_7=NA, Class_8=NA, Class_9=NA)
submission[,2:10] <- pre_result_mod

write.csv(submission, quote = F, row.names = F, file = "./predict_result.csv")


PredictWithModel(bst, test)
