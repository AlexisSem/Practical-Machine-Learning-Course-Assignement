# LOAD LIBRARIES & DATA
print("Loading libraries and data...")
library(caret)
library(randomForest)
library(gbm)

pml_training <- read.csv('pml-training.csv')
pml_testing <- read.csv('pml-testing.csv')

# TRAIN AND TEST SETS
print("Creat training and testing set...")
inTrain <- createDataPartition(y=pml_training$classe, p=0.7, list=FALSE)
trainSet <- pml_training[inTrain, ]
testSet <- pml_training[-inTrain,]

# FEATURES SELECTION

# Delete variables that are nearly always NA
print("Deleting all NA variables...")
isNearlyNA <- sapply(trainSet, function(x) mean(is.na(x)) > 0.95)
trainSet <- trainSet[, isNearlyNA==FALSE]
testSet <- testSet[, isNearlyNA==FALSE]

# Delete variables with near zero variance
print("Deleting near zero variance variables...")
nzv <- nearZeroVar(trainSet)
trainSet <- trainSet[, -nzv]
testSet <- testSet[, -nzv]

# Delete user ids variables
print("Deleting user ids variables...")
trainSet <- trainSet[, -(1:5)]
testSet <- testSet[, -(1:5)]

#Lots of variables -> we will try to reduce them by using PCA
print("Executing PCA...")
preProc <- preProcess(trainSet[, -54], method="pca", thresh = 0.95)

# With only 25 variables we capture 95% of the variance. 
#It may have an interest, we will try two sets of models : one with pca, one without

# MODEL SELECTION
# We will test 3 models : RF, GBM and RF AND GBM Combined.
# We will each time use the PCA Variant and select the model that does best on the the testset.

# 1) Random Forest
print("Executing Random Forest Algorithm...")
set.seed(1234)
controlRF <- trainControl(method = "cv", number = 3, verboseIter = FALSE)
mod_rf <- train(classe~., method = "rf", data = trainSet, trControl = controlRF)
pred_rf <- predict(mod_rf, testSet)
accuracy_rf <- confusionMatrix(pred_rf, testSet$classe)$overall[1]
print(paste("Accuracy of Random Forest : ", accuracy_rf))

# 1bis) Random Forest with PCA
# print("Executing Random Forest Algorithm with PCA...")
# set.seed(1234)
# controlRF <- trainControl(method = "cv", number = 3, verboseIter = FALSE)
# mod_rf_pca <- train(classe~., method = "rf", data = trainSet, preProcess = "pca", trControl = controlRF)
# pred_rf_pca <- predict(mod_rf_pca, testSet)
# accuracy_rf_pca <- confusionMatrix(pred_rf_pca, testSet$classe)$overall[1]
print(paste("Accuracy of Random Forest with PCA : ", accuracy_rf_pca))

# 2) GBM Algorithm
# print("Executing GBM Algorithm...")
# set.seed(1234)
# controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
# mod_gbm <- train(classe~., method = "gbm", data = trainSet, trControl = controlGBM, verbose = FALSE)
# pred_gbm <- predict(mod_gbm, testSet)
# accuracy_gbm <- confusionMatrix(pred_gbm, testSet$classe)$overall[1]
print(paste("Accuracy of GBM : ", accuracy_gbm))

# 2bis) GBM Algorithm with PCA
# print("Executing GBM Algorithm with PCA...")
# set.seed(1234)
# mod_gbm_pca <- train(classe~., method="gbm", data=trainSet, preProcess="pca", trControl=controlGBM, verbose=FALSE)
# pred_gbm_pca <- predict(mod_gbm_pca, testSet)
# accuracy_gbm_pca <- confusionMatrix(pred_gbm_pca, testSet$classe)$overall[1]
print(paste("Accuracy of GBM with PCA: ", accuracy_gbm_pca))

# We select the Random Forest Model (the 1st one)

# APPLICATION TO THE TEST DATA

prediction <- predict(mod_rf, newdata = pml_testing)
print(prediction)


