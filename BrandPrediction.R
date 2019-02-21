

####----------------------------------Import -----------------------------------------####

#call relevant libraries
library(readxl)
library(caret)
library(randomForest)
library(party)
library(corrplot)
library(hexbin)
library(RColorBrewer)
library(plot3D)
library(ggplot2)

#import dataset
survey_data <- read_excel('datasets/Survey_Key_and_Complete_Responses_excel.xlsx',
                          sheet = 2)

####---------------------------Getting familiar with dataset--------------------------####

#change datatypes of relevant attributes
survey_data$brand <- as.factor(survey_data$brand)
survey_data$zipcode <- as.factor(survey_data$zipcode)
survey_data$elevel <- as.factor(survey_data$elevel)
survey_data$car <- as.numeric(survey_data$car)

#get familiar with the dataset
#several histograms
hist(survey_data$age, col = brewer.pal(11, 'Spectral'), main = 'Age distribution', xlab = 'Age')

hist(survey_data$salary, col = brewer.pal(11, 'Spectral'), main = 'Salary distribution', xlab = 'Salary')

car_labels = c('BMW','Buick','Cadillac','Chevrolet','Chrysler','Dodge','Ford','Honda','Hyundai',
'Jeep','Kia','Lincoln','Mazda','Mercedes Benz','Mitsubishi','Nissan','Ram','Subaru','Toyota',
'None of the above')

barplot(table(survey_data$car), col = brewer.pal(11, 'Spectral'), names.arg = car_labels, las=2,
        main = 'Car distribution', ylab = 'Frequency')

#binning data
age_bins <- cut(survey_data$age, 5, include.lowest = TRUE, labels = c('young', 'young_adults',
            'adults', 'seniors', 'advanced'))
salary_bins <- cut(survey_data$salary, 5, include.lowest = TRUE, labels = c('<40k', '<70k',
            '<100k', '<130k', '>=131k'))
plot(age_bins, salary_bins, col = brewer.pal(11, 'Spectral'), xlab = 'Age bins', ylab = 'Salary Range',
     main = 'Salary distribution respect to Age')

#scatter plot age vs salary vs brand
survey_data$knn_binary <- survey_data$brand
survey_data$knn_binary <- as.numeric(survey_data$knn_binary)
survey_data[which(survey_data$brand == 'Acer'),]$knn_binary <- 0
survey_data[which(survey_data$brand == 'Sony'),]$knn_binary <- 1
survey_data$knn_binary <- as.numeric(survey_data$knn_binary)
scatter2D(survey_data$age, survey_data$salary, colvar = survey_data$knn_binary, xlab = 'Age',
          ylab = '', main = 'Age & Salary Brand trends', pch = 20, cex = 2, colkey = FALSE,
          las = 1)
par(xpd=TRUE)
legend('topright',c("Acer","Sony"), fill=c("blue","red"), cex = .5)
mtext('Salary', side = 2, outer = TRUE)

####----------------------------------Splitting dataset--------------------------------####

#correlation matrix
corr_dat <- round(cor(survey_data[,c('salary', 'age', 'credit')]),3)
corrplot(corr_dat, method = 'circle', type = 'upper', addCoef.col = 'black', 
         title = 'Correlation Matrix of numerical attributes')

#generate indices for splitting
set.seed(123)

indices <- createDataPartition( y = survey_data$brand,
                               p = .75,
                               list = FALSE)
#split the data in two!
training <- survey_data[indices,]
testing <- survey_data[-indices,]

####-------------------------------Algorithms training--------------------------------####


#function controlling the settings of the model
fitControl <- trainControl(
    method = "repeatedcv", #repeated cross validation
    number = 10, #number of folds
    repeats = 3, #times repeated the cv
    classProbs = FALSE) #shows probability of the output

#formula. I know these two because varImp(rrfit) with brand~. says salary 100%, age 70%
formula <- brand ~ age + salary

#how age and salary affect brand?
dec_tree <- ctree(formula , data=training, controls = ctree_control(maxdepth = 3))
plot(dec_tree, tp_args = list(beside = TRUE))


#knn training model
set.seed(123)
knnfit <- train(formula, # dependent variable to predict (~ means =)(. -> all variables)
                data = training,
                method = "knn",
                trControl = fitControl,
                preProcess = c("center","scale"), # transformacion Z
                tuneLength = 10)

#gradient boosted trees training model
set.seed(123)
gbmfit <- train(formula,
                data = training,
                method = "gbm",
                trControl = fitControl,
                preProcess = c("range"),
                verbose = FALSE)

#support vector machine model
set.seed(123)
svmfit <- train(formula,
                data = training,
                method = "svmRadial",
                trControl = fitControl,
                preProcess = c("range"),
                #tuneGrid = expand.grid(sigma=c()),
                verbose = FALSE)

#random forest with all variables
set.seed(123)
rffit_1 <- train(brand ~.,
                data = training,
                method = "rf",
                trControl = fitControl,
                preProcess = c("range"),
                verbose = FALSE)

#random forest with formula (age and salary)
set.seed(123)
rffit <- train(formula,
                data = training,
                method = "rf",
                trControl = fitControl,
                preProcess = c("range"),
                verbose = FALSE)

####---------------------------------Training Results-------------------------------####

#checking and plotting how important are variables in the random forest
importance_variables <- data.frame(c(varImp(rffit_1)))
importance_variables$labels <- rownames(importance_variables)
barplot(importance_variables$Overall, names.arg = importance_variables$labels, las = 2,
        col = brewer.pal(11, 'Spectral'), main = 'Variable importance', #xlab = 'Variables',
        ylab = 'Importance [%]')

#resampling data models:
# function can be used to collect, summarize and contrast the resampling
# results. Since the random number seeds were initialized to the same value 
# prior to calling `train}, the same folds were used for each model. 
resamps <- resamples(list(GBM = gbmfit,
                          SVM = svmfit,
                          KNN = knnfit,
                          RF = rffit))

#creating a box plot -> it shows the distribution + quartiles
bwplot(resamps, layout = c(4, 1))

#comparing accuracy
dotplot(resamps, metric = "Accuracy")

#comparing accuracy scatter plot matrix
splom(resamps)

#prediction with 4 different models
knnBrand <- predict(knnfit, newdata = testing)
gbmBrand <- predict(gbmfit, newdata = testing)
svmBrand <- predict(svmfit, newdata = testing)
rfBrand <- predict(rffit, newdata = testing)

#Create a confusion matrix for each model to calculate how accurate are the predictions 
knn_CM <- confusionMatrix(data = knnBrand, reference = testing$brand)
gbm_CM <- confusionMatrix(data = gbmBrand, reference = testing$brand)
svm_CM <- confusionMatrix(data = svmBrand, reference = testing$brand)
rf_CM <- confusionMatrix(data = rfBrand, reference = testing$brand)

#how do this models compare? Pull data from confusion matrix
knn_data <- c(knn_CM$byClass)
gbm_data <- c(gbm_CM$byClass)
svm_data <- c(svm_CM$byClass)
rf_data <- c(rf_CM$byClass)

#new data frame with named rows and vectors of the 4 models
mydf <- data.frame( KNN=c(knn_data), GBM=c(gbm_data), SVM=c(svm_data), RF = c(rf_data) )

#plot performance for each model
barplot(t(as.matrix(mydf)), beside=TRUE, col=brewer.pal(4, 'Spectral'),#colors()[c(93,150, 33, 88)], 
        legend=c('KNN', 'GBM', 'SVM', 'RF'), args.legend = list(x="topright"), ylim = c(0,2),
        names.arg = rownames(mydf), main = 'Predicting Performance (training data)', las = 2)
####-------------------------Plot errors of testing predictions---------------------####
knn_result = vector(mode = 'numeric', length = length(testing$brand))
gbm_result = vector(mode = 'numeric', length = length(testing$brand))
svm_result = vector(mode = 'numeric', length = length(testing$brand))
rf_result = vector(mode = 'numeric', length = length(testing$brand))
brand_binary_4err = vector(mode = 'numeric', length = length(testing$brand))
knn = vector(mode = 'numeric', length = length(testing$brand))
gbm = vector(mode = 'numeric', length = length(testing$brand))
svm = vector(mode = 'numeric', length = length(testing$brand))
rf = vector(mode = 'numeric', length = length(testing$brand))

err_df <- data.frame(testing[,7], brand_binary_4err, knnBrand, gbmBrand, svmBrand, 
                     rfBrand, knn_result, gbm_result, svm_result, rf_result,
                     knn, gbm, svm, rf)
err_df[which(err_df$brand == 'Acer'),]$brand_binary_4err <- 0
err_df[which(err_df$brand == 'Sony'),]$brand_binary_4err <- 1

err_df[which(err_df$knnBrand == 'Acer'),]$knn_result <- 0
err_df[which(err_df$knnBrand == 'Sony'),]$knn_result <- 1

err_df[which(err_df$gbmBrand == 'Acer'),]$gbm_result <- 0
err_df[which(err_df$gbmBrand == 'Sony'),]$gbm_result <- 1

err_df[which(err_df$svmBrand == 'Acer'),]$svm_result <- 0
err_df[which(err_df$svmBrand == 'Sony'),]$svm_result <- 1

err_df[which(err_df$rfBrand == 'Acer'),]$rf_result <- 0
err_df[which(err_df$rfBrand == 'Sony'),]$rf_result <- 1

err_df$knn <- abs(err_df$brand_binary_4err - err_df$knn_result)
err_df$gbm <- abs(err_df$brand_binary_4err - err_df$gbm_result)
err_df$svm <- abs(err_df$brand_binary_4err - err_df$svm_result)
err_df$rf <- abs(err_df$brand_binary_4err - err_df$rf_result)

par(mfrow=c(2,2), mai = c(.1,.35,.35,0.1))
scatter2D(testing$age, testing$salary, colvar = err_df$knn, xlab = 'Age',
             ylab = '', main = 'KNN prediction', pch = 20, cex =2, colkey = FALSE,
             las = 1)
par(xpd=TRUE)
legend('topright',c("Correct","Incorrect"), fill=c("Blue","Red"), cex = .5)

scatter2D(testing$age, testing$salary, colvar = err_df$gbm, xlab = 'Age',
          ylab = '', main = 'GBM prediction', pch = 20, cex =2, colkey = FALSE,
          las = 1)
par(xpd=TRUE)
legend('topright',c("Correct","Incorrect"), fill=c("Blue","Red"), cex = .5)

scatter2D(testing$age, testing$salary, colvar = err_df$svm, xlab = 'Age',
          ylab = '', main = 'SVM prediction', pch = 20, cex =2, colkey = FALSE,
          las = 1)
par(xpd=TRUE)
legend('topright',c("Correct","Incorrect"), fill=c("Blue","Red"), cex = .5)

scatter2D(testing$age, testing$salary, colvar = err_df$rf, xlab = 'Age',
          ylab = '', main = 'RF prediction', pch = 20, cex =2, colkey = FALSE,
          las = 1)
par(xpd=TRUE)
legend('topright',c("Correct","Incorrect"), fill=c("Blue","Red"), cex = .5)


####-------------------------For Loop (iterating RF ntrees--------------------------####

#write a for loop to iterate over the number of trees and find a performance convergence
results_train <- c()
results_test  <- c()
for(i in c(2:30)){
    cat(i)
    rf_it <- train(formula,
                data = training,
                method = "rf",
                ntree = i,
                trControl = fitControl,
                preProcess = c("range"),
                verbose = TRUE)
    rf_it_pred_train <- predict(rf_it, newdata = training)
    rf_it_pred_test <- predict(rf_it, newdata = testing)
    cm_rf_it_pred_train <- confusionMatrix(data = rf_it_pred_train, reference = training$brand)
    cm_rf_it_pred_test <- confusionMatrix(data = rf_it_pred_test, reference = testing$brand)
    accuracy_p_train <- cm_rf_it_pred_train$overall[1]
    accuracy_p_test <-cm_rf_it_pred_test$overall[1]
    kappa_p_train <- cm_rf_it_pred_train$overall[2]
    kappa_p_test <- cm_rf_it_pred_test$overall[2]
    results_train <- rbind(results_train, c(accuracy_p_train, kappa_p_train,i))
    results_test <- rbind(results_test, c(accuracy_p_test, kappa_p_test,i))
}
results_train <- as.data.frame(results_train)
results_test  <- as.data.frame(results_test)
colnames(results_train) <- c("Accuracy","kappa","ntree")
colnames(results_test) <- c("Accuracy","kappa","ntree")

####-------------------------------Plot RF convergence------------------------------####

#plot the results for the training set
plot(x = results_train$ntree, y = results_train$Accuracy, type = 'l', col = 'red', 
     main = 'RF TRAINING performance on number of trees', xlab = 'Number of trees', ylab = 'Performance')
lines(x = results_train$ntree, y = results_train$kappa, type = 'l', col = 'blue')
legend("bottomright", c("Kappa","Accuracy"), fill=c("blue","red"))

#plot the resutls for the testing set
plot(x = results_test$ntree, y = results_test$Accuracy, type = 'l', col = 'red', 
     main = 'RF TESTING performance on number of trees', xlab = 'Number of trees', ylab = 'Performance',
     ylim = c(0.75, 0.91))
lines(x = results_test$ntree, y = results_test$kappa, type = 'l', col = 'blue')
legend("right", c("Kappa","Accuracy"), fill=c("blue","red"))

####-------------------------Import new dataset (test_real)-------------------------####


#import test set (incomplete)
testSet <- read.csv('datasets/SurveyIncomplete.csv')

#Check for missing values
sum(is.na(testSet$salary))
sum(is.na(testSet$age))
sum(is.na(testSet$elevel))
sum(is.na(testSet$zipcode))
sum(is.na(testSet$credit))

#change datatypes
testSet$brand <- as.factor(testSet$brand)
testSet$zipcode <- as.factor(testSet$zipcode)
testSet$elevel <- as.ordered(testSet$elevel)
testSet$car <- as.numeric(testSet$car)

testSet[,"brand"] <- NA

####---------------------------Getting familiar with dataset--------------------------####

#get used to the new dataset
hist(testSet$age, col = brewer.pal(11, 'Spectral'), main = 'Age distribution testSet', xlab = 'Age')

hist(testSet$salary, col = brewer.pal(11, 'Spectral'), main = 'Salary distribution testSet', 
     xlab = 'Salary')

barplot(table(testSet$car), col = brewer.pal(11, 'Spectral'), names.arg = car_labels, las=2,
        main = 'Car distribution testSet', ylab = 'Frequency')

#binning age & salary
age_bins_testSet <- cut(testSet$age, 5, include.lowest = TRUE, labels = c('young', 'young_adults',
                'adults', 'seniors', 'advanced'))
salary_bins_testSet <- cut(testSet$salary, 5, include.lowest = TRUE, labels = c('<40k', '<70k',
                '<100k', '<130k', '>=131k'))
plot(age_bins_testSet, salary_bins_testSet, col = brewer.pal(11, 'Spectral'), 
     xlab = 'Age bins', ylab = 'Salary Range', 
     main = 'Salary distribution respect to Age testSet')

#correlation matrix
corr_dat_testSet <- round(cor(testSet[,c('salary', 'age', 'credit')]),3)
corrplot(corr_dat_testSet, method = 'circle', type = 'upper', addCoef.col = 'black', 
         title = 'Correlation Matrix of numerical attributes testSet')

####--------------------Predictions with trained Algorithms--------------------------####

#predictions with the new dataset
#prediction with 4 different models
knnBrand_testSet <- predict(knnfit, newdata = testSet)
gbmBrand_testSet <- predict(gbmfit, newdata = testSet)
svmBrand_testSet <- predict(svmfit, newdata = testSet)
rfBrand_testSet <- predict(rffit, newdata = testSet)

testSet$KNN <- knnBrand_testSet
testSet$GBM <- gbmBrand_testSet
testSet$SVM <- svmBrand_testSet
testSet$RF <- rfBrand_testSet

####--------------------Plot Predictions in matrix form ----------------------------####

par(mfrow=c(2,2), mai = c(.1,.35,.35,0.1))
#plot predictions for KNN prediction
testSet$knn_binary <- testSet$brand
testSet$knn_binary <- as.numeric(testSet$knn_binary)
testSet[which(testSet$KNN == 'Acer'),]$knn_binary <- 0
testSet[which(testSet$KNN == 'Sony'),]$knn_binary <- 1
testSet$knn_binary <- as.numeric(testSet$knn_binary)
a<-scatter2D(testSet$age, testSet$salary, colvar = testSet$knn_binary, xlab = 'Age',
          ylab = '', main = 'KNN', pch = 20, cex =2, colkey = FALSE,
          las = 1)
par(xpd=TRUE)
legend('topright',c("Acer","Sony"), fill=c("blue","red"), cex = .5)
#mtext('Salary', side = 2, outer = TRUE)

#plot predictions for GBM prediction
testSet$gbm_binary <- testSet$brand
testSet$gbm_binary <- as.numeric(testSet$gbm_binary)
testSet[which(testSet$GBM == 'Acer'),]$gbm_binary <- 0
testSet[which(testSet$GBM == 'Sony'),]$gbm_binary <- 1
testSet$gbm_binary <- as.numeric(testSet$gbm_binary)
b<-scatter2D(testSet$age, testSet$salary, colvar = testSet$gbm_binary, xlab = 'Age',
          ylab = '', main = 'GBM', pch = 20, cex = 2, colkey = FALSE,
          las = 1)
par(xpd=TRUE)
legend('topright',c("Acer","Sony"), fill=c("blue","red"), cex = .5)
mtext('Salary', side = 2, outer = TRUE)

#plot predictions for SVM prediction
testSet$svm_binary <- testSet$brand
testSet$svm_binary <- as.numeric(testSet$svm_binary)
testSet[which(testSet$SVM == 'Acer'),]$svm_binary <- 0
testSet[which(testSet$SVM == 'Sony'),]$svm_binary <- 1
testSet$svm_binary <- as.numeric(testSet$svm_binary)
c<-scatter2D(testSet$age, testSet$salary, colvar = testSet$svm_binary, xlab = 'Age',
          ylab = '', main = 'SVM', pch = 20, cex = 2, colkey = FALSE,
          las = 1)
par(xpd=TRUE)
legend('topright',c("Acer","Sony"), fill=c("blue","red"), cex = .5)
mtext('Salary', side = 2, outer = TRUE)

#plot predictions for RF prediction
testSet$rf_binary <- testSet$brand
testSet$rf_binary <- as.numeric(testSet$rf_binary)
testSet[which(testSet$RF == 'Acer'),]$rf_binary <- 0
testSet[which(testSet$RF == 'Sony'),]$rf_binary <- 1
testSet$rf_binary <- as.numeric(testSet$rf_binary)
d<-scatter2D(testSet$age, testSet$salary, colvar = testSet$rf_binary, xlab = 'Age',
          ylab = '', main = 'RF', pch = 20, cex = 2, colkey = FALSE,
          las = 1)
par(xpd=TRUE)
legend('topright',c("Acer","Sony"), fill=c("blue","red"), cex = .5)
mtext('Salary', side = 2, outer = TRUE)

