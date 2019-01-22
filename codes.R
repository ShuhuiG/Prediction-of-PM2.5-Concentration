prsa <- read.csv2("/Users/shuhuiguo/Desktop/Assessment/pollution.csv",
                  header=TRUE, sep=",")

# check the class of variables
sapply(prsa, class)

# convert the change the factor values to numeric
prsa$temp <- as.numeric(as.character(prsa$temp))
prsa$press <- as.numeric(as.character(prsa$press))
prsa$wnd_spd <- as.numeric(as.character(prsa$wnd_spd))
prsa$pollution <- as.numeric(as.character(prsa$pollution))

# observe data
summary(prsa)

# split training and testing data
# the data before 2014-01-01 is training data, the data after 2014-01-01 is testing data
which(prsa$date == "2014-01-01 00:00:00")
test <- prsa[c(33097:41757), ]
train <- prsa[-c(33097:41757), ]
xtrain <- model.matrix(~ .-1, train[, c(3:9)])
xtest <- model.matrix(~ .-1, test[, c(3:9)])
ytrain <- train$pollution
ytest <- test$pollution

library(glmnet)
# k is the size of each fold
num_folds <- 10
k <- floor(dim(xtrain)[1]/num_folds)
lambda <- seq(0.01, 5, 0.01)
errors <- matrix(0, nrow = 500, ncol = 9)
p<-1
for (j in seq(0.01, 5, 0.01)){
  for (i in (2:num_folds)){
    split = (i-1)/i
    x = xtrain[c(1:(k*i)),]
    y = ytrain[c(1:(k*i))]
    index = floor(dim(x)[1] * split)
    
    # folds used to train the model
    x_traincv = x[c(1:index),]
    y_traincv = y[c(1:index)]
    
    # fold used to test the model
    x_testcv = x[-c(1:index),]
    y_testcv = y[-c(1:index)]
    
    m1 <- glmnet(x_traincv, y_traincv, family = "gaussian", alpha = 1, lambda = j)
    y_pre1 <- predict(m1, x_testcv)
    errors[p,(i-1)] <- mean((y_pre1 - y_testcv)^2)
  }
  p<-p+1
}

# select the lambda which gives the lowest error
errors_lambda <- apply(errors, 1, mean)
lambda_use <- lambda[which.min(errors_lambda)]

# fit lasso with this lambda
glm.fit <- glmnet(xtrain, ytrain, alpha = 1, lambda = lambda_use)
glm_pre <- predict(glm.fit, xtest)
error_glm <- sqrt(mean((glm_pre - ytest)^2))
coef(glm.fit)

# plot the predicted and true values
library(ggplot2)
data_m1 <- data.frame(x = seq(1, length(glm_pre),1), s0 = glm_pre, s1 = ytest)
gplot <- ggplot(data_m1) + geom_line(aes(x = x, y = s1, color = "true value"))
gplot <- gplot + geom_line(aes(x = x, y = s0, color = "predicted value"))
gplot <- gplot + labs(x = "sequence",y = "pm2.5")
gplot <- gplot + labs(title = "Lasso Regression") + theme(plot.title = element_text(hjust = 0.5))
gplot <- gplot + scale_colour_manual(values = c("deepskyblue", "darkorange"))
gplot


## random forest
library(randomForest)

prsa <- read.csv2("/Users/shuhuiguo/Desktop/Assessment/pollution.csv",
                  header=TRUE, sep=",")
# convert the change the factor values to numeric
prsa$temp <- as.numeric(as.character(prsa$temp))
prsa$press <- as.numeric(as.character(prsa$press))
prsa$wnd_spd <- as.numeric(as.character(prsa$wnd_spd))
prsa$pollution <- as.numeric(as.character(prsa$pollution))

# split training and testing data
# the data before 2014-01-01 is training data, the data after 2014-01-01 is testing data
which(prsa$date == "2014-01-01 00:00:00")
test <- prsa[c(33097:41757), ]
train <- prsa[-c(33097:41757), ]
xtrain <- train[, c(3:9)]
xtest <- test[, c(3:9)]
ytrain <- train$pollution
ytest <- test$pollution

m2 = randomForest(xtrain, ytrain, ntree = 500, mtry = 7/3, nodesize = 5) # mtry=p/3
rf_pre <- predict(m2, xtest)
error_rf <- sqrt(mean((rf_pre-ytest)^2))

# plot the variable importance
RandomForest = randomForest(xtrain, ytrain, ntree = 500, mtry = 7/3, nodesize = 5) # mtry=p/3
varImpPlot(RandomForest)

library(ggplot2)
data_m2 <- data.frame(x = seq(1, length(rf_pre),1), s0 = rf_pre, s1 = ytest)
gplot <- ggplot(data_m2) + geom_line(aes(x = x, y = s1, color = "true value"))
gplot <- gplot + geom_line(aes(x = x, y = s0, color = "predicted value"))
gplot <- gplot + labs(x = "sequence",y = "pm2.5")
gplot <- gplot + labs(title = "Random Forest") + theme(plot.title = element_text(hjust = 0.5))
gplot <- gplot + scale_colour_manual(values = c("deepskyblue", "darkorange"))
gplot


## LSTM
# These are the codes for plot. The codes for model are in 'lstm.py'
result <- read.csv("/Users/shuhuiguo/Desktop/Assessment/result_train_new.csv", header = T)
library(ggplot2)
data_m3 <- data.frame(x = seq(1, length(result$predict),1), s0 = result$predict, s1 = result$true)
gplot <- ggplot(data_m3) + geom_line(aes(x = x, y = s1, color = "true value"))
gplot <- gplot + geom_line(aes(x = x, y = s0, color = "predicted value"))
gplot <- gplot + labs(x = "sequence",y = "pm2.5")
gplot <- gplot + labs(title = "LSTM Neural Network") + theme(plot.title = element_text(hjust = 0.5))
gplot <- gplot + scale_colour_manual(values = c("deepskyblue", "darkorange"))
gplot
