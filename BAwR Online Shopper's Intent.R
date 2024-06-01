setwd("E:/UTD/OneDrive - The University of Texas at Dallas/2023/Spring 2023/Business Analytics with R/Group Project/Group 9")
rm(list=ls())
cat("\014")

#import libraries
#install.packages("ggplot2")

library(caret)
library(data.table)

#import csv file
df <-read.csv("online_shoppers_intention.csv")
View(df)

#descriptive statistics
head(df)
summary(df)

#removing duplicates
df_duplicate <- nrow(df[duplicated(df),])
df <- df[!duplicated(df),]

#identification of missing values
which(is.na(df))

#Renaming June to Jun for convenience of plotting
df$Month <- as.character(df$Month)
df$Month[df$Month == "June"] <- "Jun"
df$Month <- as.factor(df$Month)
df$Month = factor(df$Month, levels = month.abb)

#DATA PRE-PROCESSING

#Transforming categorical attributes into factor types
#Factor cols Operating sys, browser, region, traff, visitor type, weekend and revenue and save in df

df$OperatingSystems = as.factor(df$OperatingSystems)
df$Browser = as.factor(df$Browser)
df$Region = as.factor(df$Region)
df$TrafficType = as.factor(df$TrafficType)
df$VisitorType = as.factor(df$VisitorType)
df$Weekend = as.integer(df$Weekend)
df$Revenue = as.integer(df$Revenue)

levels(df$OperatingSystems)

# Save the above original dataframe in df_original & factorize Revenue col

df_original <- df
df_original$Revenue <- as.factor(df_original$Revenue)
levels(df_original$Revenue)

#Split in training and testing data for df_original

#install.packages('rsample')
library(rsample)
set.seed(123)

split_df_original <- initial_split(df_original, prop = .7, strata = "Revenue")
train_df_original <- training(split_df_original)
test_df_original <- testing(split_df_original)

tab1 <- table(train_df_original$Revenue) 
prop.table(tab1)
print(tab1)

tab2 <- table(test_df_original$Revenue) 
prop.table(tab2)
print(tab2)


#Preprocess the continuous attributes by splitting into categorical & numerical ones and scaling numerical values
train_numerical <- train_df_original[,1:10] 
train_categorical <- train_df_original[,11:18]
test_numerical <- test_df_original[,1:10] 
test_categorical = test_df_original[,11:18]

#Utilization of scaling function

train_scaled = scale(train_numerical)
test_scaled = scale(test_numerical, center=attr(train_scaled, "scaled:center"), scale=attr(train_scaled, "scaled:scale"))

#scaling of x and y variables with same center and sd

#Column binding
train_data <- cbind(train_scaled, train_categorical)
test_data <- cbind(test_scaled, test_categorical)
summary(train_data)


#DATA MODELING
  
#PREDICTION USING DIFFERENT ALGORITHMS

#install.packages("e1071")
#install.packages("caret")
library(e1071)
library(caret)

########### Naive Bayes Classifier ########### 

fit.nb <- naiveBayes(Revenue ~  Administrative + Administrative_Duration + Informational + 
                       Informational_Duration + ProductRelated + ProductRelated_Duration + BounceRates + ExitRates + 
                       PageValues + SpecialDay + Month + OperatingSystems + Browser + Region + TrafficType + 
                       VisitorType + Weekend, data = train_data)

#Evaluate Performance using Confusion Matrix
actual <- test_data$Revenue
# predict class probability
nbPredict <- predict(fit.nb, test_data, type = "raw")
# predict class membership
nbPredictClass <- predict(fit.nb, test_data, type = "class")
cm <- table(nbPredictClass, actual)
cm

print(cm)
#Accuracy
tp = 2613
tn= 373
fn= 200
fp= 477

A1=(tp+tn)/(tp+tn+fp+fn); A1
A1= 0.8151788    ## Accuracy



###########  K-Nearest Neighbors ###########

#install.packages("e1071")
#install.packages("caret")
#library(e1071)
#library(caret)
# Checking distribution of outcome classes -> very few class = "1"
prop.table(table(train_data$Revenue)) * 100
prop.table(table(test_data$Revenue)) * 100
prop.table(table(df_original$Revenue)) * 100

# 10-fold cross-validation
ctrl <- trainControl(method="cv", number=10) 

knnFit <- train(Revenue ~ Administrative + Administrative_Duration + Informational + 
Informational_Duration + ProductRelated + ProductRelated_Duration + BounceRates + ExitRates 
+ PageValues + SpecialDay + OperatingSystems + Region +
VisitorType + Weekend,data = train_data, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneGrid = expand.grid(k = 1:10))

# Kappa is a more useful measure to use on problems that have imbalanced classes.
knnFit
#The final value used for the model was k = 10.
A2 = 0.8728658   ## Accuracy

# plot the # of neighbors vs. accuracy (based on repeated cross validation)
plot(knnFit)
ggplot(data=knnFit$results, aes(k, Accuracy)) + geom_line() + scale_x_continuous(breaks=1:10)
ggplot(data=knnFit$results, aes(k, Kappa)) + geom_line() + scale_x_continuous(breaks=1:10)


########### Decision Tree ########### 

set.seed(435)

#install.packages("rpart")
library(rpart)
fit <- rpart(Revenue~.,
             data=train_data,
             method="class",
             control=rpart.control(xval = 10,minsplit=50),
             parms=list(split="gini"))
fit

library(rpart.plot)
rpart.plot(fit,
           type=1,
           extra = 1,
           main="Decision Tree")


#Decision Tree Confusion Matrix
df1.predict <- predict(fit, test_data, type="class")
df1.actual  <- test_data$Revenue

cm2 <- confusionMatrix(df1.predict, df1.actual)
cm2
A3 = 0.8919 ## Accuracy


########### SVM ########### 

# Oversampling for data frame -  df_original
library(ROSE)
#library(e1071)

N_df_new = 2*length(which(train_data$Revenue == 0))
df_original_over <- ovun.sample(Revenue~.,data = train_data, method= 'over', N = N_df_new, seed = 2020)$data

##### Linear SVM #####
set.seed(123)
#Train the model
svm_fit = svm(as.factor(Revenue)~., data=df_original_over, kernel = "linear", scale = FALSE)

#Predict
pred <- predict(svm_fit, newdata = test_data)

#Confusion Matrix and Metrics of Linear SVM
print("Linear SVM")

cm3 <- confusionMatrix(pred, factor(test_data$Revenue))
cm3
A4 = 0.875  #Accuracy

##### Radial SVM #####
#Train the model
svm_fit2 = svm(as.factor(Revenue)~., data=df_original_over, kernel = "radial", scale = FALSE)

#Predict
pred_radial <- predict(svm_fit2, newdata = test_data)

#Confusion Matrix and Metrics of RBF SVM
print("Radial SVM")

cm4 <- confusionMatrix(pred_radial, factor(test_data$Revenue))
cm4
A5 = 0.8785 ## Accuracy


######  Comparison of Accuracy between different models #####

X<-c("Naive Bayes","KNN","Decision Tree", "Linear SVM","Radial SVM")
Y<-round(c(A1,A2,A3,A4,A5),4)

X_name <- "model"
Y_name <- "accuracy"

df_result <- data.frame(X,Y)
names(df_result) <- c(X_name,Y_name)

ggplot(df_result,aes(x=model,y=accuracy,fill=model))+geom_bar(stat = "identity") + geom_text(aes(label=accuracy),position=position_dodge(width=0.9), vjust=-0.25)

