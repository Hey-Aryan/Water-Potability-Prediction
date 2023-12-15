wq<-read.csv("/Users/aryankarande/Desktop/VIT/DS/water_quality.csv")

#selecting required features
library("dplyr")
wq <- wq %>% select(c(ph,	Hardness, Chloramines,	Sulfate,	Conductivity,	Organic_carbon,	Trihalomethanes,Turbidity,	Potability))
str(wq)

#KNN imputation
set.seed(9999)
library(DMwR2)
wq<-knnImputation(wq,k=6,scale = T,meth = "weighAvg",distData = NULL)

# mice imputation
library("mice")
mice_imp<-mice(wq,m=5,method = c("pmm","","","","pmm","","","pmm","",""), maxit = 20)
wq<-complete(mice_imp,5)

summary(wq)

#outliers 
wq <- subset(wq, wq$ph > (quantile(wq$ph, .25) - 1.5*IQR(wq$ph)) & wq$ph< (Q3 <- quantile(wq$ph, .75) + 1.5*IQR(wq$ph)))
wq <- subset(wq, wq$Hardness > (quantile(wq$Hardness, .25) - 1.5*IQR(wq$Hardness)) & wq$Hardness < (Q3 <- quantile(wq$Hardness, .75) + 1.5*IQR(wq$Hardness)))
wq <- subset(wq, wq$Chloramines > (quantile(wq$Chloramines, .25) - 1.5*IQR(wq$Chloramines)) & wq$Chloramines < (Q3 <- quantile(wq$Chloramines, .75) + 1.5*IQR(wq$Chloramines)))
wq <- subset(wq, wq$Sulfate > (quantile(wq$Sulfate, .25) - 1.5*IQR(wq$Sulfate)) & wq$Sulfate < (Q3 <- quantile(wq$Sulfate, .75) + 1.5*IQR(wq$Sulfate)))
wq <- subset(wq, wq$Conductivity > (quantile(wq$Conductivity, .25) - 1.5*IQR(wq$Conductivity)) & wq$Conductivity < (Q3 <- quantile(wq$Conductivity, .75) + 1.5*IQR(wq$Conductivity)))
wq <- subset(wq, wq$Organic_carbon > (quantile(wq$Organic_carbon, .25) - 1.5*IQR(wq$Organic_carbon)) & wq$Organic_carbon < (Q3 <- quantile(wq$Organic_carbon, .75) + 1.5*IQR(wq$Organic_carbon)))
wq <- subset(wq, wq$Trihalomethanes > (quantile(wq$Trihalomethanes, .25) - 1.5*IQR(wq$Trihalomethanes)) & wq$Trihalomethanes < (Q3 <- quantile(wq$Trihalomethanes, .75) + 1.5*IQR(wq$Trihalomethanes)))
wq <- subset(wq, wq$Turbidity > (quantile(wq$Turbidity, .25) - 1.5*IQR(wq$Turbidity)) & wq$Turbidity < (Q3 <- quantile(wq$Turbidity, .75) + 1.5*IQR(wq$Turbidity)))
boxplot(wq,col="green")

#box plot for outlier visualization
boxplot(wq$Solids, col = "green",   main="Solids")

#bar plot potability to check class imbalance
barplot(table(wq$Potability),main="Potability Count",xlab="Potability",ylab="Count",border="black",col="green")

#Random Oversampling
library(ROSE)
table(wq$Potability)
prop.table(table(wq$Potability))
wq<- ovun.sample(Potability~.,data = wq, method = "over")
wq<-wq$data
table(wq$Potability)

#SMOTE oversampling
library(smotefamily)
smote<-SMOTE(X=wq,target=wq$Potability,K=5,dup_size =1)
wq<-smote$data
wq<-wq[,-10]
table(wq$Potability)

#Normalization - scaling data b/w 0 to 1
wq_mod<-wq[,1:9]
n2<-function(b){
  (b-min(b))/(max(b)-min(b))
}
wq1<-as.data.frame(lapply(wq_mod, n2))
wq<-cbind(wq1,Potability=wq[,10])
str(wq)
table(wq$Potability)
library(caret)

#spliting dataset
set.seed(9999)
wq$Potability=as.factor(wq$Potability)
index <-sample(2, nrow(wq), replace=T, prob = c(0.80,0.20))
train<-wq[index==1,]
test<-wq[index==2,]

#Knn classifier
set.seed(9999)
library(caret)
library(class)
pred<-knn(train,test,train$Potability,k=2)
confusionMatrix(table(test[,"Potability"],pred))

#decision tree
library(rpart)
tree<-rpart(Potability~., train)
pred<-predict(tree,newdata = test, type = "class")
confusionMatrix(table(test[,"Potability"],pred))

#random forest
set.seed(9999)
library(randomForest)
library(caret)
train$Potability <- as.factor(train$Potability)
rfm<-randomForest(Potability~., data = train, ntree=  200)
importance(rfm)
varImpPlot(rfm)
pred<-predict(rfm, test)
confusionMatrix(table(test[,"Potability"],pred))

#random forest with k fold
library(caret)
set.seed(9999)
wq$Potability=as.factor(wq$Potability)
index <- createDataPartition(wq[,"Potability"],p=0.8,list=FALSE)
train <- wq[index,]
test <- wq[-index,]
ctrl<-trainControl(method = "cv",number = 10)
model<-train(Potability~.,data = train,method="rf",trControl=ctrl)
pred<-predict(model,test)
confusionMatrix(table(test[,"Potability"],pred))

#XGBoost
library("xgboost")
library(magrittr)
library(dplyr)
library(Matrix)
set.seed(1234)
trainxgb<-sparse.model.matrix(Potability~., data = train)
train_label<-train[,"Potability"]
train_matrix<-xgb.DMatrix(data = as.matrix(trainxgb), label=train_label)
testxgb<-sparse.model.matrix(Potability~., data = test)
test_label<-test[,"Potability"]
test_matrix<-xgb.DMatrix(data = as.matrix(testxgb), label=test_label)
nc<-length(unique(train_label))
xgb_prm<-list("objective"="multi:softprob",
              "eval_metric"="mlogloss",
              "num_class"=nc)
watchlist<- list(train=train_matrix, test= test_matrix)
bst_model<-xgb.train(params = xgb_prm,
                     data = train_matrix,
                     nrounds = 100,
                     watchlist = watchlist)
p<-predict(bst_model, newdata = test_matrix) #probabilities
pred<-matrix(p, nrow = nc, ncol = length(p)/nc)%>%
  t() %>%
  data.frame() %>%
  mutate(label= test_label, max_prob=max.col(., "last")-1)
confusionMatrix(table(Prediction= pred$max_prob, Actual= pred$label))

#logistic regression
model<-glm(Potability~.,  data = train, family = binomial)
p <- predict(model, test, type="response")
pred<-round(p)
pred<-as.factor(pred)
test$Potability=as.factor(test$Potability)
confusionMatrix(pred,test$Potability)

#SVM radial
library(e1071)
library(caret)
str(train)
classifier = svm(formula = Potability ~ ., data = train,type = 'C-classification', kernel = 'radial')
pred = predict(classifier, newdata = test[-10])
pred<-as.factor(pred)
test$Potability=as.factor(test$Potability)
confusionMatrix(pred,test$Potability)

#SVM linear
library(e1071)
train$Potability<-as.factor(train$Potability)
svmmodel<-svm(Potability~., data= train, kernel="linear", cost=10)
summary(svmmodel)
pred<-predict(svmmodel, test)
pred<-as.factor(pred)
test$Potability=as.factor(test$Potability)
confusionMatrix(pred,test$Potability)

#confusion Matrix visualization
draw_confusion_matrix <- function(cm) {
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title('Decision Tree', cex.main=2)
  # create the matrix 
  rect(150, 430, 240, 370, col='#0080ff')
  text(195, 435, '0', cex=1.2)
  rect(250, 430, 340, 370, col='#00bfff')
  text(295, 435, '1', cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col='#00bfff')
  rect(250, 305, 340, 365, col='#0080ff')
  text(140, 400, '0', cex=1.2, srt=90)
  text(140, 335, '1', cex=1.2, srt=90)
  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(40, 90, names(cm$byClass[5]), cex=1.5, font=2)
  text(40, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.5)
  text(60, 90, names(cm$byClass[6]), cex=1.5, font=2)
  text(60, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.5)
  text(80, 90, names(cm$byClass[7]), cex=1.5, font=2)
  text(80, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.5)
  text(20, 90, names(cm$overall[1]), cex=1.5, font=2)
  text(20, 70, round(as.numeric(cm$overall[1]), 3), cex=1.5)
}  
draw_confusion_matrix(cm) # cm--> confusion matrix
