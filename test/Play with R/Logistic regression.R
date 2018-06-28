# Load the raw training data and replace missing values with NA
training.data.raw <- read.csv('train_titanic_full.csv',header=T,na.strings=c(""))

# Output the number of missing values for each column
sapply(training.data.raw,function(x) sum(is.na(x)))

# Quick check for how many different values for each feature
sapply(training.data.raw, function(x) length(unique(x)))

# A visual way to check for missing data
#install.packages('Amelia')
library(Amelia)
missmap(training.data.raw, main = "Missing values vs observed")

# Subsetting the data
data <- subset(training.data.raw,select=c(2,3,5,6,7,8,10,12))

# Substitute the missing values with the average value
data$Age[is.na(data$Age)] <- mean(data$Age,na.rm=T)

#make sex numeric
data$Sex <- factor(data$Sex, levels=c("male","female"), labels=c(0,1))
data$Sex <- as.integer(as.character(data$Sex))

# Remove rows (Embarked) with NAs
data <- data[!is.na(data$Embarked),]
rownames(data) <- NULL

# Train test splitting
train <- data[1:800,]
test <- data[801:889,]

# Model fitting
model <- glm(Survived ~.,family=binomial(link='logit'),data=train)
summary(model)

# Analysis of deviance
anova(model,test="Chisq")

# McFadden R^2 I don't have a clue what this means..
#install.packages('pscl')
library(pscl)
pR2(model)

#-------------------------------------------------------------------------------
# MEASURING THE PREDICTIVE ABILITY OF THE MODEL

# If prob > 0.5 then 1, else 0. Threshold can be set for better results
fitted.results <- predict(model,newdata=subset(test,select=c(2,3,4,5,6,7,8)),type='response')
fitted.results <- ifelse(fitted.results > 0.6,1,0)
fitted.results<- as.numeric(fitted.results)
misClasificError <- mean(fitted.results != test$Survived)
print(paste('Accuracy',1-misClasificError))

# Confusion matrix
library(caret)
fitted.results<-as.factor(fitted.results)
test$Survived<-as.factor(fitted.results)
confusionMatrix(data=fitted.results, reference=test$Survived)

library(ROCR)
# ROC and AUC
p <- predict(model, newdata=subset(test,select=c(2,3,4,5,6,7,8)), type="response")
pr <- prediction(p, test$Survived)
# TPR = sensitivity, FPR=specificity
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

#'inferential statistics
#'z-score
#create a new data frame for upper class passengers
subset_data = function (data,class_passengers,sex){
  if (class_passengers == 'all'){
    result<-data
  }
  else {
    result<-subset(data,data$Pclass == class_passengers)
  }
  return (result)  
}


#function for z test
z.test2 = function(a, b, n){
  sample_mean = mean(a)
  pop_mean = mean(b)
  c = nrow(n)
  var_b = var(b)
  zeta = (sample_mean - pop_mean) / (sqrt(var_b/c))
  return(zeta)
}

#call function
#difference between class 1,class 2 and class 3
new_data<-subset_data(data,1)
z_class1<-z.test2(new_data$Survived, data$Survived, new_data)
new_data<-subset_data(data,2)
z_class2<-z.test2(new_data$Survived, data$Survived, new_data)
new_data<-subset_data(data,3)
z_class3<-z.test2(new_data$Survived, data$Survived, new_data)
print(paste('Z_score class 1,2,3:',z_class1,z_class2,z_class3))

#difference between sex



#chi-square test between Survived, Sex, Pclass
chisq.test(data$Survived, data$Sex,data$Pclass)

#another method of chi-square test
summary(table(data$Survived,data$Sex,data$Pclass))

#pearson correlation test
cor.test(data$Survived, data$Sex, method = 'pearson')

