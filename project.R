#### END OF COURSE PROJECT ####
library(ggplot2)
library(caret)
library(kernlab)
library(gridExtra)

## Read in and prepare data
pmltrain = read.csv("/Users/cmay/Documents/Training/pml-training.csv", stringsAsFactors=FALSE)
pmltest  = read.csv("/Users/cmay/Documents/Training/pml-testing.csv",  stringsAsFactors=FALSE)
  names(pmltrain)[names(pmltrain)=='X'] <- 'id'
  names(pmltest )[names(pmltest) =='X'] <- 'id'

  table(pmltrain$classe, exclude=FALSE)
  table(pmltest$classe)

  nrow(pmltrain[is.na(pmltrain$classe),])
  nrow(pmltrain[pmltrain$classe == " ",])

## Get the number of NA records in a dataframes of columns
countnas <- function(z) {
  na_count <-sapply(z, function(y) sum(length(which(is.na(y) | y == ""))))
  na_count <- data.frame(na_count)
  na_count$pct <- round(na_count$na_count  / nrow(z),4)
  return(na_count)
}

## Writeout table for report
#countnas(pmltrain)
#outtrain <- countnas(pmltrain)
#write.csv(cbind(names(pmltrain),outtrain),file="/Users/cmay/Documents/Training/outtrain.csv", quote=FALSE, row.names=FALSE)


## Remove vars with at least one NA, most have 19K NA's
## Set sample seed here.  This will be run 3 times by changing the seed to create different samples
## Could also build a function for all of this work if I want to
  ## 1) seed 41;
  ## 2) seed 92621;
  ## 3) seed 52800
set.seed(52800) 
pmltrain.clean <- pmltrain[ , apply(pmltrain, 2, function(x) {!any(is.na(x)) & !any( x == "") } )]
pmltest.clean  <- pmltest[ ,  apply(pmltest, 2, function(x) {!any(is.na(x)) & !any( x == "") } )]

## Remove character, id and timestamp variables from final test set
pmltest.clean  <- pmltest.clean[,-c(1:7)]
#str(pmltest.clean)

## Build cross-validation training and test sets
intrain  = createDataPartition(pmltrain.clean$classe, p=3/4)[[1]]
training = pmltrain.clean[ intrain, ]
testing  = pmltrain.clean[-intrain, ]
#str(training); str(testing)


## Remove indicator, character and time variables from training data
training <- training[,-c(1:7)]
#str(training); summary(training); head(training,4)
#nzv <- nearZeroVar(training, saveMetrics=TRUE)
#nzv

## Prepare 'clean' testing data set for trained models
testing.clean <- testing[ , apply(testing, 2, function(x) {!any(is.na(x)) & !any( x == "") } )]
testing.clean <- testing.clean[,-c(1:7)]
#str(testing.clean)



##########################################################################################
#### Exploratory Data Analysis
## Change 'turn.on' value to 1 if I want to view plots again
turn.on <- 0
if (turn.on == 1) {
  featurePlot(x= training$classe, y= training[,c('roll_belt', 'pitch_belt', 'yaw_belt')], plot='pairs')
  qplot(classe, magnet_arm_x, data=training)
  qplot(magnet_arm_x,roll_dumbbell, col=classe,   data=training)
  qplot(classe, total_accel_forearm, fill=classe, data=training, geom=c("boxplot"))
  qplot(magnet_forearm_z, col=classe, data=training,geom="density")
  plot(training$classe, training$roll_belt)
  
  ## Build PDF files of different types of eda plots by variable
  pdf(file="/Users/cmay/Downloads/densityplots.pdf")
  par(mar=c(3,3,2,0))
  for (k in 1:length(training)) {
    print(qplot(training[,k], col=classe, data=training, geom="density", xlab=colnames(training[k]), 
                main=paste0("Variable #",k) ) )
  }
  dev.off()
  
  pdf(file="/Users/cmay/Downloads/boxplots.pdf")
  par(mar=c(3,3,2,0))
  for (k in 1:(length(training)-1) ) {
    p1 <- qplot(classe, training[,k], fill=classe, data=training, geom=c("boxplot"), ylab=colnames(training[k]),
                main=paste0("Boxplot of ", colnames(training[k]) ) )
    
    p2 <- qplot(classe, training[,k], fill=classe, data=training, geom=c("boxplot", "jitter"), ylab=colnames(training[k]),
                main=paste0("Boxplot of ", colnames(training[k]) ) )
    
    grid.arrange(p1,p2,ncol=2)  
  }
  dev.off()
  
  pdf(file="/Users/cmay/Downloads/histograms.pdf")
  par(mar=c(3,3,2,0))
  for (k in 1:(length(training)-1)) {
    hist(training[,k], breaks=100, main=paste0("Histogram of ",colnames(training[k])))
  }
  dev.off()
  
  pdf(file="/Users/cmay/Downloads/histograms_log.pdf")
  par(mar=c(3,3,2,0))
  for (k in 1:(length(training)-1)) {
    hist(log10(training[,k]), main=paste0("Histogram of ",colnames(training[k])))
  }
  dev.off()
}


## Remove predictors that have little to know relationship with outcome variable after eda
## May not use this in the final version
training2 <- subset(training, select=-c(gyros_belt_x,gyros_belt_y,gyros_belt_z,gyros_arm_x,gyros_arm_y,gyros_arm_z,
                                        gyros_dumbbell_x,gyros_dumbbell_y,gyros_dumbbell_z,gyros_forearm_x,gyros_forearm_y,gyros_forearm_z) )

testing2 <- subset(testing.clean, select=-c(gyros_belt_x,gyros_belt_y,gyros_belt_z,gyros_arm_x,gyros_arm_y,gyros_arm_z,
                                            gyros_dumbbell_x,gyros_dumbbell_y,gyros_dumbbell_z,gyros_forearm_x,gyros_forearm_y,gyros_forearm_z) )
#str(training2); str(testing2)

##########################################################################################
#### Principal Component Analysis ####
#### Went through this but did not use PCA in my final report ####
#m <- abs(cor(training[,-c(1:6,60)]))
#diag(m) <- 0
#which(m > 0.90, arr.ind=T)
#plot(training$accel_belt_x, training$pitch_belt)
#plot(training$accel_belt_z, training$roll_belt)
#X <- -0.12*training$total_accel_belt - 0.99*training$roll_belt
#Y <- -0.99*training$total_accel_belt + 0.12*training$roll_belt
#plot(X,Y)

#typeColor <- training$classe
#table(typeColor)
#typeColor[typeColor =="A"] <- 1
#typeColor[typeColor =="B"] <- 2
#typeColor[typeColor =="C"] <- 3
#typeColor[typeColor =="D"] <- 4
#typeColor[typeColor =="E"] <- 5

## PCA- full set of variables for
#nrow(pmltrain)
#prcomp <- prcomp(training)
#prcomp$rotation
#str(prcomp)
#prcomp

#plot(prcomp$x[,1], prcomp$x[,2], xlab="PC1",ylab="PC2")
#plot(prcomp$x[,1], prcomp$x[,2], col=typeColor, xlab="PC1",ylab="PC2", cex=1)
#classPC <- predict(prcomp,training)
#plot(classPC[,1], classPC[,2], col=typeColor, xlab="PC1",ylab="PC2")


## PCA- reduced set of variables with caret package for testing
#smallset <- training2[,c(1:10)]
#str(smallset)
#preproc <- preProcess(training2, method="pca", thresh=0.90)
#str(preproc)
#preproc$rotation
#plot(preproc$x[,1], preproc$x[,2], col=typeColor, xlab="PC1",ylab="PC2")


##########################################################################################
#### Build models
#### Rpart- classification and regression tree
print("Start of RPART")
Sys.time()
mod.rpart <- train(classe ~ ., data=training, method="rpart")
mod.rpart$finalModel

## Get predictions and confusion(accuracy matrix) for rpart
pred.rpart <- predict(mod.rpart, testing)
confusionMatrix(testing$classe, pred.rpart)

## See what predictions look like on "real" test set
pred.rpart2 <- predict(mod.rpart, pmltest.clean)
pred.rpart2

## Save a plot of the tree
if (turn.on == 1) {
  pdf(file="/Users/cmay/Downloads/rpart_tree92621.pdf")
  par(mar=c(1,4,1,4))
  plot(mod.rpart$finalModel, uniform=TRUE, main="Classification Tree", cex.main=0.75)
  text(mod.rpart$finalModel, use.n=TRUE, all=TRUE, cex=0.7,pos=1)
  dev.off()
}

print("Start of RANDOM FORESTS- reduced set of variables, testing for time ")
Sys.time()
#### RF- Random Forest
#mod.rf <- train(classe ~ roll_belt + yaw_belt + total_accel_belt + accel_belt_y + accel_belt_z + magnet_belt_y + magnet_belt_z +
#                              roll_arm + pitch_arm + total_accel_arm + accel_arm_x + accel_arm_y + accel_arm_z + magnet_arm_x +
#                              magnet_arm_y + magnet_arm_z + roll_dumbbell + pitch_dumbbell, data=training2, method="rf")
mod.rf <- train(classe ~ ., data=training2, method="rf")
summary(mod.rf)
mod.rf$finalModel

## Get predictions and confusion(accuracy matrix) for rf
pred.rf <- predict(mod.rf, testing2)
confusionMatrix(testing2$classe, pred.rf)

## See what predictions look like on "real" test set
pred.rf2 <- predict(mod.rf, pmltest.clean)
pred.rf2

print("Start of GBM")
Sys.time()
#### GBM- boosting with trees
#mod.gbm <- train(classe ~ roll_belt + yaw_belt + total_accel_belt + accel_belt_y + accel_belt_z + magnet_belt_y + magnet_belt_z +
#                              roll_arm + pitch_arm + total_accel_arm + accel_arm_x + accel_arm_y + accel_arm_z + magnet_arm_x +
#                              magnet_arm_y + magnet_arm_z + roll_dumbbell + pitch_dumbbell, data=training2, method="gbm", verbose=FALSE)
mod.gbm <- train(classe ~ ., data=training2, method="gbm", verbose=FALSE)
mod.gbm$finalModel

## get predictions and confusion(accuracy matrix) for gbm
pred.gbm <- predict(mod.gbm, testing2)
confusionMatrix(testing2$classe, pred.gbm)

## See what predictions look like on "real" test set
pred.gbm2 <- predict(mod.gbm, pmltest.clean)
pred.gbm2

#### Ensembled Model
## Stack RPART, RF and GBM models
stacked.dat <- data.frame(pred.rpart, pred.rf, pred.gbm, classe=testing2$classe)
str(stacked.dat)

## Train the combined stacked predictors using random forests
mod.stack  <- train(classe ~., method="rf", data=stacked.dat)

## Get predictions and confusion(accuracy matrix) for stacked models
pred.stack <- predict(mod.stack, stacked.dat)
confusionMatrix(testing2$classe, pred.stack)


##### Final Predictions
## Make final predictions for end of project quiz using Random Forests as it had the best out of sample accuracy
final.prediction <- predict(mod.rf, pmltest.clean)
final.prediction
summary(final.prediction)

Sys.time()