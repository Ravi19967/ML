###############################################################
#                                                             #
# Feature Selection for randomForest using Random Approach    #
#                                                             #
###############################################################

#--------------------------------------------------------------
# Step 1: Checking for correct number of parameters
#--------------------------------------------------------------

# Getting command line arguments
args=commandArgs(trailingOnly = TRUE)
if(length(args)!=3){
  cat("\nError !!! Wrong number of parameters") 
  cat("\nUsages: $rscript modelFile.R <dataFileName.csv> <trainingPercentage> <numberOfIterations>")
  cat("\nExample: $rscript modelFile.R dataFileName.csv 50 20\n") 
  q()  
}


#--------------------------------------------------------------
# Step 2: Include Library
#--------------------------------------------------------------
#install.packages('randomForest')
suppressMessages(library(randomForest))
library(hmeasure)

cat("\n\nRunning randomForest Model.....")
startTime = proc.time()[3]


#--------------------------------------------------------------
# Step 3: Variable Declaration
#--------------------------------------------------------------
# 3.1: General Parameters
modelName = "randomForest"
inputDataFileName = args[1]     # Data FileName
#inputDataFileName='binaryDataSet.csv'
training = as.numeric(args[2])  # Training Percentage; Testing = 100 - Training
#training=50

# 3.2: Random Algorithm  Parameters
algoName    = "Random"	# Algo Name
iteration   = as.numeric(args[3])     # Number of Iterations
#iteration=20
bestAccuracy= 0   # Store Best Fitness Value
bestInputs  = ""

# 3.4: Final Result fileName
finalResultFileName=paste("finalResult-",modelName,"-",algoName,".csv",sep='')


#-------------------------------------------------------------
# Step 4: Fitness Functions Definitions
#-------------------------------------------------------------

fintessFunction <-function(newInputs){
  # Formula Generation
  formula <- as.formula(paste(target, "~", paste(c(newInputs), collapse = "+")))
  # Model Building (Training)
  suppressMessages(model   <- randomForest(formula, trainDataset, ntree=500,mtry=2))
  # Prediction (Testing)
  predictedProb <- predict(model, testDataset)
  # Model Evaluation: Accuracy
  accuracy <- round(mean(actual==round(predictedProb)) *100,2)
  return (accuracy)
}

#-------------------------------------------------------------
# Step 5: Start Program
#-------------------------------------------------------------

# 5.1: Data Loading
dataset <- read.csv(inputDataFileName)      # Read the datafile

# 5.2: Count total number of observations/rows.
totalDataset <- nrow(dataset)

# 5.3: Choose Target variable
target  <- names(dataset)[1]   # i.e. RMSD

# 5.4: Choose inputs Variables
inputs <- setdiff(names(dataset),target)

# 5.5: Select Training Data Set
trainDataset <- dataset[1:(totalDataset * training/100),]

# 5.6: Select Testing Data Set
testDataset <- dataset[(totalDataset * training/100):totalDataset,]


# 5.8: Model Building (Training), Prediction (Testing) & Result generation iteratevely

write.table(data.frame(A ="Run", B="Accuracy",C="Features"), 
            file=finalResultFileName,append = FALSE, row.names=FALSE,
            col.names = FALSE,quote=FALSE,sep=',')

# Extracting Actual
actual <- as.double(unlist(testDataset[target]))

for (i in 1:iteration){
  # Feature Selection randomly
  n=round(runif(1,min=2,max=length(inputs)))
  features <- sample(inputs, n)
  accuracy<-fintessFunction(features)
  if(accuracy > bestAccuracy){
    bestFeatures = features
    bestAccuracy = accuracy
  }
  cat("\n",i, "\t Best Accuracy -> ", bestAccuracy)
  write.table(data.frame(i, bestAccuracy, paste(bestFeatures, collapse=',' )),
              file=finalResultFileName,append =TRUE, row.names=FALSE,
              col.names = FALSE,quote=FALSE,sep=',')
}


# 5.10: Model Building (Training) using best parameters
# Formula Generation
formula <- as.formula(paste(target, "~", paste(c(bestFeatures), collapse = "+")))

suppressMessages(model   <- randomForest(formula, trainDataset, ntree=500,mtry=2))

# 5.11: Prediction (Testing) through best model
predictedProb <- predict(model, testDataset)

# 5.12: Model Evaluation: Accuracy
accuracy <- round(mean(actual==round(predictedProb)) *100,2)

# 5.13: Total Time
totalTime = proc.time()[3] - startTime

# 5.14: Result Merging
result <- data.frame(modelName,accuracy,totalTime)[1:1,]

# 5.15: Writing to file
# Writing to file (evaluation result)
write.csv(result, file=paste(modelName,"-Evaluation-Result.csv",sep=''), 
          row.names=FALSE)

# Writing to file (Actual and Predicted)
write.csv(data.frame(actual,round(predictedProb)), file=paste(modelName,"-ActualPredicted-Result.csv",sep=''), 
          row.names=FALSE)


# 5.16: Saving the Model
save.image(file=paste(modelName,"-Model.RData",sep=''))


# 5.17: Finished
cat("\n..Finished.....")
cat("\nBest Fitness:", bestAccuracy)
cat("\nResult is saved in", finalResultFileName)
cat("\nTotal Time Taken: ", totalTime, " sec\n")




#--------------------------------------------------------------
#                           END 
#--------------------------------------------------------------




