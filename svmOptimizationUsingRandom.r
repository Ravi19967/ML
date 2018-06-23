
###############################################################
#                                                             #
# Optimize SVM using Random Approach, Part II                 #
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
  cat("\nExample: $rscript modelFile.R dataFileName.csv 20 20\n") 
  q()  
}


#--------------------------------------------------------------
# Step 2: Include Library
#--------------------------------------------------------------
#install.packages('kernlab')
library(kernlab)

cat("\n\nRunning SVM Model.....")
startTime = proc.time()[3]


#--------------------------------------------------------------
# Step 3: Variable Declaration
#--------------------------------------------------------------
# 3.1: General Parameters
modelName = "SVM"
inputDataFileName = args[1]     # Data FileName
training = as.numeric(args[2])  # Training Percentage; Testing = 100 - Training


# 3.2: Random Algorithm  Parameters
algoName    = "Random"	# Algo Name
iteration   = as.numeric(args[3])       # Number of Iterations
bestAccuracy= 0   # Store Best Fitness Value

# 3.3: SVM Parameters
kernelList  = c('rbfdot','polydot','vanilladot','tanhdot','laplacedot','besseldot','anovadot')# 'splinedot' ,'matrix', 'stringdot' doesn't work
bestKernel  = ""
bestNu      = 0
bestEpsilon = 0

# 3.4: Final Result fileName
finalResultFileName=paste("finalResult-",modelName,"-",algoName,".csv",sep='')


#-------------------------------------------------------------
# Step 4: Fitness Functions Definitions
#-------------------------------------------------------------

fintessFunction <-function(k,n,e){
  #k="Kernel" , n="Nu", e="Epsilon" 
  
  # Model Building (Training)
  model  <- ksvm(formula, trainDataset, kernel=k, nu = n, epsilon = e, kpar=list())
  
  # Prediction (Testing)
  predicted <- round(predict(model, testDataset))
  
  # Model Evaluation: Accuracy
  accuracy <- round(mean(actual==predicted) *100,2)
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
trainDataset <- dataset[1:(totalDataset * training/100),c(inputs, target)]

# 5.6: Select Testing Data Set
testDataset <- dataset[(totalDataset * training/100):totalDataset,c(inputs, target)]

# 5.7: Formula Generation
formula <- as.formula(paste(target, "~", paste(c(inputs), collapse = "+")))

# 5.8: Model Building (Training), Prediction (Testing) & Result generation iteratevely

write.table(data.frame(A ="Run", B="Accuracy",C="Kernel" , D="Nu", E="Epsilon"), 
            file=finalResultFileName,append = FALSE, row.names=FALSE,
            col.names = FALSE,quote=FALSE,sep=',')

# Extracting Actual
actual <- as.double(unlist(testDataset[target]))

for (i in 1:iteration){
  
  # Choosing paramenters randomly
  k=sample(kernelList,1)
  n=runif(1)
  e=runif(1)
  
  accuracy<-fintessFunction(k,n,e)
  
  if(accuracy > bestAccuracy){
    bestKernel = k
    bestNu = n
    bestEpsilon = e
    bestAccuracy = accuracy
  }
  
  cat("\n",i, "\t Best Accuracy -> ", bestAccuracy)
  write.table(data.frame(i, bestAccuracy, bestKernel , bestNu, bestEpsilon),
              file=finalResultFileName,append =TRUE, row.names=FALSE,
              col.names = FALSE,quote=FALSE,sep=',')
}


# 5.10: Model Building (Training) using best parameters
model  <- ksvm(formula, trainDataset, kernel= bestKernel, nu = bestNu, 
               epsilon = bestEpsilon,kpar=list())


# 5.11: Prediction (Testing) through best model
predicted <- round(predict(model, testDataset))

# 5.12: Model Evaluation: Accuracy
accuracy <- round(mean(actual==predicted) *100,2)

# 5.13: Total Time
totalTime = proc.time()[3] - startTime

# 5.14: Result Merging
result <- data.frame(modelName,accuracy,totalTime)[1:1,]

# 5.15: Writing to file
# Writing to file (evaluation result)
write.csv(result, file=paste(modelName,"-Evaluation-Result.csv",sep=''), 
          row.names=FALSE)

# Writing to file (Actual and Predicted)
write.csv(data.frame(actual,predicted), file=paste(modelName,"-ActualPredicted-Result.csv",sep=''), 
          row.names=FALSE)


# 5.16: Saving the Model
save.image(file=paste(modelName,"-Model.RData",sep=''))


# 5.17: Finished
cat("\n..Finished.....")
cat("\nBest Chromosome: (", bestKernel,bestNu,bestEpsilon, ")\tFitness:", bestAccuracy)
cat("\nResult is saved in", finalResultFileName)
cat("\nTotal Time Taken: ", totalTime, " sec\n")




#--------------------------------------------------------------
#                           END 
#--------------------------------------------------------------




