# Load necessary libraries
library(dplyr)
library(caret)
library(ggplot2)
library(glmnet)
library(earth)
library(kernlab)
library(nnet)
library(gridExtra)

# Load the dataset (replace 'file_path' with the actual file path)
data <- read.csv('ai_job_market_insights.csv')

# Preprocessing: Handle missing values
data <- na.omit(data)  # Remove rows with missing values

# Print dimensions before dummy variable creation
cat("Dimensions before dummy variable creation:", dim(data), "\n")

# Apply Box-Cox Transformation to target variable (Salary_USD)
# Ensure all values are positive for Box-Cox transformation
if (any(data$Salary_USD <= 0)) {
  data$Salary_USD <- data$Salary_USD + abs(min(data$Salary_USD)) + 1
}
boxcox_transform <- BoxCoxTrans(data$Salary_USD)
data$Salary_BoxCox <- predict(boxcox_transform, data$Salary_USD)

# Print dimensions before dummy variable creation
cat("Dimensions before dummy variable creation:", dim(data), "\n")

# Convert categorical variables to factors and apply dummy encoding
data <- data %>%
  mutate(across(where(is.character), as.factor)) %>%
  model.matrix(~ . - 1, data = .) %>%
  as.data.frame()

# Print dimensions after dummy variable creation
cat("Dimensions after dummy variable creation:", dim(data), "\n")

# Near Zero Variance (NZV) Check
nzv <- nearZeroVar(data, saveMetrics = TRUE)
cat("Number of NZV predictors:", sum(nzv$nzv), "\n")
data <- data[, !nzv$nzv]  # Remove NZV predictors

# Split the data into training (80%) and testing (20%) sets
set.seed(123)
trainIndex <- createDataPartition(data$Salary_BoxCox, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Preprocessing: Centering and scaling predictors
preProc <- preProcess(trainData[, -which(names(trainData) %in% c("Salary_USD", "Salary_BoxCox"))], method = c("center", "scale"))
trainData_scaled <- predict(preProc, trainData)
testData_scaled <- predict(preProc, testData)

# Define training control for cross-validation
train_control <- trainControl(method = "cv", number = 10)

# Linear Models
# 1. Linear Regression
set.seed(123)
model_lm <- train(Salary_BoxCox ~ . - Salary_USD, data = trainData_scaled, method = "lm", trControl = train_control)
cat("Linear Regression Results:\n")
print(model_lm)

# 2. Ridge Regression
set.seed(123)
model_ridge <- train(Salary_BoxCox ~ . - Salary_USD, data = trainData_scaled, method = "ridge", trControl = train_control, tuneLength = 20)
cat("Ridge Regression Results:\n")
print(model_ridge)
plot(model_ridge)

# 3. Lasso Regression
set.seed(123)
model_lasso <- train(Salary_BoxCox ~ . - Salary_USD, data = trainData_scaled, method = "lasso", trControl = train_control, tuneLength = 20)
cat("Lasso Regression Results:\n")
print(model_lasso)
plot(model_lasso)

# 4. Elastic Net
set.seed(123)
model_enet <- train(Salary_BoxCox ~ . - Salary_USD, data = trainData_scaled, method = "glmnet", trControl = train_control, tuneLength = 20)
cat("Elastic Net Results:\n")
print(model_enet)
plot(model_enet)

# Non-Linear Models
# 1. Neural Networks
set.seed(123)
model_nn <- train(Salary_BoxCox ~ . - Salary_USD, data = trainData_scaled, method = "nnet", trControl = train_control, tuneLength = 20, linout = TRUE, trace = FALSE)
cat("Neural Networks Results:\n")
print(model_nn)
plot(model_nn)

# 2. Multivariate Adaptive Regression Splines (MARS)
set.seed(123)
model_mars <- train(Salary_BoxCox ~ . - Salary_USD, data = trainData_scaled, method = "earth", trControl = train_control, tuneLength = 20)
cat("MARS Results:\n")
print(model_mars)
plot(model_mars)

# 3. Support Vector Machines (SVM)
set.seed(123)
model_svm <- train(Salary_BoxCox ~ . - Salary_USD, data = trainData_scaled, method = "svmRadial", trControl = train_control, tuneLength = 20)
cat("SVM Results:\n")
print(model_svm)
plot(model_svm)

# 4. K-Nearest Neighbors (KNN)
set.seed(123)
model_knn <- train(Salary_BoxCox ~ . - Salary_USD, data = trainData_scaled, method = "knn", trControl = train_control, tuneLength = 20)
cat("KNN Results:\n")
print(model_knn)
plot(model_knn)

# Compare Model Performance
results <- resamples(list(
  Linear_Regression = model_lm,
  Ridge = model_ridge,
  Lasso = model_lasso,
  Elastic_Net = model_enet,
  Neural_Network = model_nn,
  MARS = model_mars,
  SVM = model_svm,
  KNN = model_knn
))
cat("Model Comparison:\n")
summary(results)

# Plot Model Comparison
bwplot(results, metric = "RMSE", main = "Model Comparison (RMSE)")

# Additional Normalized RMSE Calculation
normalized_rmse <- model_lm$results$RMSE / mean(data$Salary_USD)
cat("Normalized RMSE for Linear Regression:", normalized_rmse, "\n")
