rm(list = ls())
source("_functions.R")
f_install_and_load(c("readr", "dplyr", "mice", "ggplot2", "Matrix"))
library("xgboost")
library("verification")

# ---- Load data ----

train_raw <- read_csv("../data_raw/train.csv")
test_raw <- read_csv("../data_raw/test.csv")

md.pattern(train_raw)
md.pattern(test_raw)

# ---- Feature engineering ----

train_data <- train_raw %>%
  select(-ACTION) %>%
  mutate_all(as.factor) %>%
  sparse.model.matrix(~ . - 1, data = .)
train_label <- train_raw$ACTION

test_data <- test_raw %>%
  select(-id) %>%
  mutate_all(as.factor) %>% sparse.model.matrix(~ . - 1, data = .)

# ---- Train ----

dtrain <- xgb.DMatrix(train_data, label = train_label)

cv_bst <- xgb.cv(data = dtrain, nrounds = 3, nthread = 2, nfold = 5,
                 metrics = list("rmse", "auc"), max_dept = 3, eta = 1, objective = "binary:logistic")
m_bst <- xgboost(data = dtrain, nrounds = 10, objective = "binary:logistic")

# ---- Model checking ----

pred_train <- predict(m_bst, train_data)

mean(as.numeric(pred_train > 0.5) == train_label)


# ---- Predict ----

pred_test <- cbind.data.frame(id = test_raw$id,
                         ACTION = predict(m_bst, test_data))
write_csv(pred_test, path = "../result/submission1.csv")


# ---- Intepret ----

importance_matrix <- xgb.importance(colnames(train_data),
                                    model = m_bst)
importance_matrix
xgb.plot.importance(importance_matrix)
