library(tidymodels)
library(discrim)
library(vroom)
library(embed)
library(dplyr)
library(naivebayes)
library("kernlab")

amazon_train <- vroom("train.csv")
amazon_test <- vroom("test.csv")

amazon_train$ACTION = as.factor(amazon_train$ACTION)

my_recipe <- recipe(ACTION ~ ., data=amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors()) %>%
  step_pca(all_predictors(), threshold= .85) 

svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
set_engine("kernlab")

svmwf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmRadial)
## Fit or Tune Model HERE
tuning_grid <- grid_regular(cost(), rbf_sigma(),
                            levels = 5)
svm_folds <- vfold_cv(amazon_train, v = 5, repeats=1)

CV_results <- svmwf %>%
  tune_grid(resamples=svm_folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL
bestTune <- CV_results %>%
  select_best()

svmwf <-
  svmwf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazon_train)
amazon_pred <- predict(svmwf, new_data=amazon_test, type="prob")

kaggle_submission <- amazon_pred %>%
  bind_cols(., amazon_test) %>%
  rename(ACTION = .pred_1) %>% 
  select(id, ACTION)


## Write out the file
vroom_write(x=kaggle_submission, file="./RanFor.csv", delim=",")


