##use option command enter to run on server
library(tidymodels)
library(vroom)
library(embed) 
amazon_train <- vroom("348/AmazonEmployeeAccess/train.csv")
amazon_test <- vroom("348/AmazonEmployeeAccess/test.csv")

amazon_train$ACTION = as.factor(amazon_train$ACTION)

my_recipe <- recipe(ACTION ~ ., data=amazon_train) %>%
step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .01) %>% 
  step_dummy(all_nominal_predictors())

dough <- prep(my_recipe)
baked <- bake(dough, new_data = amazon_train)

################################## Homework 2

logRegModel <- logistic_reg() %>% #Type of model
  set_engine("glm")

## Put into a workflow here
logwf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(logRegModel) %>% 
  fit(data = amazon_train)
## Make predictions
amazon_predictions <- predict(logwf,
                              new_data=amazon_test,
                              type="prob") # "class" or "prob"

kaggle_submission <- amazon_predictions %>%
  bind_cols(., amazon_test) %>%
  rename(ACTION = .pred_1) %>% 
  select(id, ACTION)

  
## Write out the file
vroom_write(x=kaggle_submission, file="./amazonpreds.csv", delim=",")

######################## penalized logistic regression

my_mod <- logistic_reg(mixture=.8, penalty=.001) %>% #Type of model
  set_engine("glmnet")

amazon_workflow <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(my_mod)

## Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 3) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(amazon_train, v = 3, repeats=1)

## Run the CV
CV_results <- amazon_workflow %>%
tune_grid(resamples=folds,
          grid=tuning_grid,
          metrics=metric_set(roc_auc)) #Or leave metrics NULL

bestTune <- CV_results %>%
select_best()

## Finalize the Workflow & fit it
final_wf <-
amazon_workflow %>%
finalize_workflow(bestTune) %>%
fit(data=amazon_train)

## Predict
final_wf %>%
predict(new_data = amazon_train, type=)
amazon_pred <- predict(final_wf,
                              new_data=amazon_test,
                              type="prob") # "class" or "prob"

kaggle_submission <- amazon_pred %>%
  bind_cols(., amazon_test) %>%
  rename(ACTION = .pred_1) %>% 
  select(id, ACTION)


## Write out the file
vroom_write(x=kaggle_submission, file="./PenLog.csv", delim=",")

