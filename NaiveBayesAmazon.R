library(tidymodels)
library(discrim)
library(vroom)
library(embed)
library(dplyr)
library(naivebayes)

amazon_train <- vroom("train.csv")
amazon_test <- vroom("test.csv")

amazon_train$ACTION = as.factor(amazon_train$ACTION)

my_recipe <- recipe(ACTION ~ ., data=amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors())


nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
set_mode("classification") %>%
set_engine("naivebayes") 


nb_wf <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(nb_model)

## Tune smoothness and Laplace
tunegrid <- grid_regular(smoothness(range = c(.1,2)),
                           Laplace(),
                           levels = 5)
CV_folds <- vfold_cv(amazon_train, v = 5, repeats=1)

CV_results <- nb_wf %>%
  tune_grid(resamples=CV_folds,
            grid=tunegrid,
            metrics=metric_set(roc_auc)) 
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

nb_wf <-
  nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazon_train)
## Predict
nb_pred <- predict(nb_wf, new_data=amazon_test, type="prob")

kaggle_submission <- nb_pred %>%
  bind_cols(., amazon_test) %>%
  rename(ACTION = .pred_1) %>% 
  select(id, ACTION)


## Write out the file
vroom_write(x=kaggle_submission, file="./NaiveBayes.csv", delim=",")
