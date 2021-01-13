
# Data downloaded from kaggle at https://www.kaggle.com/stefanoleone992/fifa-20-complete-player-dataset
library(data.table)
library(dplyr)
library(tidyr)
library(stringr)
library(forcats)
library(tidygraph)
library(ggraph)
library(caret)
library(xgboost)
library(plotly)
library(viridis)
library(htmlwidgets)
library(matrixStats)

# Load data
# Use R/load_data.R to create these datasets
all_train_df <- readRDS('data/all_train_df.rds')
inference_df <- readRDS('data/inference_df.rds')

# ---- Train baseline model ----

# Create train test split by time
train <- all_train_df %>% filter(season <= 18) %>% select(improvement, overall:ncol(.), -nationality, -player_positions, -pos, -team_position, -work_rate_att, -work_rate_def)
test <- all_train_df %>% filter(season > 18) %>% bind_rows(inference_df)

# Train baseline linear model
base_model <- lm(improvement ~ ., data = train[complete.cases(train),])

# In-sample results
summary(base_model)

# Predictive accuracy
preds <- predict(base_model, newdata = test, na.action = "na.pass")

# Metrics df
eval_df <- tibble(obs = test$improvement, pred = preds)

# Display metrics
lm_res <- eval_df %>%
  mutate(pred_category = ifelse(pred > 0, "improve", "decline/same"),
         obs_category = ifelse(obs > 0, "improve", "decline/same")) %>%
  filter(!is.na(pred), !is.na(obs)) %>%
  summarise(rmse = RMSE(pred, obs),
            r_square = R2(pred, obs),
            sens = sensitivity(factor(pred_category), reference = factor(obs_category)),
            spec = specificity(factor(pred_category), reference = factor(obs_category)),
            pos_pred_rate = mean(obs_category[pred_category == "improve"] == "improve"),
            neg_pred_rate = mean(obs_category[pred_category != "improve"] != "improve"),
            num = n()) %>%
  mutate(model = "lm") %>%
  select(model, everything())

# ---- Xgboost ----

# Prepare train object
train_xgb <- train %>%
  select(-improvement) %>%
  as.matrix(na.action = na.pass) %>%
  xgb.DMatrix(label = train$improvement)

# Prepare test object
test_xgb <- test %>%
  select(names(train)) %>%
  select(-improvement) %>%
  as.matrix(na.action = na.pass) %>%
  xgb.DMatrix(label = test$improvement)

# Set parameters
xgb_params <- list(
  eta = 0.1,
  max_depth = 4,
  min_child_weight = 5,
  colsample = 0.7
)

# Cross validate with default parameters
xgb_cv_res <- xgb.cv(
  objective = 'reg:linear',
  params = xgb_params,
  data = train_xgb,
  nrounds = 500,
  nfold = 5,
  prediction = TRUE, 
  early_stopping_rounds = 50, 
  print_every_n = 20
)

# Plot training history
xgb_cv_res$evaluation_log %>%
  select(iter, rmse_mean=train_rmse_mean, rmse_std=train_rmse_std) %>%
  mutate(train_test = "train") %>%
  bind_rows(select(mutate(xgb_cv_res$evaluation_log, train_test = "test"),
                   iter, rmse_mean=test_rmse_mean, rmse_std=test_rmse_std, train_test)) %>%
  ggplot(aes(x=iter, y=rmse_mean, colour = train_test)) +
  geom_line() +
  geom_point() +
  geom_errorbar(aes(ymin=rmse_mean - rmse_std, ymax = rmse_mean + rmse_std)) +
  expand_limits(y = 0) +
  labs(title = "Training history")

# Train final model
xgb_model <- xgb.train(
  data = train_xgb,
  params = xgb_params, 
  nrounds = xgb_cv_res$best_iteration
)

# Save
saveRDS(xgb_cv_res, 'models/xgb_cv_results.rds')
xgb.save(xgb_model, 'models/xgb.model')

# ---- CV evaluation ----

# Variable importance
var_imp <- xgb.importance(feature_names = names(select(train, -improvement)), model = xgb_model)

# Plot
xgb.ggplot.importance(var_imp, top_n = 20)

# Get cv predictions
xgb_eval_df <- tibble(obs = train$improvement, pred = xgb_cv_res$pred)

# Summarise
xgb_res <- xgb_eval_df  %>%
  mutate(pred_category = ifelse(pred > 0, "improve", "decline/same"),
         obs_category = ifelse(obs > 0, "improve", "decline/same")) %>%
  filter(!is.na(pred), !is.na(obs)) %>%
  summarise(rmse = RMSE(pred, obs),
            r_square = R2(pred, obs),
            sens = sensitivity(factor(pred_category), reference = factor(obs_category)),
            spec = specificity(factor(pred_category), reference = factor(obs_category)),
            pos_pred_rate = mean(obs_category[pred_category == "improve"] == "improve"),
            neg_pred_rate = mean(obs_category[pred_category != "improve"] != "improve"),
            num = n()) %>%
  mutate(model = "xgboost") %>%
  select(model, everything())

# Compare with lm
bind_rows(lm_res, xgb_res)

# Add predictions to data
cv_df <- all_train_df %>%
  filter(season <= 18) %>%
  mutate(pred_improvement = xgb_cv_res$pred) %>%
  select(1:improvement, pred_improvement, everything())

# Create a csv file with some examples
examples <- cv_df %>%
  group_by(sofifa_id) %>%
  arrange(season) %>%
  mutate(last_club = lag(club)) %>%
  ungroup %>%
  select(season, short_name, last_club, club, age, player_positions, potential, overall, overall_next, improvement, pred_improvement) %>%
  arrange(desc(overall)) %>%
  group_by(improvement > 0) %>%
  filter((short_name == "A. SchÃ¼rrle" & season == 17) |
         (short_name == "M. Salah" & season == 18) |
           row_number() %in% sample(1:nrow(.), 9)) %>%
  ungroup %>%
  arrange(desc(improvement))

# Save
write.csv(examples, 'data/example_cv_predictions.csv', row.names = FALSE)

# Are the top improvers predicted to get better?
cv_df %>% select(season, short_name, age, pos, overall, overall_next, improvement, pred_improvement) %>% arrange(desc(improvement))

# Do the top predicted improvers actually improve?
cv_df %>% select(season, short_name, age, pos, overall, overall_next, improvement, pred_improvement) %>% arrange(desc(pred_improvement))

# Do the players predicted to become top-class live up to their predictions?
cv_df %>% 
  filter(overall + pred_improvement > 80,
         overall < 80) %>%
  select(season, short_name, age, pos, overall, overall_next, improvement, pred_improvement) %>% arrange(desc(pred_improvement))

# Summarise this
cv_df %>% 
  filter(overall + pred_improvement > 80,
         overall < 80) %>%
  select(season, short_name, age, pos, overall, overall_next, improvement, pred_improvement) %>% arrange(desc(pred_improvement)) %>%
  summarise(improved = mean(improvement > 0),
            became_top_class = mean(overall_next > 80))

# Plot it
cv_df %>% 
  mutate(pred_top = overall + pred_improvement > 80) %>%
  filter(overall + pred_improvement > 80 | overall_next > 80,
         overall <= 80) %>%
  ggplot(aes(x = pred_improvement, y = improvement, colour = pred_top)) +
  geom_jitter(alpha = 0.5) +
  geom_hline(yintercept = 0) +
  geom_vline(xintercept = 0)

# ---- Evaluate ----

# Predict on test set
preds <- predict(xgb_model, newdata = test_xgb)

# Add to data
test <- test %>%
  mutate(pred_improvement = preds,
         residual = improvement - pred_improvement) %>%
  select(1:improvement, pred_improvement, residual, everything())

# Metrics
test %>%
  filter(season == 19) %>%
  mutate(pred = pred_improvement, obs = improvement) %>%
  mutate(pred_category = ifelse(pred > 0, "improve", "decline/same"),
         obs_category = ifelse(obs > 0, "improve", "decline/same")) %>%
  filter(!is.na(pred), !is.na(obs)) %>%
  summarise(rmse = RMSE(pred, obs),
            r_square = R2(pred, obs),
            mae = MAE(pred, obs),
            sens = sensitivity(factor(pred_category), reference = factor(obs_category)),
            spec = specificity(factor(pred_category), reference = factor(obs_category)),
            pos_pred_rate = mean(obs_category[pred_category == "improve"] == "improve"),
            neg_pred_rate = mean(obs_category[pred_category != "improve"] != "improve"),
            num = n()) %>%
  mutate(model = "xgboost") %>%
  select(model, everything())

# Histogram of residuals
test %>%
  ggplot(aes(x=residual)) +
  geom_histogram(fill = "grey60", colour = "grey40") +
  geom_vline(xintercept = mean(test$residual, na.rm = TRUE))

# Histogram of predictions vs actuals
test %>%
  filter(season == 19) %>%
  select(improvement, pred_improvement) %>%
  gather(variable, value) %>%
  ggplot(aes(x=value, fill=variable)) +
  geom_density(colour = "grey40", alpha = 0.5) +
  geom_vline(xintercept = mean(test$residual, na.rm = TRUE))  +
  labs(title = "All players",
       subtitle = "Distributions of predictions and actuals")

# Players that changed club?
test %>%
  group_by(sofifa_id) %>%
  arrange(season) %>%
  filter(club != lead(club)) %>%
  ungroup %>%
  filter(season == 19) %>%
  summarise(rmse = RMSE(pred_improvement, improvement),
            mae = MAE(pred_improvement, improvement))

# Predictions vs actual distributions
test %>%
  group_by(sofifa_id) %>%
  arrange(season) %>%
  filter(club != lead(club)) %>%
  ungroup %>%
  filter(season == 19) %>%
  select(improvement, pred_improvement) %>%
  gather(variable, value) %>%
  ggplot(aes(x=value, fill=variable)) +
  geom_density(colour = "grey40", alpha = 0.5) +
  geom_vline(xintercept = mean(test$residual, na.rm = TRUE)) +
  labs(title = "Players who changed clubs",
       subtitle = "Distributions of predictions and actuals")

# Error is higher on these players

# Preds vs obs plot
# Colour by simpler positions
test %>%
  filter(season == 19) %>%
  ggplot(aes(x=pred_improvement, y=improvement, colour = pos)) +
  geom_jitter(alpha = 0.5) +
  geom_abline()

# Case studies
# Over estimates
test %>%
  filter(residual < -5) %>%
  select(short_name, age, player_positions, club, pred_improvement, improvement, overall, overall_next, potential)

# Under estimates
test %>%
  filter(residual > 5) %>%
  select(short_name, age, player_positions, club, pred_improvement, improvement, overall, overall_next, potential)

# Confusion matrix
conf_mat <- test %>%
  filter(season == 19) %>%
  mutate(pred_change = ifelse(pred_improvement > 0, "Up", "Down/Same"),
         actual_change = ifelse(improvement > 0, "Up", "Down/Same")) %>%
  select(pred_change, actual_change) %>%
  group_by(pred_change, actual_change) %>%
  summarise(num = n())

# Display
conf_mat %>% spread(key = actual_change, value = num)

# Plot
conf_mat %>%
  ggplot(aes(x = pred_change, y = actual_change)) +
  geom_tile(aes(fill = num, alpha = num), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", num)), vjust = 1) +
  #scale_fill_gradient(low = "blue", high = "red") +
  theme_bw() + 
  coord_equal() +
  theme(legend.position = "none")

# Sens and spec
conf_mat %>%
  ungroup %>%
  filter(actual_change == "Up") %>%
  summarise(sens = .$num[2]/sum(num))

conf_mat %>%
  ungroup %>%
  filter(actual_change == "Down/Same") %>%
  summarise(spec = .$num[1]/sum(num))

# Positive predictive value (precision)
# Number of correct positive predictions over all positive predictions
# "If we predict that you'll get better, how likely is it that you actually will?"
conf_mat %>%
  ungroup %>%
  filter(pred_change == "Up") %>%
  summarise(precision = num[2]/sum(num))