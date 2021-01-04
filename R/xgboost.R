# Random forest
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

# ---- New predictions ----

# Get predictions for who will improve from 2020 to 2021
test %>%
  filter(season == 20) %>%
  mutate(overall_next_pred = overall + pred_improvement) %>%
  ggplot(aes(x=overall_next_pred, y=pred_improvement, colour=pos, text = short_name)) +
  theme_minimal() +
  theme(panel.grid = element_blank()) +
  geom_point() +
  geom_vline(xintercept = mean(test$overall)) +
  geom_hline(yintercept = 0) +
  scale_colour_viridis(discrete = TRUE)

# Plotlify
fig <- test %>%
  filter(season == 20) %>%
  mutate(overall_next_pred = overall + pred_improvement,
         impovement_formatted = ifelse(pred_improvement > 0, 
                                       paste0("+", round(pred_improvement, 1)),
                                       round(pred_improvement, 1)),
         hovertext = paste0("Name: ", short_name, "<br>Age: ", age,
                            "<br>Club: ", club, "<br>Position: ", player_positions, "<br>",
                            "<br>Value: ", as.character(prettyNum(value_eur, big.mark = ",", big.interval = 3)), "<br>",
                            "Current rating: ", round(overall, 1), "<br>",
                            "Predicted rating this season: ", round(overall_next_pred, 1), "<br>",
                            "Improvement: ", impovement_formatted)) %>%
  plot_ly(x = ~ overall,
          y = ~pred_improvement, 
          color = ~pos,
          colors = viridis_pal()(length(unique(test$pos))),
          opacity = 0.7,
          text = ~hovertext,
          hoverinfo = text,
          type = "scatter",
          mode = "markers")

saveWidget(fig, "predicted_improvement.html", selfcontained = F, libdir = "lib")

# ---- Transfers ----

# Function to get best transfer targets
get_best_transfers <- function(selected_club, min_rating, max_val = Inf, train, test, xgb_model) {
  
  # Get club attributes
  club_attrs <- test %>%
    filter(club == selected_club, season == max(season)) %>%
    mutate(mean_club_overall = mean(overall, na.rm = TRUE),
           mean_club_potential = mean(potential, na.rm = TRUE),
           mean_starting_overall = mean(overall[first_11], na.rm = TRUE),
           mean_attack_overall = mean(overall[first_11 & pos %in% c("ST", "WING", "ATT_MID")], na.rm = TRUE),
           mean_mid_overall = mean(overall[first_11 & pos == "MID"], na.rm = TRUE),
           mean_def_overall = mean(overall[first_11 & pos %in% c("CB", "FB")], na.rm = TRUE),
           mean_gk_overall = mean(overall[first_11 & pos == "GK"], na.rm = TRUE)) %>%
    group_by(pos) %>%
    arrange(desc(overall)) %>%
    mutate(mean_overall_pos = mean(overall, na.rm = TRUE), # Average of other players in same position
           mean_potential_pos = mean(potential, na.rm = TRUE),
           mean_starting_overall_pos = mean(overall[first_11], na.rm = TRUE),
           max_overall_pos_others = max(overall, na.rm = TRUE) # Maximum rating of the *other* players available in this position
    ) %>%
    ungroup %>%
    select(club, pos, mean_club_overall:max_overall_pos_others) %>%
    distinct()
  
  # Club players
  club_players <- test %>% filter(club == selected_club, season == max(season))
  
  # For calculating what each player's rank would be by position in the new club,
  # create a matrix of players. We reverse the 'overall' rating as the function sorts ascending
  player_matrix <- club_players %>%
    select(pos, overall) %>%
    group_by(pos) %>%
    mutate(player_rank = row_number(desc(overall))) %>%
    ungroup %>%
    complete(pos, player_rank) %>%
    arrange(pos, player_rank) %>%
    mutate(overall = 100 - ifelse(is.na(overall), 0, overall)) %>%
    pivot_wider(id_cols = "pos", names_from = "player_rank", values_from = "overall")
  
  # Get player attributes, mutate club attributes equal to those of the selected club,
  # create xgbo object, predict. compare with original predictions
  transfer_attrs <- test %>%
    filter(season == max(season)) %>%
    mutate(club = selected_club) %>%
    select(1:nationalityOther) %>%
    left_join(club_attrs, by = c("club", "pos")) %>%
    left_join(player_matrix, by = "pos") %>%
    mutate(overall_rev = 100 - overall)
  
  # Get the ranks
  pos_ranks <- rowRanks(as.matrix(select(transfer_attrs, overall_rev, names(select(player_matrix, -pos)))),
                        ties.method = "first")[,1]
  
  # Add to data
  transfer_attrs <- transfer_attrs %>%
    mutate(position_rank = pos_ranks) %>%
    select(names(train)) %>%
    select(-improvement)
  
  # Convert to xgb object
  transfers_xgb <- transfer_attrs %>%
    as.matrix() %>%
    xgb.DMatrix()
  
  # Create predictions
  transfer_preds <- predict(xgb_model, transfers_xgb)
  
  # Create final data for comparison
  transfers_final <- test %>%
    filter(season == max(season)) %>%
    mutate(pred_transfer_improvement = transfer_preds) %>%
    select(1:pred_improvement, pred_transfer_improvement, everything())
  
  # Find players who would do better in this club than if they remained where they are
  best_transfers <- transfers_final %>%
    mutate(overall_pred_transfer = pred_transfer_improvement + overall,
           value_eur = value_eur/1e6) %>%
    filter(pred_transfer_improvement > pred_improvement,
           overall_pred_transfer > min_rating,
           value_eur <= max_val) %>%
    select(short_name, age, value_eur, club, pos, potential, overall, overall_pred_transfer, pred_improvement, pred_transfer_improvement) %>%
    arrange(desc(pred_transfer_improvement))
  
}

# Have a look!
best_transfers <- get_best_transfers(selected_club = "Leeds United", min_rating = 60, max_val = 15, train, test, xgb_model)
best_transfers %>% arrange(desc(overall_pred_transfer)) %>% filter(pred_transfer_improvement > 0) %>% View

# With some filters
best_transfers %>%
  filter(pos == "CB",
         pred_transfer_improvement > 0)

best_transfers %>%
  filter(pos == "WING",
         pred_transfer_improvement > 0)

# ---- Tailored recommendations ----

# Choose parameters
selected_club <- "Liverpool"
budget <- 100e6
max_age <- 40

# Get players
test %>%
  filter(season == 20,
         club == selected_club) %>%
  select(short_name, pos, player_positions, overall) %>%
  arrange(pos, desc(overall))

# Get stats
team_stats <- test %>%
  filter(season == 20,
         club == selected_club) %>%
  group_by(pos) %>%
  summarise(mean_overall = mean(overall),
            best_overall = max(overall),
            mean_potential = mean(potential),
            best_potential = max(potential))

# Show stats
team_stats

# Filter
test %>%
  inner_join(team_stats, by = "pos") %>%
  filter(season == 20,
         overall <= best_overall + 1, # Can't afford players who are way better than your best already
         overall + pred_improvement >= mean_overall,
         value_eur <= budget, # Want players who will add value
         age <= max_age,
         club != selected_club) %>% 
arrange(desc(overall + pred_improvement)) %>%
select(short_name, 
       age,
       value_eur,
       club,
       pos,
       player_positions,
       overall,
       potential,
       pred_improvement,
       best_overall) %>% View
