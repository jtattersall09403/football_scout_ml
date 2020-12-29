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
library(ranger)
library(plotly)
library(viridis)
library(htmlwidgets)

# Load data
all_train_df <- readRDS('data/all_train_df.rds')
inference_df <- readRDS('data/inference_df.rds')

# ---- Train baseline model ----

# Create train test split by time
train <- all_train_df %>% filter(season <= 18) %>% select(improvement, overall:ncol(.), -nationality, -player_positions, -team_position, -work_rate_att, -work_rate_def)
test <- all_train_df %>% filter(season > 18) %>% bind_rows(inference_df)

# Train model
tune_res <- train(improvement ~ .,
                  data = train,
                  method = "ranger")

# Save
saveRDS(tune_res, 'models/randomforest.rds')

# Load
tune_res <- readRDS('models/randomforest.rds')

# Results
tune_res$results %>% arrange(RMSE)

# ---- Evaluate ----

# Predict on test set
preds <- predict(tune_res, newdata = select(test, names(train)))

# Add to data
test <- test %>%
  mutate(pred_improvement = preds,
         residual = improvement - pred_improvement,
         pos = case_when(player_positions == "GK" ~ "GK",
                         str_detect(player_positions, "M") ~ "MID",
                         str_detect(player_positions, "B") ~ "DEF",
                         TRUE ~ "FWD")) %>%
  select(1:improvement, pred_improvement, residual, everything())

# Metrics
test %>%
  filter(season == 19) %>%
  summarise(rmse = RMSE(pred_improvement, improvement),
            mae = MAE(pred_improvement, improvement))

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
                            "<br>Value: ", value_eur,
                            "Current rating: ", round(overall, 1), "<br>",
                            "Predicted rating this season: ", round(overall_next_pred, 1), "<br>",
                            "Improvement: ", impovement_formatted)) %>%
  plot_ly(x = ~ overall,
          y = ~pred_improvement, 
          color = ~pos,
          opacity = 0.7,
          text = ~hovertext,
          hoverinfo = text,
          type = "scatter",
          mode = "markers")

saveWidget(fig, "p1.html", selfcontained = F, libdir = "lib")

# ---- tailored recommendations ----

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
