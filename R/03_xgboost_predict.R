
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

# ---- Load data ----

# Load data
# Use R/load_data.R to create these datasets
all_train_df <- readRDS('data/all_train_df.rds')
inference_df <- readRDS('data/inference_df.rds')

# Create train test split by time
train <- all_train_df %>% filter(season <= 18) %>% select(improvement, overall:ncol(.), -nationality, -player_positions, -pos, -team_position, -work_rate_att, -work_rate_def)
test <- all_train_df %>% filter(season > 18) %>% bind_rows(inference_df)

# Load model
xgb_model <- xgb.load('models/xgb.model')

# Convert test to xgb object
transfers_xgb <- test %>%
  select(names(train)) %>%
  select(-improvement) %>%
  as.matrix() %>%
  xgb.DMatrix()

# Create predictions
preds <- predict(xgb_model, transfers_xgb)

# Add to data
test <- test %>%
  mutate(pred_improvement = preds) %>%
  select(1:improvement, pred_improvement, everything())

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
get_best_transfers <- function(selected_club, min_rating, max_val = Inf, train, test, xgb_model, type = "best") {
  
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
  # create xgb object, predict. Compare with original predictions
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
    mutate(pred_transfer_improvement = transfer_preds,
           overall_pred_transfer = pred_transfer_improvement + overall,
           value_eur = value_eur/1e6) %>%
    filter(club != selected_club) %>%
    mutate_at(vars(pred_improvement, overall_pred_transfer, pred_transfer_improvement), round, digits = 2)
  
  # Either get players to sign or avoid
  if (type == "best") {
    
    # Find players who would do better in this club than if they remained where they are
    best_transfers <- transfers_final %>%
      filter(pred_transfer_improvement > pred_improvement,
             overall_pred_transfer > min_rating,
             value_eur <= max_val) %>%
      select(short_name, age, value_eur, club, pos, player_positions, potential, overall, overall_pred_transfer, pred_improvement, pred_transfer_improvement) %>%
      arrange(desc(pred_transfer_improvement))
    
  } else if (type == "worst") {
    
    # Find players who would do worse in this club than if they remained where they are
    best_transfers <- transfers_final %>%
      filter(pred_transfer_improvement < pred_improvement,
             value_eur <= max_val) %>%
      select(short_name, age, value_eur, club, pos, player_positions, potential, overall, overall_pred_transfer, pred_improvement, pred_transfer_improvement) %>%
      arrange(pred_transfer_improvement)
    
  }

  
}

# Have a look!
best_transfers <- get_best_transfers(selected_club = "Liverpool", min_rating = 78, max_val = 90, train, test, xgb_model) %>%
  arrange(desc(overall_pred_transfer)) %>% 
  filter(pred_transfer_improvement > 0) %>%
  slice(1:100)

# Save
write.csv(best_transfers, file = "outputs/liverpool_targets.csv", row.names = FALSE)

# Ones to avoid
worst_transfers <- get_best_transfers(selected_club = "Liverpool", min_rating = 80, max_val = 90, train, test, xgb_model, type = "worst")

# Ones to avoid, who might otherwise look good
worst_transfers %>%
  filter(overall > 80,
         age < 30)

# With some filters
best_transfers %>%
  filter(pos == "CB",
         pred_transfer_improvement > 0)

best_transfers %>%
  filter(pos == "WING",
         pred_transfer_improvement > 0)

# ---- All premier league ----

prem_clubs <- c("Liverpool", "Manchester City", "Chelsea", "Leicester City", "Manchester United",
                "Wolverhampton Wanderers", "Arsenal", "Tottenham Hotspur", "Sheffield United",
                "Burnley", "Everton", "Crystal Palace", "Newcastle United", "Watford", "West Ham United",
                "Southampton", "Aston Villa", "Brighton & Hove Albion", "Bournemouth", "Norwich City")

# Get transfer targets for all
all_targets <- lapply(prem_clubs, function(club_name) {
  
  # Progress
  message(club_name)
  
  # Get value of current most expensive player
  maxval <- 1.2 * max(test$value_eur[test$club == club_name], na.rm = TRUE)/1e6
  
  # Get rating of current worst first team player and take a couple off
  minrating <- min(test$overall[test$club == club_name & test$first_11==TRUE], na.rm = TRUE) - 2
  
  # Get targets
  get_best_transfers(selected_club = club_name, min_rating = minrating, max_val = maxval, train, test, xgb_model) %>%
    arrange(desc(overall_pred_transfer)) %>% 
    filter(pred_transfer_improvement > 0) %>%
    slice(1:50) %>%
    mutate(new_club = club_name) %>%
    select(new_club, everything())
  
})

# Bind rows and save
all_targets %>%
  bind_rows %>%
  write.csv('outputs/all_transfer_targets.csv')

# ---- Case studies ----

# Load cross validation results
xgb_cv_res <- readRDS('models/xgb_cv_results.rds')

# CV predictions
cv_df  <- all_train_df %>% filter(season <= 18) %>% 
  mutate(pred_improvement = xgb_cv_res$pred) %>%
  select(1:improvement, pred_improvement, everything())

# Find players who:
# - Changed clubs
# - looked like they would be good
# - but turned out not to be
# - and were correctly predicted
bad_examples <- cv_df %>%
  group_by(sofifa_id) %>%
  arrange(season) %>%
  mutate(last_club = lag(club)) %>%
  filter(club != lag(club)) %>%
  filter(age < 27,
         improvement < 0,
         pred_improvement < 0,
         potential > overall) %>%
  ungroup %>%
  select(season, short_name, last_club, club, age, player_positions, overall, potential, improvement, pred_improvement) %>%
  arrange(desc(overall))

# Compare with players who:
# - Changed clubs
# - looked like they would be good
# - And did improve
# - and were correctly predicted
good_expamples <- cv_df %>%
  group_by(sofifa_id) %>%
  arrange(season) %>%
  mutate(last_club = lag(club)) %>%
  filter(club != lag(club)) %>%
  filter(age < 27,
         improvement > 0,
         pred_improvement > 0,
         potential > overall) %>%
  ungroup %>%
  select(season, short_name, last_club, club, age, player_positions, overall, potential, improvement, pred_improvement) %>%
  arrange(desc(overall))

# Compare
View(good_expamples)
View(bad_examples)
