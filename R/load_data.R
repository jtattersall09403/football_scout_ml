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

# ---- Load data ----

# Get names of files
files <- list.files('data', full.names = TRUE)
files <- files[grepl("players", files)]

# Load all data
players_df <- lapply(files, function(filename) {
  fread(filename) %>%
    mutate(season = filename) %>%
    mutate_all(as.character) %>%
    select(season, everything())
}) %>% 
  bind_rows %>%
  filter(!nationality == club) %>% # Remove national teams
  mutate(season = str_remove(season, "data/players_"),
         season = str_remove(season, "\\.csv")) %>%
  select(
    season, sofifa_id, short_name, dob, value_eur,
    overall, potential,
    age, height_cm, weight_kg, nationality, club,
    player_positions,
    team_position,
    international_reputation,
    weak_foot,
    skill_moves,
    work_rate,
    attacking_crossing:goalkeeping_reflexes
  ) 

# Text in numeric columns needs cleaning
# Only keep columns that are populated for all versions of the game
# lump small nationalities together
players_df_2 <- players_df %>%
  mutate_at(vars(attacking_crossing:goalkeeping_reflexes),
            str_extract, pattern = '^[[:digit:]]+') %>%
  mutate_at(vars(season, value_eur, overall, potential, age, height_cm, weight_kg, 
                 international_reputation, weak_foot, skill_moves,
                 attacking_crossing:goalkeeping_reflexes),
            as.numeric) %>%
  select(-mentality_composure) %>%
  separate(work_rate, into = c("work_rate_att", "work_rate_def"), sep = "/") %>%
  mutate(player_positions = str_remove(player_positions, ", .*")) %>% # keep first preferred position only
  mutate_at(vars(nationality, 
                 team_position, 
                 player_positions,
                 work_rate_att,
                 work_rate_def), as.factor) %>%
  mutate(nationality = fct_lump_n(nationality, n = 20)) %>%
  group_by(sofifa_id) %>%
  arrange(season) %>%
  mutate(overall_next = lead(overall)) %>%
  ungroup %>%
  mutate(improvement = overall_next - overall) %>%
  select(season:dob, value_eur, club, improvement, overall_next, everything())

# Convert factors to dummies
factor_vars <- model.matrix( ~ work_rate_att + work_rate_def + player_positions + team_position + nationality - 1,
                             data = players_df_2) %>%
  as.data.frame()

# Add the one hot encoded columns on
players_df_3 <- players_df_2 %>%
  bind_cols(factor_vars)

# Separate out data that we can use for training vs inference

# Training: only players who appear in every version of the game,
# and who have a populated target variable
all_train_df <- players_df_3 %>% 
  group_by(sofifa_id) %>%
  filter(n() == length(files)) %>%
  ungroup %>%
  filter(!is.na(improvement))

# Inferece
inference_df <- players_df_3 %>% filter(season == max(season))

# Tidy up
rm(list = c("players_df", "players_df_2", "players_df_3"))

# Save
saveRDS(all_train_df, 'data/all_train_df.rds')
saveRDS(inference_df, 'data/inference_df.rds')

nodes <- all_train_df %>%
  mutate(from = row_number())

edges <- nodes %>%
  select(season, club, from) %>%
  inner_join(select(., season, club, to=from)) %>%
  select(from, to) %>%
  as.matrix

# ---- EDA ----
# TODO

# Which players improved the most over the course of a season?
all_train_df %>%
  filter(row_number(desc(improvement)) <= 10) %>%
  select(short_name, club, player_positions, season, age, overall, overall_next, potential)

# Which players declined the most over the course of a season?
all_train_df %>%
  filter(row_number(improvement) <= 10) %>%
  select(sofifa_id, short_name, club, player_positions, season, age, overall, overall_next, potential)