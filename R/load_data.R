# Data downloaded from kaggle at https://www.kaggle.com/stefanoleone992/fifa-20-complete-player-dataset
library(data.table)
library(dplyr)
library(tidyr)
library(stringr)
library(forcats)

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
  mutate(season = str_remove(season, "data/players_"),
         season = str_remove(season, "\\.csv")) %>%
  select(
    season, sofifa_id, short_name, dob,
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
  mutate_at(vars(season, overall, potential, age, height_cm, weight_kg, 
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
  select(season:dob, club, improvement, overall_next, everything())
  

# Which columns have any NAs in?
summary(players_df_2)

# Only keep players who appear in every edition of the game
players_df_3 <- players_df_2 %>%
  group_by(sofifa_id) %>%
  filter(n() == length(files)) %>%
  ungroup

# Which players improved the most over the course of a season?
players_df_3 %>%
  filter(row_number(desc(improvement)) <= 10) %>%
  select(short_name, club, player_positions, season, age, overall, overall_next, potential)

# Which players declined the most over the course of a season?
players_df_3 %>%
  filter(row_number(improvement) <= 10) %>%
  select(sofifa_id, short_name, club, player_positions, season, age, overall, overall_next, potential)

# Convert factors to dummies
