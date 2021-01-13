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
library(fmsb)
library(ggplot2)
library(viridis)

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
  mutate(nationality = fct_lump(nationality, n = 20)) %>%
  group_by(sofifa_id) %>%
  arrange(season) %>%
  mutate(overall_next = lead(overall)) %>%
  ungroup %>%
  mutate(improvement = overall_next - overall,  
         pos = case_when(player_positions == "GK" ~ "GK",          # Broader position category
                         str_detect(player_positions, "CB") ~ "CB",
                         str_detect(player_positions, "B") ~ "FB",
                         str_detect(player_positions, "AM") ~ "ATT_MID",
                         str_detect(player_positions, "RW") | str_detect(player_positions, "LW") 
                         | str_detect(player_positions, "RM")
                         | str_detect(player_positions, "LM") ~ "WING",
                         str_detect(player_positions, "M") ~ "MID",
                         str_detect(player_positions, "ST") | str_detect(player_positions, "CF") ~ "ST"),
         first_11 = ifelse(team_position %in% c("SUB", "RES", ""), FALSE, TRUE)) %>% 
  select(season:dob, value_eur, club, improvement, overall_next, everything())

# Which positions are in each category?
players_df_2 %>% 
  group_by(pos) %>%
  summarise(pos_included = paste(unique(player_positions), collapse = ", "))

# Convert factors to dummies
factor_vars <- model.matrix( ~ work_rate_att + work_rate_def + pos + player_positions + nationality - 1,
                             data = players_df_2) %>%
  as.data.frame()

# Add the one hot encoded columns on
# Create club-level features
players_df_3 <- players_df_2 %>%
  bind_cols(factor_vars) %>%
  group_by(season, club) %>%
  mutate(mean_club_overall = mean(overall, na.rm = TRUE),
         mean_club_potential = mean(potential, na.rm = TRUE),
         mean_starting_overall = mean(overall[first_11], na.rm = TRUE),
         mean_attack_overall = mean(overall[first_11 & pos %in% c("ST", "WING", "ATT_MID")], na.rm = TRUE),
         mean_mid_overall = mean(overall[first_11 & pos == "MID"], na.rm = TRUE),
         mean_def_overall = mean(overall[first_11 & pos %in% c("CB", "FB")], na.rm = TRUE),
         mean_gk_overall = mean(overall[first_11 & pos == "GK"], na.rm = TRUE)) %>%
  group_by(season, club, pos) %>%
  arrange(desc(overall)) %>%
  mutate(mean_overall_pos = (sum(overall, na.rm = TRUE) - overall)/(n() - 1), # Average of other players in same position
         mean_potential_pos = (sum(potential, na.rm = TRUE) - potential)/n() - 1,
         mean_starting_overall_pos = mean(overall[first_11], na.rm = TRUE),
         max_overall_pos_others = ifelse(overall == max(overall, na.rm = TRUE), overall[2], max(overall, na.rm = TRUE)), # Maximum rating of the *other* players available in this position
         position_rank = row_number(desc(overall))) %>%
  ungroup

# Credit to Steven Beaupre on Stackoverflow for the max_overall_pos_code
# https://stackoverflow.com/questions/30770006/dplyr-max-value-in-a-group-excluding-the-value-in-each-row

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

# ---- EDA ----

# Reload data
all_train_df <- readRDS('data/all_train_df.rds')
inference_df <- readRDS('data/inference_df.rds')

# Distribution of improvement
all_train_df %>%
  ggplot(aes(x=improvement)) +
  geom_histogram(fill = "grey60", colour = "grey40") +
  geom_vline(xintercept = mean(all_train_df$improvement, na.rm = TRUE),
             linetype = "dashed") +
  annotate("text",
           x = mean(all_train_df$improvement, na.rm = TRUE) + 2,
           y = 5000,
           label = paste("Average:", round(mean(all_train_df$improvement, na.rm = TRUE), 2))) +
  labs(title = "Target variable distribution")
  

# Biggest winners and losers
all_train_df %>%
  group_by(improvement > 0) %>%
  arrange(desc(abs(improvement))) %>%
  filter(row_number() <= 5) %>%
  ungroup %>%
  mutate(label = paste0(short_name, " (", club, ")"),
         bar_fill = improvement > 0) %>%
  select(season, short_name, club, label, age, bar_fill, improvement, overall, overall_next, pos) %>%
  ggplot(aes(x=reorder(label, improvement), y=improvement, fill = bar_fill, label = label)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  scale_fill_manual(values = c("#e03531", "#51b364")) +
  theme_minimal() +
  labs(x = "", title = "Biggest year-on-year changes")

# Improvement by position and age
all_train_df %>%
  group_by(pos) %>%
  sample_n(500) %>%
  ggplot(aes(x=age, y=improvement, colour = pos)) +
  geom_jitter(alpha = 0.5) +
  facet_wrap(~pos, ncol = 3) +
  scale_colour_viridis(discrete = TRUE) +
  geom_smooth(method = "lm", colour = "grey40", se = FALSE) +
  labs(title = "Improvement by age and position")


# Create spider graph plotting function
spider_plot <- function(inference_df, selected_club) {
  
  # Create data in the format required for fmsb spider charts
  spider_df <- inference_df %>%
    filter(club == selected_club, season == max(season)) %>%
    select(mean_club_overall:mean_gk_overall) %>%
    distinct()
  
  spider_df <- spider_df %>%
    rbind(rep(90, ncol(.)), rep(50, ncol(.)), .) %>%
    rename_all(.funs = funs(sub("mean_", "", names(spider_df))))
  
  # Custom the radarChart !
  radarchart(spider_df, axistype=1 , title = selected_club,
             
             #custom polygon
             pcol=rgb(0.2,0.5,0.5,0.9) , pfcol=rgb(0.2,0.5,0.5,0.5) , plwd=4 , 
             
             #custom the grid
             cglcol="grey", cglty=1, axislabcol="grey", caxislabels=seq(50,90,length.out = 5), cglwd=0.8,
             
             #custom labels
             vlcex=0.8 
  )
  
}

# Case study of a club
spider_plot(inference_df, "Liverpool")
spider_plot(inference_df, "Chelsea")
spider_plot(inference_df, "AFC Wimbledon")

# Which players improved the most over the course of a season?
all_train_df %>%
  filter(row_number(desc(improvement)) <= 10) %>%
  select(short_name, club, player_positions, season, age, overall, overall_next, potential)

# Which players became top-class over the course of a season?
all_train_df %>%
  filter(overall < 80, overall_next > 80, improvement > 2) %>%
  arrange(desc(improvement)) %>%
  select(short_name, club, player_positions, season, age, overall, overall_next, potential)

# Which players declined the most over the course of a season?
all_train_df %>%
  filter(row_number(improvement) <= 10) %>%
  select(sofifa_id, short_name, club, player_positions, season, age, overall, overall_next, potential)

# How does 'pecking order' correlate with improvement?
all_train_df %>%
  filter(pos == "ST",
         overall > 65, overall < 71) %>%
  select(season, short_name, overall, improvement, position_rank, max_overall_pos_others) %>%
  ggplot(aes(x=factor(position_rank), y=improvement)) +
  geom_boxplot()

all_train_df %>%
  filter(pos == "ST",
         overall > 65, overall < 71) %>%
  select(season, short_name, overall, improvement, position_rank, max_overall_pos_others) %>%
  ggplot(aes(x=max_overall_pos_others, y=improvement, colour=factor(position_rank))) +
  geom_jitter(alpha = 0.5)
