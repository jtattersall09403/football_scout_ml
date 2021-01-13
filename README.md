# AI Scout: Using machine learning to identify high-value targets in the football transfer market

The football transfer market is big business. This repo provides a playground for exploring the kinds of tools that football clubs can use to predict whether a transfer target is likely to succeed at their club, using machine learning (specifically `xgboost`).

## Running the code

1. Clone the repo
2. Open the Rproj file
3. Install `renv` (`install.packages("renv")`)
4. Run renv::restore() to install dependencies
5. Run the scripts in the `R` folder in numerical order. The final script produces predictions for the latest data for you to explore

## The data

The project is based on the [Kaggle Complete FIFA 20 Dataset](https://www.kaggle.com/stefanoleone992/fifa-20-complete-player-dataset), which also includes data from each version of the game since 2015.

## The objective

As the datasets provide a snapshot of each player's ratings at the start of the corresponding season, these ratings approximately represent how well that player performed in the *previous* season.

For each season, we therefore aim to predict how much higher or lower each player's overall rating will be in the subsequent season. In doing so, we leverage features not only about the player (e.g. FIFA's 'potential' rating; physical and footballing attributes; nationality etc), but also about the other players around them (e.g. the average overall rating of players in their club in each position).

When making predictions about how a player will improve/decline in a new club, we can replace these features with those relating to that new club, and see how it affects the prediction