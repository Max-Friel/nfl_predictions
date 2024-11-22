library(tidyverse)
library(ggrepel)
library(nflreadr)
library(nflplotR)
library(fuzzyjoin)
options(scipen = 9999)

pbp <- load_pbp(2021:2024)
pbp$uid <- paste(pbp$game_id,pbp$play_id,sep="_")

teams <- load_teams()

games <- load_schedules(2021:2024)
games$home_roster_id <- paste(games$season,games$week,games$home_team,sep="_") 
games$away_roster_id <- paste(games$season,games$week,games$away_team,sep="_")

depth <- load_depth_charts(2021:2024)
depth$game_id <- paste(depth$season,depth$week,depth$club_code,sep="_")

gamesdepthhome <- merge(x=as.data.frame(filter(depth,depth_team < 3)), y=as.data.frame(select(games,season,week,home_team,game_id)), all.x=TRUE, by.x=c("season","week","club_code"), by.y=c("season","week","home_team"))
gamesdepthaway <- merge(x=as.data.frame(filter(depth,depth_team < 3)), y=as.data.frame(select(games,season,week,away_team,game_id)), all.x=TRUE, by.x=c("season","week","club_code"), by.y=c("season","week","away_team"))

participation <- load_participation(2021:2024)
participation[c('year','week','home','away')] <- str_split_fixed(participation$nflverse_game_id,'_',4)
participation$year <- as.numeric(as.character(participation$year))
participation$week <- as.numeric(as.character(participation$week))
participation$uid <- paste(participation$nflverse_game_id,participation$play_id,sep="_")

participation.home <- regex_right_join(as.data.frame(gamesdepthhome), as.data.frame(participation), by = c("gsis_id" = "offense_players"))
  
games %>% select(game_type,weekday,gametime,away_team,home_team,away_score,home_score,location,away_rest,home_rest,div_game,roof,surface,temp,wind,stadium_id) %>% view()

