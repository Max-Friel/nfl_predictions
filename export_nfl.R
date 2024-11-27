library(tidyverse)
library(ggrepel)
library(nflreadr)
library(nflplotR)
library(fuzzyjoin)
library(RMySQL)
options(scipen = 9999)

mysql = dbConnect(RMySQL::MySQL(),
                            dbname='nfl_schema',
                            host='localhost',
                            port=3306,
                            user='root',
                            password='Demo1234!')

pbp <- load_pbp(2020:2024)
pbp$uid <- paste(pbp$game_id,pbp$play_id,sep="_")
pbp <- select(pbp,game_id,week,season,uid,yards_gained,td_player_id,interception,penalty,score_differential,fumble_forced,fumble_not_forced,fumble_lost,sack,touchdown,receiver_player_id,passer_player_id,rusher_player_id,interception_player_id,sack_player_id,half_sack_1_player_id,half_sack_2_player_id,penalty_player_id,penalty_yards)
pbp$season_week <- (pbp$season * 100) + pbp$week
dbWriteTable(mysql,value = as.data.frame(pbp), name = "pbp", overwrite=TRUE)

teams <- load_teams()
dbWriteTable(mysql,value = as.data.frame(teams), name = "teams", overwrite=TRUE)

games <- load_schedules(2021:2024)
games <- select(games,game_id,season,week,gameday,weekday,gametime,away_team,home_team,home_score,away_score,away_rest,home_rest,div_game,roof,surface,temp,wind,stadium_id)
games$home_roster_id <- paste(games$season,games$week,games$home_team,sep="_") 
games$away_roster_id <- paste(games$season,games$week,games$away_team,sep="_")
games$season_week <- (games$season * 100) + games$week
dbWriteTable(mysql,value = as.data.frame(games), name = "games", overwrite=TRUE)

depth <- load_depth_charts(2021:2024)
depth$depth_team <- as.numeric(as.character(depth$depth_team))
depth$game_id <- paste(depth$season,depth$week,depth$club_code,sep="_")
dbWriteTable(mysql,value = as.data.frame(depth), name = "depth", overwrite=TRUE)

participation <- load_participation(2020:2024)
participation[c('year','week','home','away')] <- str_split_fixed(participation$nflverse_game_id,'_',4)
participation$year <- as.numeric(as.character(participation$year))
participation$week <- as.numeric(as.character(participation$week))
participation$uid <- paste(participation$nflverse_game_id,participation$play_id,sep="_")
participation$players <- paste(participation$offense_players,participation$defense_players,sep=";")

dbRemoveTable(mysql,"part_join")
for(row in 1:nrow(participation)){
  part.players <- data.frame(player=character(),play=character())
  for(playerlist in participation[row, "players"]){
    for(player in strsplit(playerlist,";")){
      part.players <- part.players %>% add_row(player = player, play= as.data.frame(participation)[row, "uid"])
    }
  }
  dbWriteTable(mysql,value = part.players, name = "part_join", append=TRUE)
}
dbWriteTable(mysql,value = as.data.frame(participation), name = "participation", overwrite=TRUE)

fields <- list('AVG:yards_gained','AVG:touchdown','AVG:sack')
sides <- list('Offense','Defense')
teams <- list('home','away')

testGames <- games

get_query <- function(season,week,weeks_back,home_team,away_team){
  query <- paste0('SELECT d.club_code,d.formation')
  for(field in fields){
    for(split in strsplit(field,":")){
      query <- paste0(query,",",split[[1]],"(pbp.",split[[2]],") AS ", split[[2]])
    }
  }
  prev_week = week - weeks_back
  season_back = season
  while(prev_week < 0){
    prev_week = prev_week + 17
    season_back = season_back - 1
  }
  query <- paste0(query,' FROM depth as d JOIN part_join as j ON d.gsis_id = j.player JOIN pbp ON pbp.uid = j.play WHERE d.season = ',season,' AND d.week = ',week,' AND (d.club_code = \'',home_team,'\' OR d.club_code = \'',away_team,'\') AND d.depth_team < 3 AND (d.formation = \'Offense\' OR d.formation = \'Defense\') AND pbp.season_week >= ',((season_back)*100) + prev_week,' AND pbp.season_week < ',((season)*100) + week,' AND ABS(pbp.score_differential) < 21 GROUP BY d.club_code, d.formation;')
  print(query)
}
#run indexes before running this
newColumnYear <- list()
newColumn4 <- list()
for(row in 1:nrow(testGames)){
  game <-as.data.frame(testGames)[row, "game_id"]
  seasonweek <- as.data.frame(testGames)[row, "season_week"]
  season <- as.data.frame(testGames)[row, "season"]
  week <- as.data.frame(testGames)[row, "week"]
  home_team <- as.data.frame(testGames)[row, "home_team"]
  away_team <- as.data.frame(testGames)[row, "away_team"]
  
  queryYear <- get_query(season,week,17,home_team,away_team)
  rsYear <- dbSendQuery(mysql,queryYear)
  resYear <- fetch(rsYear,n=-1)
  
  query4 <- get_query(season,week,4,home_team,away_team)
  rs4 <- dbSendQuery(mysql,query4)
  res4 <- fetch(rs4,n=-1)
  
  i = 1
  for(team in teams){
    t =  away_team
    if(team == 'home'){
      t = home_team
    }
    for(side in sides){
      for(field in fields){
        for(split in strsplit(field,":")){
          fld <- split[[2]]
          if(i > length(newColumnYear)){
            newColumnYear[[i]] <- vector()
            newColumn4[[i]] <- vector()
          }
          valYear <- filter(resYear,club_code == t & formation == side)[1,fld]
          newColumnYear[[i]] <- append(newColumnYear[[i]],valYear)
          
          val4 <- filter(res4,club_code == t & formation == side)[1,fld]
          newColumn4[[i]] <- append(newColumn4[[i]],val4)
          
          i = i + 1
        }
      }
    }
  }
}

i = 1
for(team in teams){
  for(side in sides){
    for(field in fields){
      for(split in strsplit(field,":")){
        fld <- split[[2]]
        
        fld_name <- paste0('pastYear_',team,'_',side,'_',fld)
        testGames[fld_name] <- newColumnYear[[i]]
        
        fld_name <- paste0('past4_',team,'_',side,'_',fld)
        testGames[fld_name] <- newColumn4[[i]]
        i = i + 1
      }
    }
  }
}
write.csv(testGames,"C:\\Users\\Max\\Documents\\nfl_predictions\\data\\games.csv")
