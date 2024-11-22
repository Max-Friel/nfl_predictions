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

dbListTables(mysql)

pbp <- load_pbp(2021:2024)
pbp$uid <- paste(pbp$game_id,pbp$play_id,sep="_")
pbp <- select(pbp,game_id,week,season,uid,yards_gained,td_player_id,interception,penalty,fumble_forced,fumble_not_forced,fumble_lost,sack,touchdown,receiver_player_id,passer_player_id,rusher_player_id,interception_player_id,sack_player_id,half_sack_1_player_id,half_sack_2_player_id,penalty_player_id,penalty_yards)
dbWriteTable(mysql,value = as.data.frame(pbp), name = "pbp", overwrite=TRUE)

teams <- load_teams()
dbWriteTable(mysql,value = as.data.frame(teams), name = "teams", overwrite=TRUE)

games <- load_schedules(2021:2024)
games$home_roster_id <- paste(games$season,games$week,games$home_team,sep="_") 
games$away_roster_id <- paste(games$season,games$week,games$away_team,sep="_")
dbWriteTable(mysql,value = as.data.frame(games), name = "games", overwrite=TRUE)

depth <- load_depth_charts(2021:2024)
depth$game_id <- paste(depth$season,depth$week,depth$club_code,sep="_")
dbWriteTable(mysql,value = as.data.frame(depth), name = "depth", overwrite=TRUE)

gamesdepthhome <- merge(x=as.data.frame(filter(depth,depth_team < 3)), y=as.data.frame(select(games,season,week,home_team,game_id)), all.x=TRUE, by.x=c("season","week","club_code"), by.y=c("season","week","home_team"))
gamesdepthaway <- merge(x=as.data.frame(filter(depth,depth_team < 3)), y=as.data.frame(select(games,season,week,away_team,game_id)), all.x=TRUE, by.x=c("season","week","club_code"), by.y=c("season","week","away_team"))
dbWriteTable(mysql,value = as.data.frame(gamesdepthhome), name = "gamesdepthhome", overwrite=TRUE)
dbWriteTable(mysql,value = as.data.frame(gamesdepthaway), name = "gamesdepthaway", overwrite=TRUE)

participation <- load_participation(2021:2024)
participation[c('year','week','home','away')] <- str_split_fixed(participation$nflverse_game_id,'_',4)
participation$year <- as.numeric(as.character(participation$year))
participation$week <- as.numeric(as.character(participation$week))
participation$uid <- paste(participation$nflverse_game_id,participation$play_id,sep="_")
participation$players <- paste(participation$offense_players,participation$defense_players,sep=";")

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

fields <- list('AVG:yards_gained','SUM:touchdown','SUM:sack')
sides <- list('Offense','Defense')
teams <- list('home','away')
gameFields <- list('game_id','season','week','gameday','weekday','gametime','away_team','home_team','home_score','away_score','away_rest','home_rest','div_game','roof','surface','temp','wind','stadium_id')

query <- 'CREATE VIEW player_stat AS SELECT j.player'
for(field in fields){
  for(split in strsplit(field,":")){
    query <- paste0(query,",",split[[1]],"(pbp.",split[[2]],") AS ", split[[2]])
  }
}
query <- paste0(query," FROM pbp JOIN part_join as j ON j.play = pbp.uid GROUP BY j.player;")
print(query)

for(side in sides){
  query <- paste0('CREATE VIEW ',side,'_stat AS SELECT d.game_id')
  for(field in fields){
    for(split in strsplit(field,":")){
      query <- paste0(query,",",split[[1]],"(stats.",split[[2]],") AS ", split[[2]])
    }
  }
  query <- paste0(query,' FROM depth as d JOIN player_stat AS stats ON d.gsis_id = stats.player WHERE depth_team < 3 AND formation = \'',side,'\' GROUP BY d.game_id;')
  print(query)
}

query <- 'CREATE VIEW stats AS SELECT Offense.game_id as game_id'
for(side in sides){
  for(field in fields){
    for(split in strsplit(field,":")){
      query <- paste0(query,",",side,".",split[[2]]," AS ", side,"_",split[[2]])
    }
  }
}
query <- paste0(query,' FROM Offense_stat as Offense JOIN Defense_stat as Defense ON Defense.game_id = Offense.game_id;')
print(query)

header <- 'SELECT '
query <- 'SELECT '
for(fld in gameFields){
  query <- paste0(query,",g.",fld)
  header <- paste0(header,'\'',fld,'\',')
}
for(side in sides){
  for(team in teams){
    for(field in fields){
      for(split in strsplit(field,":")){
        query <- paste0(query,",",team,".",side,"_",split[[2]]," AS ",team,"_", side,"_",split[[2]])
        header <- paste0(header,"\'",team,"_", side,"_",split[[2]],"\',")
      }
    }
  }
}
query <- paste0(query,' FROM games as g JOIN stats as home ON CONCAT(g.season,\'_\',g.week,\'_\',g.home_team) = home.game_id JOIN stats as away ON CONCAT(g.season,\'_\',g.week,\'_\',g.away_team) = away.game_id INTO OUTFILE \'C:\\ProgramData\\MySQL\\MySQL Server 8.0\\Uploads\\nfl.csv\' FIELDS TERMINATED BY \',\' ENCLOSED BY \'"\' LINES TERMINATED BY \'\n\';')
header <- paste0(header, " UNION ALL ")
print(header)
print(query)