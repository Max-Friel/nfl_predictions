library(tidyverse)
library(ggrepel)
library(nflreadr)
library(nflplotR)
library(fuzzyjoin)
library(RMySQL)
options(scipen = 9999)

games$elo_predicted_win_perc <- 0
games$elo_home <- 1600
games$elo_away <- 1600

players <- data.frame(uid=character(), elo = integer(), count = integer())
for (year in 2020:2023){
  pbp2020 <- load_pbp(year)
  pbp2020$turnover <- pbp2020$fumble_lost + pbp2020$interception
  pbp2020$td <- ifelse(is.na(pbp2020$td_team) | pbp2020$td_team != pbp2020$posteam,0,1)
  pbp2020$uid <- paste(pbp2020$game_id,pbp2020$play_id,sep="_")
  pbp2020 <- select(pbp2020,uid,td,turnover,yards_gained,week)
  pbp2020 <- filter(pbp2020, !is.na(yards_gained))
  participation2020 <- load_participation(year)
  participation2020 <- filter(participation2020,offense_players != "")
  participation2020$uid <- paste(participation2020$nflverse_game_id,participation2020$play_id,sep="_")
  merg <- merge(as.data.frame(participation2020),as.data.frame(pbp2020), by="uid")
  merg$k <- ifelse((merg$turnover + merg$td) > 0, 20,10)
  merg$off_out <- ifelse(merg$yards_gained > 2, 1,0)
  merg$def_out <- ifelse(merg$yards_gained <= 0, 1,0)
  merg$off_out <- ifelse(merg$yards_gained > 0 & merg$yards_gained < 2, .5,merg$off_out)
  merg$def_out <- ifelse(merg$yards_gained > 0 & merg$yards_gained < 2, .5,merg$def_out)
  merg <- merg[with(merg,order(week,play_id)),]
  
  depth2 <- load_depth_charts(year)
  
  week = 1
  for(i in 1:nrow(merg)){
    
    if (week != merg[i,"week"]){
      
      gamesWeek <- filter(games, merg[i,"week"] == games$week & games$season == year)
      rostersWeek <- filter(depth2,depth_team == 1 & merg[i,"week"] == depth2$week & depth2$season == year)
      if(nrow(gamesWeek) > 5 & nrow(rostersWeek) > 1){
       
        
        for(g in 1:nrow(gamesWeek)){
         
          home_team = gamesWeek[g,"home_team"][[1]]
          away_team = gamesWeek[g,"away_team"][[1]]
          gamesWeek[g,"elo_home"] <- mean(merge(x=as.data.frame(filter(rostersWeek,club_code == home_team)),y=as.data.frame(players),by.x="gsis_id",by.y="uid")$elo)
          gamesWeek[g,"elo_away"] <- mean(merge(x=as.data.frame(filter(rostersWeek,club_code == away_team)),y=as.data.frame(players),by.x="gsis_id",by.y="uid")$elo)
        }
        gamesWeek$predicted_win_perc <- 1/(1 + 10^((mean(gamesWeek$elo_away)-gamesWeek$elo_home)/400))
        games$elo_home[match(gamesWeek$game_id, games$game_id)] <- gamesWeek$elo_home
        games$elo_away[match(gamesWeek$game_id, games$game_id)] <- gamesWeek$elo_away
        games$elo_predicted_win_perc[match(gamesWeek$game_id, games$game_id)] <- gamesWeek$predicted_win_perc
      }
      week = merg[i,"week"]
    }
    opl <- strsplit(merg[i,"offense_players"][[1]],";")[[1]]
    dpl <- strsplit(merg[i,"defense_players"][[1]],";")[[1]]
    pl <- append(opl,dpl)
    newPlayers <- pl[!(pl %in% players$uid)]
    if(length(newPlayers) != 0){
      playerAdd <- data.frame(uid=pl[!(pl %in% players$uid)], elo = 1600, count = 0)
      players <- rbind(players, playerAdd)
    }
    
    offense = players[players$uid %in% opl,]
    defense = players[players$uid %in% dpl,]
    
    offense$prob = 1/(1 + 10^((mean(defense$elo)-offense$elo)/400))
    defense$prob = 1/(1 + 10^((mean(offense$elo)-defense$elo)/400))
    offense$elo = offense$elo + (merg[i,"k"]*(merg[i,"off_out"] - offense$prob))
    offense$count = offense$count +1
    defense$elo = defense$elo + (merg[i,"k"]*(merg[i,"def_out"] - defense$prob))
    defense$count = defense$count +1
    
    players$elo[match(offense$uid, players$uid)] <- offense$elo
    players$count[match(offense$uid, players$uid)] <- offense$count
    players$elo[match(defense$uid, players$uid)] <- defense$elo
    players$count[match(defense$uid, players$uid)] <- defense$count
  }
  depth2 <- load_depth_charts(year+1)
  gamesWeek <- filter(games, 1 == games$week & games$season == year + 1)
  rostersWeek <- filter(depth2,depth_team == 1 & week == 1 & depth2$season == year + 1)
  print(nrow(gamesWeek))
  if(nrow(gamesWeek) > 1){
    for(g in 1:nrow(gamesWeek)){
      gamesWeek[g,"elo_home"] <- mean(merge(x=as.data.frame(filter(rostersWeek,club_code == gamesWeek[g,"home_team"][[1]])),y=as.data.frame(players),by.x="gsis_id",by.y="uid")$elo)
      gamesWeek[g,"elo_away"] <- mean(merge(x=as.data.frame(filter(rostersWeek,club_code == gamesWeek[g,"away_team"][[1]])),y=as.data.frame(players),by.x="gsis_id",by.y="uid")$elo)
    }
    gamesWeek$predicted_win_perc <- 1/(1 + 10^((mean(gamesWeek$elo_away)-gamesWeek$elo_home)/400))
    games$elo_home[match(gamesWeek$game_id, games$game_id)] <- gamesWeek$elo_home
    games$elo_away[match(gamesWeek$game_id, games$game_id)] <- gamesWeek$elo_away
    games$elo_predicted_win_perc[match(gamesWeek$game_id, games$game_id)] <- gamesWeek$predicted_win_perc
  }
}
