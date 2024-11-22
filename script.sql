SET GLOBAL local_infile=1;

SELECT COUNT(*) FROM part_join;
DROP VIEW IF EXISTS player_stat;
CREATE VIEW player_stat AS SELECT j.player,AVG(pbp.yards_gained) AS yards_gained,SUM(pbp.touchdown) AS touchdown,SUM(pbp.sack) AS sack FROM pbp JOIN part_join as j ON j.play = pbp.uid GROUP BY j.player;
DROP VIEW IF EXISTS offense_stat;
DROP VIEW IF EXISTS defense_stat;
CREATE VIEW Offense_stat AS SELECT d.game_id,AVG(stats.yards_gained) AS yards_gained,SUM(stats.touchdown) AS touchdown,SUM(stats.sack) AS sack FROM depth as d JOIN player_stat AS stats ON d.gsis_id = stats.player WHERE depth_team < 3 AND formation = 'Offense' GROUP BY d.game_id;
CREATE VIEW Defense_stat AS SELECT d.game_id,AVG(stats.yards_gained) AS yards_gained,SUM(stats.touchdown) AS touchdown,SUM(stats.sack) AS sack FROM depth as d JOIN player_stat AS stats ON d.gsis_id = stats.player WHERE depth_team < 3 AND formation = 'Defense' GROUP BY d.game_id;
DROP VIEW IF EXISTS stats;
CREATE VIEW stats AS SELECT Offense.game_id as game_id,Offense.yards_gained AS Offense_yards_gained,Offense.touchdown AS Offense_touchdown,Offense.sack AS Offense_sack,Defense.yards_gained AS Defense_yards_gained,Defense.touchdown AS Defense_touchdown,Defense.sack AS Defense_sack FROM Offense_stat as Offense JOIN Defense_stat as Defense ON Defense.game_id = Offense.game_id;
SELECT g.*,
stats.o_yards AS home_o_yards,stats.o_sacks AS home_o_sacks,stats.o_tds AS home_o_tds,stats.d_yards AS home_d_yards,stats.d_sacks AS home_d_sacks,stats.d_tds AS home_d_tds,
away.o_yards AS away_o_yards,away.o_sacks AS away_o_sacks,stats.o_tds AS away_o_tds,away.d_yards AS away_d_yards,away.d_sacks AS away_d_sacks,away.d_tds AS away_d_tds 
FROM games as g 
JOIN stats ON CONCAT(g.season,'_',g.week,'_',g.home_team) = stats.game_id 
JOIN stats as away ON CONCAT(g.season,'_',g.week,'_',g.away_team) = away.game_id
INTO OUTFILE 'C:\\ProgramData\\MySQL\\MySQL Server 8.0\\Uploads\\nfl.csv'
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n';

SELECT * FROM games;

SELECT 'game_id','season','week','gameday','weekday','gametime','away_team','home_team','home_score','away_score','away_rest','home_rest','div_game','roof','surface','temp','wind','stadium_id','home_Offense_yards_gained','home_Offense_touchdown','home_Offense_sack','away_Offense_yards_gained','away_Offense_touchdown','away_Offense_sack','home_Defense_yards_gained','home_Defense_touchdown','home_Defense_sack','away_Defense_yards_gained','away_Defense_touchdown','away_Defense_sack' UNION ALL 
SELECT g.game_id,g.season,g.week,g.gameday,g.weekday,g.gametime,g.away_team,g.home_team,g.home_score,g.away_score,g.away_rest,g.home_rest,g.div_game,g.roof,g.surface,g.temp,g.wind,g.stadium_id,home.Offense_yards_gained AS home_Offense_yards_gained,home.Offense_touchdown AS home_Offense_touchdown,home.Offense_sack AS home_Offense_sack,away.Offense_yards_gained AS away_Offense_yards_gained,away.Offense_touchdown AS away_Offense_touchdown,away.Offense_sack AS away_Offense_sack,home.Defense_yards_gained AS home_Defense_yards_gained,home.Defense_touchdown AS home_Defense_touchdown,home.Defense_sack AS home_Defense_sack,away.Defense_yards_gained AS away_Defense_yards_gained,away.Defense_touchdown AS away_Defense_touchdown,away.Defense_sack AS away_Defense_sack FROM games as g JOIN stats as home ON CONCAT(g.season,'_',g.week,'_',g.home_team) = home.game_id JOIN stats as away ON CONCAT(g.season,'_',g.week,'_',g.away_team) = away.game_id INTO OUTFILE 'C:\\ProgramData\\MySQL\\MySQL Server 8.0\\Uploads\\nfl.csv' FIELDS TERMINATED BY ',' ENCLOSED BY '\"' LINES TERMINATED BY '\n';