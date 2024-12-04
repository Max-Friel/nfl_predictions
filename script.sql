SET GLOBAL local_infile=1;

ALTER TABLE `pbp` ADD INDEX `season_week_index` (`season_week`);
ALTER TABLE pbp MODIFY uid VARCHAR(20);
ALTER TABLE `pbp` ADD INDEX `uid_index` (`uid`);
ALTER TABLE `pbp` ADD INDEX `query_index` (`season_week`,`score_differential`);

ALTER TABLE part_join MODIFY player VARCHAR(10);
ALTER TABLE `part_join` ADD INDEX `player_index` (`player`);
ALTER TABLE part_join MODIFY play VARCHAR(20);
ALTER TABLE `part_join` ADD INDEX `play_index` (`play`);

ALTER TABLE depth MODIFY gsis_id VARCHAR(10);
ALTER TABLE `depth` ADD INDEX `gsis_id_index` (`gsis_id`);
ALTER TABLE depth MODIFY club_code VARCHAR(3);
ALTER TABLE depth MODIFY formation VARCHAR(20);
ALTER TABLE `depth` ADD INDEX `query_index1` (`club_code`,`formation`);
ALTER TABLE `depth` ADD INDEX `query_index` (`season`,`week`,`club_code`,`depth_team`,`formation`);

UPDATE games SET temp = 75 WHERE temp IS NUll;

SELECT home_team, surface, roof, COUNT(*) as c FROM games GROUP BY home_team,surface,roof ORDER BY home_team,c DESC;
SELECT AVG(temp),home_team FROM games group by home_team;
SELECT posteam,AVG(time) as time FROM poss WHERE season_week >= 202306 AND season_week < 202406 AND (posteam = 'CAR' OR posteam = 'ATL') GROUP BY posteam;
SELECT posteam,AVG(time) as time FROM poss WHERE season_week >= 202402 AND season_week < 202406 AND (posteam = 'CAR' OR posteam = 'ATL') GROUP BY posteam;
SELECT d.club_code,d.formation,AVG(pbp.yards_gained) AS yards_gained,AVG(pbp.touchdown) AS touchdown,AVG(pbp.sack) AS sack,AVG(pbp.penalty_yards) AS penalty_yards,AVG(pbp.fumble_lost) AS fumble_lost,AVG(pbp.interception) AS interception FROM depth as d JOIN part_join as j ON d.gsis_id = j.player JOIN pbp ON pbp.uid = j.play WHERE d.season = 2024 AND d.week = 8 AND (d.club_code = 'WAS' OR d.club_code = 'CHI') AND d.depth_team < 3 AND (d.formation = 'Offense' OR d.formation = 'Defense') AND pbp.season_week >= 202308 AND pbp.season_week < 202408 AND ABS(pbp.score_differential) < 21 GROUP BY d.club_code, d.formation;