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
ALTER TABLE `depth` ADD INDEX `query_index` (`season`,`week`,`club_code`,`depth_team`,`formation`);

SELECT COUNT(*) FROM part_join;
SELECT d.club_code,d.formation,AVG(pbp.yards_gained) AS yards_gained,SUM(pbp.touchdown) AS touchdown,SUM(pbp.sack) AS sack FROM depth as d JOIN part_join as j ON d.gsis_id = j.player JOIN pbp ON pbp.uid = j.play WHERE d.season = 2023 AND d.week = 13 AND (d.club_code = 'LA' OR d.club_code = 'CLE') AND d.depth_team < 3 AND (d.formation = 'Offense' OR d.formation = 'Defense') AND pbp.season_week >= 202309 AND pbp.season_week < 202313 AND ABS(pbp.score_differential) < 21 GROUP BY d.club_code, d.formation;