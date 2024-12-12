There are two parts to our project.  
First the data extraction which produces games.csv found in the data folder.  
Second the data analysis which runs off of the games.csv.
All the machine learning logic is contained within the nfl_process.py file

To produce the games.csv
1 - Setup a local mysql instance with root password of Demo1234!
2 - Run everything in export_nfl.r until #run indexes before running this
3 - Run the indexing setup in script.sql, this step is technically not required but will make the next step finish in hours rather than days 
4 - Run the rest of export_nfl.r script
5 - Verify data\games.csv has been created

Once you have a games.csv
1 - Run nfl_process.py 