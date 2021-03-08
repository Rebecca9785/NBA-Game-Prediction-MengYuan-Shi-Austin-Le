# NBA-Game-Prediction
Project Group: MengYuan Shi, Austin Le

This repository contains a data science project that discover the NBA Game Prediction. We investigate the social network for individual NBA players and the relationship between each team. We will use team's statistics and players' statistics and analysis for predicting who wins the games by leveraging the team's statistics and players' statistics from 2015 season to 2019 season. We will use GraphSAGE which is a generalizable embedding framework to create a graph classification.

### Running the project
- To get the data, from the project root dir, run python run.py data
- src/data/data.py gets features, labels for graph data
- src/models/models.py contains our neural networl models
- run.py can be run from the command line to ingest data, train a model, and present relevant statistics for model performance to the shell

### Responsibility 
- Austin Le: Responsible for the data cleaning and data scraping and the coding part as well as the report.
- MengYuan Shi: Responsible for the paper researching and writing the report part as well as the visualization.
