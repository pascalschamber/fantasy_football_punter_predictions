**Motivation**

Punting is an underappreciated facet of football. Even though the majority of drives end with punts, the statistics are haphazardly tracked and there is a lack of established metrics for determining the skill levels of punters. This is especially true for fantasy football, where most leagues do not even include punters as a position, leading to a dearth of information on which punters to draft and fantasy scoring predictions. This motivated the creation of this package, to conglomerate historic punting data and use this to generate a model to predict punter scoring on a weekly basis, which does not appear to have been done before. 

Punters are valuable fantasy contributors, scoring on average a similar number of points as running backs and wide receivers, which command a majority of fantasy attention (fig 1). Like all positions in fantasy football, success varies substantially week-to-week, and punters are no exception (fig 2). Therefore, the challenge of fantasy football is to evaluate a long list of players, select the ones you think harbor the most favorable matchups, and hope for the best. For most fantasy team manager this amounts to looking at the top performers and adding them to your roster. This approach fails to address matchup related information and can be subject to the woes of inconsistent performance demonstrated in fig 2 to be a major concern.

For example, in 2019, if you just ran one of the eventual top 5 punters each week you would have scored 280 points. However, if you somehow managed to pick the best performing punter each week over a season you would have scored 547 points, which is more points than any other fantasy player scored that season. While this an impossible expectation, there is more hope for the possibility of doing this for punters since the majority of punters are available to pickup each week. While most of the best performing running backs and wide receivers are locked up on rosters, the free agency pool for punters is comparatively vast (66% availability) and represents an opportunity for insightful predictions to pay off. Figure 3 shows binned punter scoring on a week-to-week basis dating back to 1999, the average number of players that score that many points and what their punter rank was at that time. This highlights that valuable waiver wire pickups can be had each week and underscores the value a predictive model could bring.

**What makes a good punter?**

As with any game of chance, quantifiable predictors of future success bring valuable non-subjective information to a world dominated by opinions. Logically we can boil this down for punters to a few key facets for success.
•	Being on a low-middle rank team -> not too good to never be punting, but not too bad to be so far behind you stop punting all together
•	Strong defense -> keep games competitive so punting isn’t abandoned late in the game
•	Being on an offense that moves the ball up to mid field but fails shortly after -> gives best possibility to score a punt inside 10
•	Have a bad kicker -> less likely to kick long field goals and instead kick short punts
•	Great special team coverage -> avoid long punt returns by opponsent and down the ball close to the end zone
•	Non-aggressive coaching -> Low “go for it” on 4th down in opponent territory therefore more opportunities to kick short punts
This makes it clear that punting is heavily reliant on aspects of the whole football team as well as the team they are playing on a given week. This justifies the use of team-level statics as the features for the training data.

**Generating the training data**

The first thing I did was scrape all punting data from 1990 to 2022 from play-by-play information recorded by nflfastR and calculate fantasy scores (can be found here: https://github.com/nflverse/nflverse-data/releases/tag/pbp) 
Then to address the multifaceted nature of the matchups to predicting punter success, I scraped historical team data from NFL.com and appended these statistics to each matchup for the punter’s team and opponent. 
This amounts to over 12,000 training examples, comprising 3,000,000 features. 

**Feature selection**

This is arguably too many features to train a generalizable model. Following this step, I performed some feature selection and engineering to whittle away redundancies and extraneous variables. Removing features with low variance and high correlation to other features brought this number down to 88. I also tried recursive feature elimination, but this failed to improve model performance.

**Model**

With the dataset curated, I set up a simple neural network with two hidden layers and one dropout layer to train for this regression task. This model takes in all of the team-level statistics and generates a prediction for each of the scoring categories on which punter fantasy scores are calculated. I choose to use a k-fold cross-validation scheme since there are lots of training examples, fantasy scores follow a normal distribution, and don’t fluctuate significantly from season to season. And used mean squared error as the evaluation metric as it performed best when calculating the error summed over the testing data. After testing different amounts of parameters, I arrived at the following model.


**Future work**

Overall, I think there is much more to be desired from the model’s predictive power. Perhaps some features could be engineered that more accurately reflect the points outlined for punter success. I will experiment with other models as well, though decision trees did not offer any performance increases. 

At the end of the day, football statistics are highly variable and there are definite elements of chance involved. Even the best players throw up a dud occasionally, I’m sure punters have good and days as well. Though I do see some promise in this model and hope it will be useful to get the ball rolling on giving punters the credit they deserve.
