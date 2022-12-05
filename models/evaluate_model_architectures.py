import pandas as pd
import os
import sys
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns
from preprocess_training_data import Preprocessor
from feature_engineering import get_preprocessor
sys.path.insert(0, r'..\utilities')
import ff_utils as fu

def save_fig(afig, outpath):
    afig.savefig(outpath,
        bbox_inches = 'tight', dpi=300)

def plot_history(history):
    history_df = pd.DataFrame(history.history)
    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')
    ax.plot(history_df.index, history_df['loss'], label='Train Error')
    ax.plot(history_df.index, history_df['val_loss'], label='Val Error')
    ax.legend()
    plt.show()

def plot_legend_outside_ax(g):
    g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=2)

def save_model_results(model,preprocssing_obj, test_mse, sum_season_err,epochs, lr, loss_fxn, loss_metric):
    '''save a summary of the model that was tested'''
    thedate = datetime.now().strftime('%Y_%m%d_%X') 
    outmode = 'w'
    if os.path.exists('model_results_history.txt'):
        outmode = 'a+'
    with open('model_results_history.txt', outmode, encoding='utf-8') as f:
        f.write('#'*60+ '\n')
        f.write(thedate + '#' + '\n')
        f.write('#'*20+ '\n')
        f.write(preprocssing_obj.get_preprocessing_parameters())
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write(f'test mse:\t{test_mse}'+ '\n')
        f.write(f'sum season err:\t{sum_season_err}'+ '\n')
        f.write(f'learning rate:\t{lr}'+ '\n')
        f.write(f'loss function:\t{loss_fxn}'+ '\n')
        f.write(f'loss metric:\t{loss_metric}'+ '\n')
        f.write(f'epochs:\t{epochs}'+ '\n')
        f.write('#'*60+ '\n'*3)

def simple_model_summary(**kwargs):
    '''simple model summary for writing dates and errors to, use this to quickly compare then get
        full model params from model results history to reimplement '''
    thedate = datetime.now().strftime('%Y_%m%d_%X') 
    astr = f'{thedate} #\n' + '#'*20+ '\n'
    for k, v in kwargs.items():
        astr += f'{k}: {v}\n'
    
    # append to a single text file
    outmode='w'
    if os.path.exists('model_results_simple.txt'):
        outmode = 'a+'
    with open('model_results_simple.txt', outmode, encoding='utf-8') as f:
        f.write('#'*60+ '\n')
        f.write(astr)
        f.write('#'*60+ '\n'*3)

def unscale_prediction_array(test_predict):
    arrays = []
    for col_index in range(test_predict.shape[1]):
        col_array = test_predict[:, col_index]
        col_name = preprocessor.targets[col_index]
        scaler = preprocessor.get_scaler(col_name)
        unscaled = scaler.inverse_transform(col_array.reshape(-1, 1))
        arrays.append(unscaled)

    return np.hstack(arrays)

def get_punt_yds_score(yds):
    if yds >= 44.0:
        score = 5
    elif yds >= 42.0:
        score = 4
    elif yds >= 40.0:
        score = 3
    elif yds >= 38.0:
        score = 2
    elif yds >= 36.0:
        score = 1
    else:
        score = -2
    return score 

def get_fantasy_score(arr):
    # TODO add faircatches
    fantasy_score = get_punt_yds_score(arr[0]) + (-4*arr[1]) + (5*arr[2]) + (3*arr[3]) + (-2*arr[4]) + (2*arr[5])
    return fantasy_score

def convert_stats_to_fantasy_score(arrs):
    # input is expected to be a x,y array where len(x) is n matchups and len(y) is number of stat fields
    return [get_fantasy_score(arr) for arr in arrs]

def mean_square_error(arr1, arr2):
    # get the mean-squared error and array of errors
    ses = [(el1 - el2)**2 for (el1, el2) in zip(arr1, arr2)]
    mse = np.sum(ses)/len(ses)
    return mse, ses

def mean_error(arr1, arr2):
    # get the mean-squared error and array of errors
    ses = [(el1 - el2) for (el1, el2) in zip(arr1, arr2)]
    me = np.sum(ses)/len(ses)
    return me, ses


def logcosh(true, pred):
    # log cosh loss
    loss = np.log(np.cosh(pred - true))
    return np.sum(loss)


def plot_test_prediction_over_games(predicted_scores, real_scores, SAVE=False):
    
    fig, ax = plt.subplots()
    ax.set_xlabel('Games')
    ax.set_ylabel('P_TOT_PTS')
    ax.plot(range(len(predicted_scores)), predicted_scores, label='prediction', alpha=0.9)
    ax.plot(range(len(real_scores)), real_scores, label='real scores', alpha=0.5)
    # mse, ses = mean_error(real_scores, predicted_scores)
    # ax.plot(range(len(ses)), ses, label=f'mse = {mse}', c='k', alpha=1.0)
    plt.legend()
    if SAVE:
        save_fig(fig, 'plot_prediction_over_games_2019.svg')
    plt.show()

def get_df_summed_season_predictions_real(preprocessor, predicted_scores, real_scores):
    # get total scores for each team, for real and predictions
    test_df = preprocessor.testing_data
    test_df_teams = test_df.p_team.values
    pred_df = pd.DataFrame({'p_team':test_df_teams, 'pscore':predicted_scores})
    real_df = pd.DataFrame({'p_team':test_df_teams, 'rscore':real_scores})

    psumdf = pred_df.groupby('p_team').sum().reset_index(level=[0]).set_index('p_team')
    rsumdf = real_df.groupby('p_team').sum().reset_index(level=[0]).set_index('p_team')

    combine_df = pd.melt(psumdf.join(rsumdf), ignore_index=False).sort_index()
    return combine_df

'''
######################################################################################
# Pre-processing
    #just need to specify path to training data if up to date
        # otherwise call func below to merge historical matchups onto team season ranks
            # merged_pdf = fu.merge_team_stats_onto_punter_matchup()
            # merged_pdf.to_excel('../data/training_data_2009-2019.xlsx', index=False)
######################################################################################
'''

# if updating preprocessor change to true, else just load it
UPDATE_PREPROCESSOR = bool(0)
preprocessor = get_preprocessor(UPDATE_PREPROCESSOR)
trainX, trainY, valX, valY, testX, testY = preprocessor.get_datasets()

print(f'shape of\n\ttraining: {trainX.shape}, {trainY.shape}\n\tval: {valX.shape}, {valY.shape}\n\ttest: {testX.shape}, {testY.shape}')

'''
######################################################################################
# define model
    best architecture
        l1, l2 nodes = 256, 512 
            though results appear variable 
######################################################################################
'''

# evaluate loss functions
#######################################################################################################


# tuned model params
l1_n = 256
l2_n = 512
epochs = 100
lr = 1e-05
loss_fxn = tf.keras.losses.MeanSquaredError()
callable_loss_fxn = mean_square_error
loss_metric = tf.keras.metrics.MeanSquaredError()

# define model
#######################################################################################################
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(l1_n, input_shape=[trainX.shape[1]], activation='relu'),
tf.keras.layers.Dense(l2_n, activation='relu'),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(trainY.shape[1])
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
    loss=loss_fxn, 
    metrics = [loss_metric])

# print(model.summary())

# fit model
#######################################################################################################
history = model.fit(
    trainX, trainY, validation_data=(valX, valY), 
    verbose=2, epochs=epochs
)
plot_history(history)


# test model
#######################################################################################################
# make predictions
test_predict = model.predict(testX)
unscaled_predictions = unscale_prediction_array(test_predict)
unscaled_real = unscale_prediction_array(testY)
# convert predictions to fantasy score
predicted_scores = convert_stats_to_fantasy_score(unscaled_predictions)
real_scores = convert_stats_to_fantasy_score(unscaled_real)
# get error
test_error = callable_loss_fxn(real_scores, predicted_scores)[0] 
print('error:', test_error)
# get error of summed season scores
combine_df = get_df_summed_season_predictions_real(preprocessor, predicted_scores, real_scores)
sum_season_err = callable_loss_fxn(
    combine_df[combine_df.variable=='rscore']['value'].values, 
    combine_df[combine_df.variable=='pscore']['value'].values
)[0]
print('sum_season_err:', sum_season_err)

# save model results
save_model_results(model, preprocessor, test_error, sum_season_err, epochs, lr, loss_fxn, loss_metric)
simple_model_summary(test_error=test_error, season_sum_error=sum_season_err, loss_fxn=loss_fxn, archl1l2=[l1_n, l2_n])
# plot mse
plot_test_prediction_over_games(predicted_scores, real_scores, SAVE=True)





########################################################################################
# inspect model
########################################################################################

if bool(0):
    # does the model accurately predict the season leaders?
    # short answer, not really
    # bar chart for each team with stacked pred vs real    
    # plot
    fig, ax = plt.subplots()
    ax.set_xlabel('team')
    ax.set_ylabel('score')
    ax.set_xticklabels(ax.get_xticks(), rotation=-90)

    g = sns.barplot(data=combine_df, x=combine_df.index, y='value', hue='variable', ax=ax)
    plot_legend_outside_ax(g)
    # save_fig(fig, 'sum_season_points_2019.svg')
    plt.show()

    # print ranked scores side by side
    print(combine_df[combine_df.variable=='rscore'].sort_values(['value'], ascending=False).head(15))
    print(combine_df[combine_df.variable=='pscore'].sort_values(['value'], ascending=False).head(15))

    # get mse
    callable_loss_fxn(
        combine_df[combine_df.variable=='rscore']['value'].values, 
        combine_df[combine_df.variable=='pscore']['value'].values
    )[0]
    


    # plot histogram of predictions vs GT, could have better binning to make them more comparable
    sns.histplot(real_scores, color='b')
    sns.histplot(predicted_scores, color='orange') 



# apply model to 2022 season
nfl_season_df = pd.read_excel('../data/2022_NFL_Schedule.xlsx')
nfl_season_df = nfl_season_df.rename(columns={'home':'p_team', 'away':'OPP'})
nfl_season_df['season'] = 2022

# add features 
nfl_season_rank_df = pd.read_excel('../data/all_nflcom_historical_data.xlsx', sheet_name=None)


def check_if_should_average(col, specify_cols_strs_contains_to_not_average):
    for astr in specify_cols_strs_contains_to_not_average:
        if astr in col:
            return False
    return True

def merge_features(
    pdf, tdf,
    sheet_exclude = [],
    specific_exclude_stat = [],
    general_exclude_stat = ['Unnamed: 0', 'Team', 'season', 'Lng'],
    specify_sheet_cols_to_keep = {'special-teams_field-goals':['Team','FGM','Att','FG %','FG Blk','season']},
    specify_cols_strs_contains_to_not_average = ['%' , '/', 'Rate', 'YPC', 'Pct', 'Avg'],
    ):
    for rowi, row in pdf.iterrows():
        for matchup_side in ['p_team', 'OPP']:
            for sheet, df in tdf.items():
                if sheet in sheet_exclude: 
                    continue
                if sheet in specify_sheet_cols_to_keep.keys():
                    df = df[specify_sheet_cols_to_keep[sheet]]

                for col in df.columns:
                    
                    var_colname = f'{sheet}_{col}'
                    
                    # skip specific columns
                    if (var_colname in specific_exclude_stat):
                        continue

                    # skip general_exclude_stat cols
                    continue_flag = False
                    for gex in general_exclude_stat:
                        if gex in col:
                            continue_flag = True
                    if continue_flag: continue
                    
                    
                    # get val where team and season align to matchup
                    team_name = row[matchup_side]
                    get_loc = df[(df.Team == team_name) & (df.season == row['season'])][col]
                    assert len(get_loc) == 1
                    get_val = get_loc.values[0]

                    # if average, determine how many games played by this team, this season
                    TO_AVERAGE = check_if_should_average(col, specify_cols_strs_contains_to_not_average)
                    if TO_AVERAGE:
                        n_games = len(pdf[(pdf.p_team==team_name) & (pdf.season==row['season'])])
                        try:
                            get_val = get_val/n_games
                        except TypeError:
                            pass

                    # set val 
                    pdf_val_colname = f'{matchup_side}_{var_colname}'
                    pdf.loc[rowi, pdf_val_colname] = get_val
    return pdf

test_df = merge_features(
    nfl_season_df, nfl_season_rank_df)

preprocessor_test = Preprocessor(
    test_df,
    features=['p_team_offense_passing_Att', 'p_team_offense_passing_Cmp %', 'p_team_offense_passing_Yds/Att', 'p_team_offense_passing_TD', 'p_team_offense_passing_20+', 'p_team_offense_passing_Sck', 'p_team_offense_rushing_Att', 'p_team_offense_rushing_Rush Yds', 'p_team_offense_rushing_YPC', 'p_team_offense_rushing_Rush 1st%', 'p_team_offense_receiving_Yds/Rec', 'p_team_offense_receiving_Rec 1st%', 'p_team_offense_scoring_Tot TD', 'p_team_offense_downs_3rd Att', 'p_team_offense_downs_Scrm Plys', 'p_team_defense_passing_Att', 'p_team_defense_passing_Cmp', 'p_team_defense_passing_Cmp %', 'p_team_defense_passing_Yds/Att', 'p_team_defense_passing_Yds', 'p_team_defense_passing_TD', 'p_team_defense_passing_1st%', 'p_team_defense_passing_Sck', 'p_team_defense_rushing_Att', 'p_team_defense_rushing_Rush Yds', 'p_team_defense_rushing_YPC', 'p_team_defense_rushing_Rush 1st%', 'p_team_defense_downs_3rd Att', 'p_team_defense_downs_3rd Md', 'p_team_defense_downs_4th Att', 'p_team_defense_downs_Scrm Plys', 'p_team_defense_interceptions_INT Yds', 'p_team_special-teams_field-goals_FGM', 'p_team_special-teams_field-goals_FG %', 'p_team_special-teams_scoring_XP Pct', 'p_team_special-teams_kickoffs_TB', 'p_team_special-teams_kickoffs_Ret', 'p_team_special-teams_kickoffs_Ret Avg', 'p_team_special-teams_kickoff-returns_Avg', 'p_team_special-teams_kickoff-returns_Ret', 'p_team_special-teams_punt-returns_Avg', 'p_team_special-teams_punt-returns_Ret', 'p_team_special-teams_punt-returns_Yds', 'p_team_special-teams_punt-returns_FC', 'OPP_offense_passing_Att', 'OPP_offense_passing_Cmp %', 'OPP_offense_passing_Yds/Att', 'OPP_offense_passing_TD', 'OPP_offense_passing_20+', 'OPP_offense_passing_Sck', 'OPP_offense_rushing_Att', 'OPP_offense_rushing_Rush Yds', 'OPP_offense_rushing_YPC', 'OPP_offense_rushing_Rush 1st%', 'OPP_offense_receiving_Yds/Rec', 'OPP_offense_receiving_Rec 1st%', 'OPP_offense_scoring_Tot TD', 'OPP_offense_downs_Scrm Plys', 'OPP_defense_passing_Att', 'OPP_defense_passing_Cmp', 'OPP_defense_passing_Cmp %', 'OPP_defense_passing_Yds/Att', 'OPP_defense_passing_Yds', 'OPP_defense_passing_TD', 'OPP_defense_passing_INT', 'OPP_defense_passing_1st%', 'OPP_defense_passing_Sck', 'OPP_defense_rushing_Att', 'OPP_defense_rushing_Rush Yds', 'OPP_defense_rushing_YPC', 'OPP_defense_rushing_Rush 1st%', 'OPP_defense_downs_3rd Att', 'OPP_defense_downs_3rd Md', 'OPP_defense_downs_4th Att', 'OPP_defense_downs_Scrm Plys', 'OPP_defense_interceptions_INT Yds', 'OPP_special-teams_field-goals_FGM', 'OPP_special-teams_field-goals_FG %', 'OPP_special-teams_scoring_XP Pct', 'OPP_special-teams_kickoffs_TB', 'OPP_special-teams_kickoffs_Ret', 'OPP_special-teams_kickoffs_Ret Avg', 'OPP_special-teams_kickoff-returns_Avg', 'OPP_special-teams_kickoff-returns_Ret', 'OPP_special-teams_punt-returns_Avg', 'OPP_special-teams_punt-returns_Ret', 'OPP_special-teams_punt-returns_Yds', 'OPP_special-teams_punt-returns_FC'],
    split_mode=None,
)
pred_df = preprocessor_test.data

test_predict = model.predict(pred_df)
unscaled_predictions = unscale_prediction_array(test_predict)
predicted_scores = convert_stats_to_fantasy_score(unscaled_predictions)
test_df['PREDICTION'] = predicted_scores
test_df.to_excel('2022_1204_punter_predictions.xlsx')
best_teams = []
for dfn, dfw in test_df.groupby('week'):
    best_row = dfw.sort_values('PREDICTION', ascending=False).iloc[0]
    best_teams.append({'week':best_row.week,'team':best_row.p_team, 'projection': best_row.PREDICTION})
best_df = pd.DataFrame(best_teams)
best_df.to_excel('2022_1204_punter_predictions_best.xlsx')
