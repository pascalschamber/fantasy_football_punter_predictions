
import pandas as pd
import os
import sys
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np # ultimately should revert to version 1.14.0 to silence tensorflow compatibility warnings
from datetime import datetime
import seaborn as sns
import pickle
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

from preprocess_training_data import Preprocessor
sys.path.insert(0, r'..\utilities')
import ff_utils as fu



def get_preprocessor(UPDATE_PREPROCESSOR, NORMALIZE=True, feature_set='high_var_and_low_corr'):
    if UPDATE_PREPROCESSOR:
        # get preprocessor and pickle it
        merged_pdf_path = '../data/training_data_1999-2022.xlsx'
        # notes on feature sets below
        feature_sets = {
            'only_high_var': ['p_team_offense_passing_Cmp', 'p_team_offense_passing_Cmp %', 'p_team_offense_passing_Yds/Att', 'p_team_offense_passing_Pass Yds', 'p_team_offense_passing_TD', 'p_team_offense_passing_Rate', 'p_team_offense_passing_1st', 'p_team_offense_passing_1st%', 'p_team_offense_passing_20+', 'p_team_offense_passing_Sck', 'p_team_offense_passing_SckY', 'p_team_offense_rushing_Rush 1st', 'p_team_offense_rushing_Rush 1st%', 'p_team_offense_receiving_Rec', 'p_team_offense_receiving_Yds', 'p_team_offense_receiving_TD', 'p_team_offense_receiving_Rec 1st', 'p_team_offense_receiving_Rec 1st%', 'p_team_offense_scoring_Rec TD', 'p_team_offense_downs_Rec 1st', 'p_team_offense_downs_Rec 1st%', 'p_team_offense_downs_Rush 1st', 'p_team_offense_downs_Rush 1st%', 'p_team_defense_passing_Att', 'p_team_defense_rushing_Att', 'p_team_defense_rushing_Rush 1st', 'p_team_defense_rushing_Rush 1st%', 'p_team_defense_downs_Rush 1st', 'p_team_defense_downs_Rush 1st%', 'p_team_defense_downs_Scrm Plys', 'p_team_special-teams_field-goals_FGM', 'p_team_special-teams_field-goals_FG %', 'p_team_special-teams_scoring_FGM', 'p_team_special-teams_scoring_FG %', 'p_team_special-teams_punting_Cmp', 'p_team_special-teams_punting_Cmp %', 'p_team_special-teams_punting_Yds/Att', 'p_team_special-teams_punting_Pass Yds', 'p_team_special-teams_punting_TD', 'p_team_special-teams_punting_Rate', 'p_team_special-teams_punting_1st', 'p_team_special-teams_punting_1st%', 'p_team_special-teams_punting_20+', 'p_team_special-teams_punting_Sck', 'p_team_special-teams_punting_SckY', 'OPP_offense_passing_Att', 'OPP_offense_passing_Cmp', 'OPP_offense_passing_Cmp %', 'OPP_offense_passing_Yds/Att', 'OPP_offense_passing_Pass Yds', 'OPP_offense_passing_TD', 'OPP_offense_passing_Rate', 'OPP_offense_passing_1st', 'OPP_offense_passing_1st%', 'OPP_offense_passing_20+', 'OPP_offense_passing_Sck', 'OPP_offense_passing_SckY', 'OPP_offense_rushing_Rush 1st', 'OPP_offense_rushing_Rush 1st%', 'OPP_offense_receiving_Rec', 'OPP_offense_receiving_Yds', 'OPP_offense_receiving_TD', 'OPP_offense_receiving_Rec 1st', 'OPP_offense_receiving_Rec 1st%', 'OPP_offense_scoring_Rec TD', 'OPP_offense_downs_Rec 1st', 'OPP_offense_downs_Rec 1st%', 'OPP_offense_downs_Rush 1st', 'OPP_offense_downs_Rush 1st%', 'OPP_defense_rushing_Rush 1st', 'OPP_defense_rushing_Rush 1st%', 'OPP_defense_downs_Rush 1st', 'OPP_defense_downs_Rush 1st%', 'OPP_special-teams_field-goals_FGM', 'OPP_special-teams_field-goals_FG %', 'OPP_special-teams_scoring_FGM', 'OPP_special-teams_scoring_FG %', 'OPP_special-teams_punting_Att', 'OPP_special-teams_punting_Cmp', 'OPP_special-teams_punting_Cmp %', 'OPP_special-teams_punting_Yds/Att', 'OPP_special-teams_punting_Pass Yds', 'OPP_special-teams_punting_TD', 'OPP_special-teams_punting_Rate', 'OPP_special-teams_punting_1st', 'OPP_special-teams_punting_1st%', 'OPP_special-teams_punting_20+', 'OPP_special-teams_punting_Sck', 'OPP_special-teams_punting_SckY'],
            'high_var_and_low_corr': ['p_team_offense_passing_Att', 'p_team_offense_passing_Cmp %', 'p_team_offense_passing_Yds/Att', 'p_team_offense_passing_TD', 'p_team_offense_passing_20+', 'p_team_offense_passing_Sck', 'p_team_offense_rushing_Att', 'p_team_offense_rushing_Rush Yds', 'p_team_offense_rushing_YPC', 'p_team_offense_rushing_Rush 1st%', 'p_team_offense_receiving_Yds/Rec', 'p_team_offense_receiving_Rec 1st%', 'p_team_offense_scoring_Tot TD', 'p_team_offense_downs_3rd Att', 'p_team_offense_downs_Scrm Plys', 'p_team_defense_passing_Att', 'p_team_defense_passing_Cmp', 'p_team_defense_passing_Cmp %', 'p_team_defense_passing_Yds/Att', 'p_team_defense_passing_Yds', 'p_team_defense_passing_TD', 'p_team_defense_passing_1st%', 'p_team_defense_passing_Sck', 'p_team_defense_rushing_Att', 'p_team_defense_rushing_Rush Yds', 'p_team_defense_rushing_YPC', 'p_team_defense_rushing_Rush 1st%', 'p_team_defense_downs_3rd Att', 'p_team_defense_downs_3rd Md', 'p_team_defense_downs_4th Att', 'p_team_defense_downs_Scrm Plys', 'p_team_defense_interceptions_INT Yds', 'p_team_special-teams_field-goals_FGM', 'p_team_special-teams_field-goals_FG %', 'p_team_special-teams_scoring_XP Pct', 'p_team_special-teams_kickoffs_TB', 'p_team_special-teams_kickoffs_Ret', 'p_team_special-teams_kickoffs_Ret Avg', 'p_team_special-teams_kickoff-returns_Avg', 'p_team_special-teams_kickoff-returns_Ret', 'p_team_special-teams_punt-returns_Avg', 'p_team_special-teams_punt-returns_Ret', 'p_team_special-teams_punt-returns_Yds', 'p_team_special-teams_punt-returns_FC', 'OPP_offense_passing_Att', 'OPP_offense_passing_Cmp %', 'OPP_offense_passing_Yds/Att', 'OPP_offense_passing_TD', 'OPP_offense_passing_20+', 'OPP_offense_passing_Sck', 'OPP_offense_rushing_Att', 'OPP_offense_rushing_Rush Yds', 'OPP_offense_rushing_YPC', 'OPP_offense_rushing_Rush 1st%', 'OPP_offense_receiving_Yds/Rec', 'OPP_offense_receiving_Rec 1st%', 'OPP_offense_scoring_Tot TD', 'OPP_offense_downs_Scrm Plys', 'OPP_defense_passing_Att', 'OPP_defense_passing_Cmp', 'OPP_defense_passing_Cmp %', 'OPP_defense_passing_Yds/Att', 'OPP_defense_passing_Yds', 'OPP_defense_passing_TD', 'OPP_defense_passing_INT', 'OPP_defense_passing_1st%', 'OPP_defense_passing_Sck', 'OPP_defense_rushing_Att', 'OPP_defense_rushing_Rush Yds', 'OPP_defense_rushing_YPC', 'OPP_defense_rushing_Rush 1st%', 'OPP_defense_downs_3rd Att', 'OPP_defense_downs_3rd Md', 'OPP_defense_downs_4th Att', 'OPP_defense_downs_Scrm Plys', 'OPP_defense_interceptions_INT Yds', 'OPP_special-teams_field-goals_FGM', 'OPP_special-teams_field-goals_FG %', 'OPP_special-teams_scoring_XP Pct', 'OPP_special-teams_kickoffs_TB', 'OPP_special-teams_kickoffs_Ret', 'OPP_special-teams_kickoffs_Ret Avg', 'OPP_special-teams_kickoff-returns_Avg', 'OPP_special-teams_kickoff-returns_Ret', 'OPP_special-teams_punt-returns_Avg', 'OPP_special-teams_punt-returns_Ret', 'OPP_special-teams_punt-returns_Yds', 'OPP_special-teams_punt-returns_FC'],
            'rfe_high_var_low_corr': ['p_team_offense_passing_Att', 'p_team_offense_passing_Cmp %', 'p_team_offense_passing_Yds/Att', 'p_team_offense_passing_TD', 'p_team_offense_passing_20+', 'p_team_offense_passing_Sck', 'p_team_offense_rushing_Att', 'p_team_offense_rushing_YPC', 'p_team_offense_receiving_Yds/Rec', 'p_team_offense_scoring_Tot TD', 'p_team_offense_downs_Scrm Plys', 'p_team_defense_passing_Yds/Att', 'p_team_defense_passing_TD', 'p_team_defense_passing_Sck', 'p_team_defense_rushing_YPC', 'p_team_defense_downs_4th Att', 'p_team_special-teams_field-goals_FGM', 'p_team_special-teams_kickoffs_TB', 'p_team_special-teams_kickoffs_Ret', 'p_team_special-teams_kickoff-returns_Ret', 'p_team_special-teams_punt-returns_Avg', 'p_team_special-teams_punt-returns_Ret', 'p_team_special-teams_punt-returns_Yds', 'p_team_special-teams_punt-returns_FC', 'OPP_offense_passing_Att', 'OPP_offense_passing_Cmp %', 'OPP_offense_passing_Yds/Att', 'OPP_offense_passing_20+', 'OPP_offense_passing_Sck', 'OPP_offense_rushing_Att', 'OPP_offense_rushing_YPC', 'OPP_offense_receiving_Yds/Rec', 'OPP_offense_scoring_Tot TD', 'OPP_offense_downs_Scrm Plys', 'OPP_defense_passing_INT', 'OPP_defense_downs_3rd Att', 'OPP_defense_downs_3rd Md', 'OPP_defense_downs_4th Att', 'OPP_special-teams_field-goals_FGM', 'OPP_special-teams_kickoffs_TB', 'OPP_special-teams_kickoffs_Ret', 'OPP_special-teams_kickoff-returns_Ret', 'OPP_special-teams_punt-returns_Ret', 'OPP_special-teams_punt-returns_FC'], 
        }

        preprocessor = Preprocessor(
                merged_pdf_path,
                NORMALIZE = NORMALIZE,
                split_mode='random',
                features = feature_sets[feature_set],
        )
        preprocessor.cache()
    else:
        #read the pickle file
        picklefile = open('preprocesser_pickle', 'rb')
        preprocessor = pickle.load(picklefile)
        picklefile.close()

    return preprocessor

'''
Feature selection 
'''
def get_low_variance_features(df, all_features):
    df_features = df[all_features]
    print('features before low variance threshold', df_features.shape)
    var_thresh = VarianceThreshold(threshold=0.1)
    transformed_data = var_thresh.fit_transform(df_features)
    low_var_features = list(set(var_thresh.feature_names_in_) - set(var_thresh.get_feature_names_out()))
    resulting_features_df = df_features.loc[:, ~df_features.columns.isin(low_var_features)]
    # potentially unneeded computations
    # high_var_features = var_thresh.get_feature_names_out()
    # applied_to_og_df = df.loc[:, ~df.columns.isin(low_var_features)]
    print('features after low variance threshold', transformed_data.shape, '\n')
    return resulting_features_df, low_var_features

def get_high_corr_features(df, thresh=0.85, use_abs=False):
    corr = df.corr()
    thresh = 0.85
    high_corr_cols = set()
    print('features before high corr removal', df.shape)
    for i in range(len(corr.columns)):
        for j in range(i): # iter series   
            val = corr.iloc[i,j]
            if use_abs: 
                val = abs(val)
            # apply threshold
            if val > thresh:
                this_col = corr.columns[i]
                high_corr_cols.add(this_col)
    resulting_cols = set(df.columns) - high_corr_cols
    df = df.loc[:, ~df.columns.isin(high_corr_cols)]
    print('features after high corr removal', df.shape, '\n')

    return df, high_corr_cols

def plot_corr(df):
    corr = df.corr()
    fig,ax = plt.subplots(figsize=(30,30))
    ax.set_title('Correlation Matrix', fontsize=16)
    ax.matshow(corr)
    ax.set_xticklabels(list(corr.index), rotation=-90)
    ax.set_yticklabels(list(corr.index))
    # color bar
    PCM=ax.get_children()[0] #get the mappable, the 1st and the 2nd are the x and y axes
    cb = plt.colorbar(PCM, ax=ax)
    cb.ax.tick_params(labelsize=14)
    plt.show()

def recursive_feature_elimination(X,Y):
    print('features before rfe removal', X[0].shape)
    # initialize the model
    model = LinearRegression()
    # initialize RFE
    rfe = RFE(
        estimator=model,
    )
    # fit RFE
    rfe.fit(X, Y)
    # get the transformed data with
    # selected columns
    X_transformed = rfe.transform(X)

    most_important_features = [col_name for (col_name, mask) in zip(df.columns, rfe.support_) if mask==True]
    feature_ranks = rfe.ranking_
    print('features after rfe removal', X_transformed.shape, '\n')

    return most_important_features

'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main

Model decicions
    Cross Validation Sceme
        k-fold cross-validation
            lots of examples
            normal distribution (non-skewed)
            comparable season to season
            these reasons, choose this over stratified
    
    Evaluation metrics
        mean-squared error 
            12/4 produced the lowest error when calculating the error summed over the whole season for each team
    
    Feature selection
        too many features == curse of dimensionality
            remove features with low variance
            remove features with high correlation
                # since there are so many features not sure how best to remove those with high correlation
                # also just feel that we can't throw away variable just becuase they happen to correlate
            perform low impact features with RFE
        
        notes
        ~~~~~~~~~~~~~~~
        removing low var and high corr best improved test season sum error
            better than adding rfe, though rfe lowered training error

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
if __name__ == '__main__':
    # if updating preprocessor change to true, else just load it
    UPDATE_PREPROCESSOR = bool(0)
    preprocessor = get_preprocessor(UPDATE_PREPROCESSOR, NORMALIZE=False)
    trainX, trainY, valX, valY, testX, testY = preprocessor.get_datasets()
    training_data, validation_data, testing_data = preprocessor.get_dfs()
    df = training_data.copy()
    
    removed_features = set()
    ###############################################################
    # removing features (doing this on non-scaled features, as normalization prevents analysis)
    ###############################################################
    # low variance
    df, low_var_features = get_low_variance_features(df, preprocessor.features)
    removed_features.update(low_var_features)
    # # high correlation
    df, high_corr_features = get_high_corr_features(df, thresh=0.85)
    removed_features.update(high_corr_features)

    # greedy feature selection
    ###########################################################################
    # recursive feature elimination
    X,Y = preprocessor.merged_pdf[df.columns].to_numpy(), preprocessor.merged_pdf[preprocessor.targets].to_numpy()
    most_important_features = recursive_feature_elimination(X,Y)
    




    
        