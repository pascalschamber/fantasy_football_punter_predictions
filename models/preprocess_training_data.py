'''

Merge a punters game data onto own and opp team stats

'''

import pandas as pd
import os
import sys
from pathlib import Path
import numpy as np
import random
import pickle
from sklearn.preprocessing import MinMaxScaler
# sys.path.insert(0, r'..\utilities')
# import ff_utils as fu

class Preprocessor:
    def __init__(
        self, 
        merged_pdf_path, 
        NORMALIZE = True,
        features=None, targets=None, 
        split_mode='random', split_seed=398275,
        test_seasons=[2019], train_percent=0.85):
        '''
        ````````````````````````````````````````````````````````````````````````````````````````
        Description
        ~~~~~~~~~~~~~~
            
        Params
        ~~~~~~~~~
            merged_pdf_path: str: path to data
            features: list of strings: 
            targets: list of strings: 
            NORMALIZE: bool: whether to perform rescaling or not (as in feature engineering)
            split_mode: str: use None if trying to prepare data for predicting
        
        Attributes
        ~~~~~~~~~~~~~

        

        ````````````````````````````````````````````````````````````````````````````````````````
        '''
        self.merged_pdf_path = merged_pdf_path
        self.NORMALIZE = NORMALIZE
        self.features, self.targets = features, targets
        self.split_mode = split_mode
        self.split_seed = split_seed
        self.test_seasons = test_seasons
        self.train_percent = train_percent

        print('loading data')
        # self.og_merged_pdf = pd.read_excel(merged_pdf_path) # unscaled version of the data
        if isinstance(self.merged_pdf_path, str): 
            self.merged_pdf = pd.read_excel(merged_pdf_path)
        elif isinstance(self.merged_pdf_path, pd.core.frame.DataFrame):
            print('in here')
            self.merged_pdf = self.merged_pdf_path
        else:
            raise ValueError('type not recognized', type(self.merged_pdf_path))
        print('starting preprocessing')
        self.preprocess_training_data()
    
    def preprocess_training_data(self):
        ''' main pipeline call'''
        print('extracting features and targets')
        self.extract_features_targets()
        print('rescaling data')
        self.rescale_data()
        print('spliting datasets')
        self.split_data()
        print('spliting features and targets')
        self.split_features_targets()

    def get_preprocessing_parameters(self):
        ''' return a pretty print string corresponding to the parameters used to preprocess'''
        return '_-'*50 + '\nPreprocessing Parameters\n' +'_-'*30 + '\n'+ \
                f'\tTargets: {self.targets}\n' + \
                f'\tFeatures: {self.features}\n' + \
                f'\tSplit Mode: {self.split_mode}\n' + \
                f'\tSplit Seed: {self.split_seed}\n' + \
                '_-'*50 + '\n'

    def get_datasets(self):
        return self.trainX, self.trainY, self.valX, self.valY, self.testX, self.testY
    def get_dfs(self):
        return self.training_data, self.validation_data, self.testing_data

    def extract_features_targets(self):
        # extract features and target
        if self.split_mode == None:
            pass # for predictions
        elif self.targets == None:
            targets = ['AVG_DIST', 'BLK', 'IN10', 'IN20', 'PTTB', 'FC']
            self.targets = targets

        if self.features == None:
            features_st_col_index = self.merged_pdf.columns.get_loc('date')+1
            pre_features = self.merged_pdf.columns[features_st_col_index:]
            # filter out columns that are not int or float
            features = []
            for col in pre_features:
                dtype_col = self.merged_pdf[col].dtype
                if dtype_col != 'object':
                    features.append(col)
            self.features = features

    def rescale_data(self):
        # rescale the data so all discrete values are in range [0,1]
        # also replace any nan values with 0
        self.merged_pdf = self.merged_pdf.fillna(0)
        # skip rescaling (e.g. for feature engineering)
        if not self.NORMALIZE: return self.merged_pdf
        # rescale
        self.scalers = []
        self.col_names = []
        col_groups = [self.features, self.targets]
        if self.split_mode == None:
            col_groups = [self.features]
        for col_group in col_groups:
            for col in col_group:
                get_col = self.merged_pdf[col] #.fillna(0)
                get_values = get_col.values.astype('float32').reshape(-1, 1)
                scaler = MinMaxScaler(feature_range=(0,1))
                scaled = scaler.fit_transform(get_values)
                self.merged_pdf.loc[:, col] = scaled

                self.scalers.append(scaler)
                self.col_names.append(col)
        return self.merged_pdf
    
    def get_scaler(self, col_name):
        return self.scalers[self.col_names.index(col_name)]
    
    def unscale(self, col_name):
        # todo
        scaler = self.get_scaler(col_name)
        unscaled = scaler.inverse_transform(self.merged_pdf[col_name].values.reshape(-1, 1))
        return unscaled

    def split_data(self):
        '''
            mode
                'random' = split test and val into two groups that pull randomly from all seasons not it test set
                'season' = split seasons randomly and assign to each group

        '''
        if self.split_mode == None:
            return 
        # split into train, validation, and test
        testing_data = self.merged_pdf[self.merged_pdf.season.isin(self.test_seasons)]
        np.random.seed(seed=self.split_seed)
        if self.split_mode == 'season':
            all_seasons = self.merged_pdf.season.unique()[:-1]
            train_seasons = np.random.choice(all_seasons, size=7, replace=False)
            val_seasons = [season for season in all_seasons if season not in train_seasons]
            self.train_seasons, self.val_seasons = train_seasons, val_seasons
            # [2014 2010 2012 2011 2015 2016 2017] [2009, 2013, 2018]
            # extract seasons for each set
            training_data = self.merged_pdf[self.merged_pdf.season.isin(train_seasons)]
            validation_data = self.merged_pdf[self.merged_pdf.season.isin(val_seasons)]
        elif self.split_mode == 'random':
            train_val_df = self.merged_pdf[~self.merged_pdf['season'].isin(self.test_seasons)]
            training_data = train_val_df.sample(frac = self.train_percent, random_state=self.split_seed)
            validation_data = train_val_df.drop(training_data.index)
        else:
            raise ValueError(f'split mode {self.split_mode} was not recognized, must be either \'season\' or \'random\'')

        self.training_data, self.validation_data, self.testing_data = training_data, validation_data, testing_data
        return training_data, validation_data, testing_data

    def split_features_targets(self):
        # extract and convert to numpy
        if self.split_mode==None:
            self.data = self.merged_pdf[self.features].to_numpy('float32')
            return 
        
        trainX, trainY = self.training_data[self.features].to_numpy('float32'), self.training_data[self.targets].to_numpy('float32')
        valX, valY = self.validation_data[self.features].to_numpy('float32'), self.validation_data[self.targets].to_numpy('float32')
        testX, testY = self.testing_data[self.features].to_numpy('float32'), self.testing_data[self.targets].to_numpy('float32')
        self.trainX, self.trainY, self.valX, self.valY, self.testX, self.testY = trainX, trainY, valX, valY, testX, testY
        return trainX, trainY, valX, valY, testX, testY
    
    def cache(self):
        pickle_file = open('preprocesser_pickle', 'wb')
        pickle.dump(self, pickle_file)
        pickle_file.close()

if __name__ == '__main__':
    # specify path to training data if up to date
    # otherwise call func below to merge historical matchups onto team season ranks
        # merged_pdf = fu.merge_team_stats_onto_punter_matchup()
        # merged_pdf.to_excel('../data/training_data_2009-2019.xlsx', index=False)
    merged_pdf_path = '../data/training_data_2009-2019.xlsx'
    preprocessor = Preprocessor(merged_pdf_path)
    trainX, trainY, valX, valY, testX, testY = preprocessor.get_datasets()
    preprocessor.unscale('AVG_DIST')
