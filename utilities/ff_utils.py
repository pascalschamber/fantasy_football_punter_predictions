def get_fantasy_score_from_array():
    # implemented in train_model.py for now
    pass

def normalize_week(anint):
   return(anint/100-.01)/(0.18 - .01)

def get_punter_weekly_stats(path_to_weekly_stats=None):
    # return a df with punter weekly stats for all seasons
    import os
    import pandas as pd
    if path_to_weekly_stats == None:
        path_to_weekly_stats = os.path.join(r'..\data', 'all_seasons_punter_stats.xlsx')
    df = pd.read_excel(path_to_weekly_stats)
    df['date'] = [row.season + normalize_week(row.week) for (rowi, row) in df.iterrows()]
    return df 

def get_nflcom_team_stats(path_to_nflcom_historical_data=None):
    import os
    import pandas as pd
    if path_to_nflcom_historical_data == None:
        path_to_nflcom_historical_data = os.path.join(r'..\data', 'all_nflcom_historical_data.xlsx')
    df = pd.read_excel(path_to_nflcom_historical_data, sheet_name=None)
    return df

def print_nflcom_stat_cols():
    # pretty print columns in nflcom team stats
    tdf = get_nflcom_team_stats()
    for sheet, df in tdf.items():
        print(sheet)
        print('='*42)
        for col in df.columns:
            print('\t', col)
        print()

def merge_team_stats_onto_punter_matchup(
    path_to_weekly_stats = None,
    path_to_nflcom_historical_data = None,
    sheet_exclude = [],
    specific_exclude_stat = [],
    general_exclude_stat = ['Unnamed: 0', 'Team', 'season', 'Lng'],
    specify_sheet_cols_to_keep = {'special-teams_field-goals':['Team','FGM','Att','FG %','FG Blk','season']},
    specify_cols_strs_contains_to_not_average = ['%' , '/', 'Rate', 'YPC', 'Pct', 'Avg'],
    ):
    '''
    ````````````````````````````````````````````````````````````````````````````````````````
    Description
    ~~~~~~~~~~~~~~
        iterate through a df of weekly matchups for any length historical period
        extract team and opp, then find each teams season stats in the nflcom df
        and append these stats to the matchup, prefixing each with either p_team_ 
        or OPP_
        takes about 22 min 
    Params
    ~~~~~~~~~
        pdf: df from get_punter_weekly_stats()
        tdf: df from get_nflcom_team_stats()
        sheet_exclude: list of strings: whole sheets in nflcom stats to skip
        specific_exclude_stat: list of strings: sheet name + _ + column name to skip
        general_exclude_stat: list of strings: redundant columns that appear in nflcom data
    
    Notes
    ~~~~~~~~~~~~~~
        to average each stat by number of seasons
            most? require this, but no way to get num seasons directly from data
            some do not require it, if it contains:
                '%' , or a '/', is a rate, YPC, Pct, Avg, 
                some can be dropped outright
                    'Lng' : any 'long' such as longest pass
            some stat sheets can be mostly dropped
                special teams
                    field goals
                        every thing except: FGM, ATT, FG %
        
        to specify how many games were played
            use a general heuristic that specifies n_games by the following ranges
                {range(1990,2021):16, range(2021-2030):17}
            an issue that arises is from seasons currently being played
                for this it would makes sense to calculate how many games have been played so far
                by each team
        
    ````````````````````````````````````````````````````````````````````````````````````````
    '''
    # load punter weekly stats
    pdf = get_punter_weekly_stats(path_to_weekly_stats=path_to_weekly_stats)
    print(pdf.head())
    # load team stats
    tdf = get_nflcom_team_stats(path_to_nflcom_historical_data=path_to_nflcom_historical_data)

    # iterate through historical matchups appending team stats
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
        
    pdf.to_excel('../data/training_data_1999-2022.xlsx')
    return pdf                    

def check_if_should_average(col, specify_cols_strs_contains_to_not_average):
    for astr in specify_cols_strs_contains_to_not_average:
        if astr in col:
            return False
    return True


def convert_team_name(ateam:str, input_mode='3-letter-abrv') -> str:
    import pandas as pd
    df = pd.read_excel('../data/NFL_team-name_lookup-table.xlsx')
    input_str = ateam.lower()

    #find abrv. in df and return FullName
    if input_mode == '3-letter-abrv':
        search_cols = ['3-letter',	'nflgame_abrv',	'nflgame_abrv_alt']
        for col in search_cols:
            search_result = df[df[col].str.lower()==input_str]
            if len(search_result) == 1:
                return search_result['FullName'].values[0]
        raise IndexError(f'search for {ateam} returned {len(search_result)} results')


def inspect_play_attributes(aplay):
    ''' inspect the attributes of nflgames play object '''
    to_return = {}
    attrs = ['data',
            'desc',
            'down',
            'drive',
            'events',
            'has_player',
            'home',
            'kicking_all_yds',
            'kicking_tot',
            'kicking_touchback',
            'kicking_yds',
            'kickret_touchback',
            'note',
            'players',
            'playid',
            'team',
            'time',
            'touchdown',
            'yardline',
            'yards_togo']
    for attr in attrs:
        if hasattr(aplay, attr):
            to_return[attr] = getattr(aplay, attr)
    
    return to_return

def inspect_drive_attributes(aplay):
    to_return = {}
    drive = aplay.drive
    attrs = ['drive_num',
            'field_end',
            'field_start',
            'first_downs',
            'game',
            'home',
            'penalty_yds',
            'play_cnt',
            'plays',
            'pos_time',
            'result',
            'team',
            'time_end',
            'time_start',
            'total_yds']
    for attr in attrs:
        if hasattr(drive, attr):
            to_return[attr] = getattr(drive, attr)
    
    return to_return
    
def inspect_game_attributes(aplay):
    to_return = {}
    game = aplay.drive.game
    attrs = ['away',
            'data',
            'down',
            'drives',
            'eid',
            'game_over',
            'gamekey',
            'gcJsonAvailable',
            'home',
            'is_home',
            'loser',
            'max_player_stats',
            'nice_score',
            'playing',
            'rawData',
            'save',
            'schedule',
            'score_away',
            'score_away_q1',
            'score_away_q2',
            'score_away_q3',
            'score_away_q4',
            'score_away_q5',
            'score_home',
            'score_home_q1',
            'score_home_q2',
            'score_home_q3',
            'score_home_q4',
            'score_home_q5',
            'scores',
            'season',
            'stats_away',
            'stats_home',
            'time',
            'togo',
            'winner']
    for attr in attrs:
        if hasattr(game, attr):
            to_return[attr] = getattr(game, attr)
    
    return to_return



