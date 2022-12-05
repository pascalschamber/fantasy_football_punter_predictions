import os
import pandas as pd
import numpy as np
from pathlib import Path

def convert_team_name(ateam, input_mode='3-letter-abrv'):
    input_str = ateam.lower()
    #find abrv. in df and return FullName
    if input_mode == '3-letter-abrv':
        search_cols = ['3-letter',	'nflgame_abrv',	'nflgame_abrv_alt']
        for col in search_cols:
            search_result = map_df[map_df[col].str.lower()==input_str]
            if len(search_result) == 1:
                return search_result['FullName'].values[0]
        raise IndexError(f'search for {ateam} returned {len(search_result)} results')

def fix_abrvs(iterable):
    fix_abrvs_dict = {
        'OAK':'LV',
        'SD':'LAC',
        'STL':'LA',
        }
    for abrv, replace_with in fix_abrvs_dict.items():
        if abrv in iterable:
            iterable[iterable.index(abrv)] = replace_with
    return iterable

def filter_plays(df, filter_dict=None):
    ''' filter out non regular season games, non-punts'''
    if filter_dict == None:
        filter_dict = {
            'season_type':'REG',
            'play_type':'punt'
        }

    for col, val in filter_dict.items():
        df = df[df[col] == val]
    
    return df
    

def filter_columns(df, cols_to_keep_raw=None):
    if cols_to_keep_raw == None:
        cols_to_keep_raw = 'play_id	game_id	old_game_id	home_team	away_team	season_type	week	posteam	posteam_type	defteam	side_of_field	yardline_100	game_date	time	yrdln	ydstogo	ydsnet	desc	kick_distance	punt_blocked	touchback	punt_inside_twenty	punt_in_endzone	punt_out_of_bounds	punt_downed	punt_fair_catch	punter_player_id	punter_player_name	return_yards	season	drive_end_transition	drive_game_clock_start	drive_game_clock_end	drive_start_yard_line	drive_end_yard_line	drive_play_id_started	drive_play_id_ended	away_score	home_score	result	total	spread_line	total_line	roof	surface	out_of_bounds'
    cols_to_keep = cols_to_keep_raw.split('\t')
    return df[cols_to_keep]
    

def num_inside_10(teamdf):
    in20df = teamdf[teamdf.punt_inside_twenty == 1]
    if len(in20df) == 0:
        return 0

    num_in10 = 0
    for rowi, row in in20df.iterrows():
        punt_from = row['yardline_100']
        kick_dist = row['kick_distance']
        ret_yds = row['return_yards']
        result_ydl = punt_from - kick_dist + ret_yds
        if result_ydl < 10:
            num_in10 += 1
    return num_in10

def extract_team_punting_data(teams_in_game, team_name, teamdf):
    # sum stats
    in10 = num_inside_10(teamdf)
    in20 = teamdf['punt_inside_twenty'].sum()
    blks = teamdf['punt_blocked'].sum()
    pttb = teamdf['touchback'].sum()
    fair = teamdf['punt_fair_catch'].sum()

    # fix punt yds that may result as nan after taking mean
    # set kick distance when touchback as yardline 100
    for rowi, row in teamdf.iterrows():
        if row['touchback'] == 1:
            teamdf.loc[rowi, 'kick_distance'] = float(row['yardline_100'])
    punt_yds_avg = 0
    punt_yds_temp = teamdf['kick_distance'].mean()
    if punt_yds_temp > punt_yds_avg:
        punt_yds_avg = punt_yds_temp
        
    # extract fantasy scoring
    fantasy_score = (3*in20) + (-4*blks) + (-2*pttb) + (2*fair) + (get_punt_yds_score(punt_yds_avg))  + (5*in10)

    # generate out dict which becomes a row in outdf
    try:
        out_dict = {
                'p_name' : teamdf['punter_player_name'].values[0],
                'p_team' : team_name,
                'OPP' : teams_in_game[1] if teams_in_game.index(team_name) == 0 else teams_in_game[0],
                'FINAL_SCORE' : f'{teamdf.home_score.values[0]} - {teamdf.away_score.values[0]}',
                'PROJECTION' : None,
                'P_TOT_PTS' : fantasy_score,
                'N_punts' : len(teamdf),
                'AVG_DIST' :punt_yds_avg ,
                'BLK' : blks,
                'IN10' : in10,
                'IN20' : in20,
                'PTTB' : pttb,
                'FC': fair,
                'game_status' : 'home' if teamdf.home_team.values[0] == team_name else 'away',
                'week' : teamdf['week'].values[0],
                'season' : teamdf['season'].values[0],
            }
    except:
        raise ValueError(teams_in_game, team_name)

    return out_dict


def get_no_stat_dict(ateam, opp_team, opp_team_df):
    return {
            'p_name' : None,
            'p_team' : ateam,
            'OPP' : opp_team,
            'FINAL_SCORE' : f'{opp_team_df.home_score.values[0]} - {opp_team_df.away_score.values[0]}',
            'PROJECTION' : None,
            'P_TOT_PTS' : -2,
            'N_punts' : 0,
            'AVG_DIST' :0 ,
            'BLK' : 0,
            'IN10': 0,
            'IN20' : 0,
            'PTTB' : 0,
            'FC': 0,
            'game_status' : 'away' if opp_team_df.home_team.values[0] == opp_team else 'home',
            'week' : opp_team_df['week'].values[0],
            'season' : opp_team_df['season'].values[0],
        }


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


def get_games_with_no_punts(ogdf, df):
    # get games with no punts
    og_all_games = ogdf.game_id.unique() # if len != len at end, some games must have had no punts
    df_all_games = df.game_id.unique()
    no_punt_games = set(og_all_games) - set(df_all_games)
    if len(no_punt_games) != 0:
        # append an empty dict for each punter 
        for no_punt_game_id in no_punt_games:
            nop_df = ogdf[ogdf.game_id == no_punt_game_id]
            if len(nop_df) == 0:
                print(no_punt_game_id)
            nop_teams = fix_abrvs(no_punt_game_id.split('_')[2:])
            for team in nop_teams:
                opp_team = nop_teams[1] if nop_teams.index(team) == 0 else nop_teams[0]
                opp_team_df = nop_df[nop_df.posteam == opp_team]
                out_dict = get_no_stat_dict(team, opp_team, opp_team_df)
                out_dicts.append(out_dict)


def check_if_one_team_doesnt_punt(game_name, team_dfs):    
    NO_STATS = False # insert blank dict
    NO_STAT_TEAM = None
    if len(team_dfs.groups) == 1: 
        NO_STATS = True
        possible_teams = fix_abrvs(game_name.split('_')[2:])
        have_this_team = list(team_dfs.groups.keys())[0]
        possible_teams.remove(have_this_team)
        NO_STAT_TEAM = possible_teams[0]
        
    elif len(team_dfs.groups) == 2:
        pass # normal
    else:
        raise ValueError
    
    return NO_STATS, NO_STAT_TEAM




'''
#######################################################################################
'''

dir_path = Path(r'C:\Users\pasca\Desktop\FF_PUNTERS\data\nflfastR_play_by_play')
csv_paths = sorted([os.path.join(dir_path, c) for c in os.listdir(dir_path) if c.endswith('.csv')])

out_dicts = []
for ip, p in enumerate(csv_paths[-1:]):

    ogdf = pd.read_csv(p)
    
    df = filter_columns(filter_plays(ogdf))
    # df.to_csv(r'C:\Users\pasca\Desktop\FF_PUNTERS\request_data\nflfastR_test.csv')

    # get games with no punts
    get_games_with_no_punts(filter_plays(ogdf,filter_dict = {'season_type':'REG'}), df)
    subset = df[(df['week'] == 12.0) & (df.posteam=='BAL')]
    break
    # extract output df for all games
    dfs = df.groupby('game_id')
    for game_name, gamedf in dfs:
        team_dfs = gamedf.groupby('posteam')
        teams_in_game = fix_abrvs(game_name.split('_')[2:])
        
        

        # extract punts for each team
        for team_name, teamdf in team_dfs:
            out_dict = extract_team_punting_data(teams_in_game, team_name, teamdf)
            out_dicts.append(out_dict)

        # generate an empty dict if no punts for one team
        # check for games where one team didn't punt
        NO_STATS, NO_STAT_TEAM = check_if_one_team_doesnt_punt(game_name, team_dfs)

        if NO_STATS:
            out_dict = get_no_stat_dict(NO_STAT_TEAM, team_name, teamdf)
            out_dicts.append(out_dict)

if bool(0):  
    out_df = pd.DataFrame(out_dicts)

    # map team names to full names
    map_df = pd.read_excel('../data/NFL_team-name_lookup-table.xlsx')

    out_df.p_team = out_df.p_team.map(convert_team_name)
    out_df.OPP = out_df.OPP.map(convert_team_name)
if bool(0): 
    out_df.to_excel('../data/all_seasons_punter_stats.xlsx', index=False)
