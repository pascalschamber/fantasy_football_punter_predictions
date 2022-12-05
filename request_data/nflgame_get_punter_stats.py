

'''
##################################################################################################

Dataset Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
in order to get 10000 samples I would need 20 seasons of data
    32 punters for 16 weeks of games = 512 samples per season
this library only offers data from 11 seasons
    2009 -> 2019

Dopes league scoreing for punters 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Punts Inside the 10 (PT10)5
    Punts Inside the 20 (PT20)3
    Blocked Punts (PTB)-4
    Touchbacks (PTTB)-2
    Fair Catches (PTFC)2
    Punt Average 44.0+ (PTA44)5
    Punt Average 42.0-43.9 (PTA42)4
    Punt Average 40.0-41.9 (PTA40)3
    Punt Average 38.0-39.9 (PTA38)2
    Punt Average 36.0-37.9 (PTA36)1
    Punt Average 33.9 or less (PTA33)-2

    Notes on scoring definitions
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Punt average 
            is gross punt distance, not final result (net)
        inside 10/20 
            is determined as being less than the number (e.g. punt must be 9 yds or less for in10)
            points earned are not mutually exclusive (e.g. earns 3+5 pts for 1 in10)
            net is taken into account (e.g. punt to 9, ret to 20 == no points)
            #TODO unsure how to count if runner fumbles, OWN recovers, and result is in10/20

Notes on data format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Player Names
        Note that NFL GameCenter formats their names like "T.Brady" and
        "W.Welker". Thus, `name` should also be in this format.
'''
import nflgame
# import ff_utils as fu 
import sys
sys.path.insert(0, r'..\utilities')
import ff_utils as fu
import pandas as pd
import os

# get punt result of a play
#################################
def get_punt_result(aplay):
    # return scoring parameters and stat tracking
    result_dict = {
        'in10':False, 
        'in20':False, 
        'is_blocked':False,#
        'is_touchback':False,
        'is_faircatch':False,
        'punt_yds':None,
        'resulting_ydl':None,
        'is_home':False,
        'p_team':None,
        'p_name':None,
        'OPP_team':None,#
        # 'play_obj':aplay,
        'desc':None,
        'season':None,
        'week':None,
        'gameid':None,
    }


    # copy over needed play attributes directly
    result_dict['p_team'], result_dict['OPP_team'] = get_teams(aplay)
    result_dict['p_name'] = get_p_name(aplay)
    result_dict['is_home'] = aplay.home
    result_dict['desc'] = aplay.desc
    result_dict['season'] = aplay.drive.game.season()
    result_dict['week'] = aplay.drive.game.schedule['week']
    result_dict['gameid'] = aplay.drive.game.gamekey

    # calculate punt metrics
    IS_OWN, punt_from_int, punt_yds, punt_ret_yds = get_punt_metrics(aplay)
    # determine final positon after all is considered
    resulting_side, resulting_fp, resulting_ydl = get_final_position(IS_OWN, punt_from_int, punt_yds, punt_ret_yds)
    
    # determine if blocked
    result_dict = is_blocked(result_dict, aplay)
    # determine if touchback, if so do not calc if in20/10 (handled in respective func)
    result_dict = is_touchback(result_dict, aplay)
    # determine if it was in side 10 or 20 or false
    result_dict = is_inside_10or20(result_dict, resulting_side, resulting_fp)
    # determine if fair catch
    result_dict = is_faircatch(result_dict, aplay)

    # append non-calc stats
    result_dict['punt_yds'] = punt_yds
    result_dict['resulting_ydl'] = resulting_ydl


    return result_dict


# assisting functions
################################################################
# calculate final position
def get_final_position(IS_OWN, punt_from_int, punt_yds, punt_ret_yds):
    if IS_OWN:
        resulting_fp = (punt_from_int + punt_yds - punt_ret_yds)
        resulting_side = 'OPP' if resulting_fp >50 else 'OWN'

    else:
        resulting_fp = (punt_from_int - punt_yds + punt_ret_yds)
        resulting_side = 'OWN' if resulting_fp >50 else 'OPP'

    if resulting_fp >50:
        resulting_fp = 100 - resulting_fp

    resulting_ydl = f'{resulting_side} {resulting_fp}'

    return (resulting_side, resulting_fp, resulting_ydl)

def filter_plays(plays_list):
    return [play for play in plays_list if play_is_punt(play)]

def play_is_punt(aplay):
    events = aplay.events
    for ev in events:
        if 'punting_tot' in ev:
            if ev['punting_tot'] == 1:
                return True
    return False
    

# punt metric determiners
################################################################
def update_result_dict():
    ''' calls all below methods to update the return results '''
    pass

def get_punt_metrics(aplay):
    punt_from = str(aplay.yardline)
    IS_OWN = True if (punt_from.startswith('OWN') or punt_from=='MIDFIELD') else False
    if punt_from == 'MIDFIELD':
        punt_from_int = 50
    else:
        punt_from_int = int(punt_from.split(' ')[1])
    punt_yds = int(aplay.punting_yds)
    punt_ret_yds = int(aplay.puntret_yds)
    punt_net = punt_yds - punt_ret_yds
    return (IS_OWN, punt_from_int, punt_yds, punt_ret_yds)

def is_inside_10or20(result_dict, resulting_side, resulting_fp):
    if result_dict['is_touchback'] == True: # do not count in10/20
        return result_dict
    if resulting_side == 'OPP':
        if resulting_fp < 20:
            result_dict['in20'] = True
        if resulting_fp < 10:
            result_dict['in10'] = True
    return result_dict

def is_blocked(result_dict, aplay):
    blocked_keywords = ['deflected', 'blocked', ' hand on']
    for bkw in blocked_keywords:
        if bkw in aplay.desc.lower():
            result_dict['is_blocked'] = True
    return result_dict

def is_(aplay, attr, result_key, result_dict):
    if hasattr(aplay, attr):
        if getattr(aplay, attr) == 1:
            result_dict[result_key] = True
    return result_dict
def is_touchback(result_dict, aplay):
    return is_(aplay, 'punting_touchback', 'is_touchback', result_dict)
def is_faircatch(result_dict, aplay):
    return is_(aplay, 'puntret_fair', 'is_faircatch', result_dict)

def get_teams(aplay):
    game = aplay.drive.game
    if aplay.home == True:
        OPP_team = game.away
    else:
        OPP_team = game.home
    return (fu.convert_team_name(aplay.team), fu.convert_team_name(OPP_team))

def get_p_name(aplay):
    events = aplay.events
    for ev in events:
        if 'punting_yds' in ev:
            return ev['playername']
    return 'noneFound'

def get_no_stat_dict(ateam, opp_team_df):
    return {
            'p_name' : None,
            'p_team' : ateam,
            'OPP' : opp_team_df['p_team'].values[0],
            'FINAL_SCORE' : None,
            'PROJECTION' : None,
            'P_TOT_PTS' : -2,
            'N_punts' : 0,
            'AVG_DIST' :0 ,
            'BLK' : 0,
            'IN10': 0,
            'IN20' : 0,
            'PTTB' : 0,
            'FC': 0,
            'game_status' : 'away' if opp_team_df['is_home'].values[0] == True else 'home',
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

def season_to_xlsx(results):
    # condense play-by-play to game summary
    
    out_dicts = []
    df = pd.DataFrame(results)
    ugameids = df['gameid'].unique()
    for gameid in ugameids[:]:
        agame = df[df['gameid'] == gameid]
        NO_HOME_STATS = False
        NO_AWAY_STATS = False
        if len(agame[agame['is_home'] == True]['p_team']) > 0:
            home = agame[agame['is_home'] == True]['p_team'].values[0]
        else:
            home = agame[agame['is_home'] == False]['OPP_team'].values[0]
            NO_HOME_STATS = True
        if len(agame[agame['is_home'] == False]['p_team']) > 0:
            away = agame[agame['is_home'] == False]['p_team'].values[0]
        else:
            away = agame[agame['is_home'] == True]['OPP_team'].values[0]
            NO_AWAY_STATS = True

        for teami, ateam in enumerate([home, away]):
            if teami == 0 and NO_HOME_STATS == True:
                out_dict = get_no_stat_dict(ateam, agame[agame['p_team'] == away])
            elif teami == 1 and NO_AWAY_STATS == True:
                out_dict = get_no_stat_dict(ateam, agame[agame['p_team'] == home])
            else:
                teamdf = agame[agame['p_team'] == ateam]
                # sum stats
                in10 = teamdf['in10'].sum()
                in20 = teamdf['in20'].sum()
                blks = teamdf['is_blocked'].sum()
                pttb = teamdf['is_touchback'].sum()
                fair = teamdf['is_faircatch'].sum()
                punt_yds_avg = teamdf['punt_yds'].mean()
                fantasy_score = (5*in10) + (3*in20) + (-4*blks) + (-2*pttb) + (2*fair) + (get_punt_yds_score(punt_yds_avg))

                out_dict = {
                    'p_name' : teamdf['p_name'].values[0],
                    'p_team' : teamdf['p_team'].values[0],
                    'OPP' : teamdf['OPP_team'].values[0],
                    'FINAL_SCORE' : None,
                    'PROJECTION' : None,
                    'P_TOT_PTS' : fantasy_score,
                    'N_punts' : len(teamdf),
                    'AVG_DIST' :punt_yds_avg ,
                    'BLK' : blks,
                    'IN10' : in10,
                    'IN20' : in20,
                    'PTTB' : pttb,
                    'FC': fair,
                    'game_status' : 'home' if teamdf['is_home'].values[0] == True else 'away',
                    'week' : teamdf['week'].values[0],
                    'season' : teamdf['season'].values[0],
                }
            out_dicts.append(out_dict)

    return out_dicts
    

def get_season_results(ayear):
    games = nflgame.games(ayear)
    plays = nflgame.combine_plays(games)
    plays_list = list(plays.sort('punting_yds', descending=False))
    plays_list_filter = filter_plays(plays_list)
    print(f'season: {ayear}')
    print(f'filtered plays: {len(plays_list_filter)}, pre-filter num: {len(plays_list)}')
    results = season_to_xlsx([get_punt_result(p) for p in plays_list_filter])
    return results

def get_all_seasons(date_range=(2009,2020)):
    all_seasons = []
    for year in range(*date_range):
        all_seasons.extend(get_season_results(year))

    out_df = pd.DataFrame(all_seasons)
    out_df.to_excel('../data/all_seasons_punter_stats.xlsx', index=False)
    return out_df
    


if __name__ == '__main__':
    get_all_seasons()
    fu.merge_team_stats_onto_punter_matchup(
        path_to_weekly_stats = os.path.join(r'..\data', 'all_seasons_punter_stats.xlsx'),
        path_to_nflcom_historical_data = os.path.join(r'..\data', 'all_nflcom_historical_data.xlsx'),
        sheet_exclude = [],
        specific_exclude_stat = [],
        general_exclude_stat = ['Unnamed: 0', 'Team', 'season'],
    )










