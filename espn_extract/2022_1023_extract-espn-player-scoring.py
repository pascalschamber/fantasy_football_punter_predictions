import os
import pandas as pd
import numpy as np
from pathlib import Path


# punter weekly stats
input_dir = r'C:\Users\pasca\Desktop\2022_FF_Punters'
input_fn = '2022_PunterWeeklyStats.xlsx'
fn = os.path.join(input_dir, input_fn)
out_path = os.path.join(input_dir, Path(input_fn).stem + '_extracted' + '.xlsx')
df_sheets = pd.read_excel(fn, sheet_name=None)

# team ranks
input_fn2 = 'TeamRanksOFF-DEF.xlsx'
fn2 = os.path.join(input_dir, input_fn2)
team_sheets = pd.read_excel(fn2, sheet_name=None)
team_sheet = team_sheets[list(team_sheets.keys())[-1]]



all_dfs = []
extract_sheets = 'P_wk'
for sheet in df_sheets:
    if extract_sheets in sheet:
        df = df_sheets[sheet]
        week_i = int(sheet[-1])
        nfl_week_str = f'NFL WEEK {week_i}'

        # split on 151, 203
        dfpt1 = df[df.index<151]
        dfpt2 = df[(df.index>=151) & (df.index<203)]
        dfpt3 = df[df.index>=203]

        # format 2 and 3
        df2_extract_cols = ['PUNTERS',	'STATUS',	'Unnamed: 2',	nfl_week_str,	'Unnamed: 4']
        df2_rename_cols = ['N_punts', 'AVG_DIST', 'BLK', 'IN20', 'PTTB']
        df2_rename_dict = dict(zip(df2_extract_cols,df2_rename_cols))
        df2 = dfpt2[df2_extract_cols]
        df2.rename(columns=df2_rename_dict, inplace=True)
        df2 = df2[df2.index>152]

        df3 = dfpt3['PUNTERS']
        df3 = df3[df3.index>204]
        df3 = df3.rename('P_TOT_PTS')

        # format 1
        df1_extract_cols = ['PUNTERS',nfl_week_str,'Unnamed: 4',	'Unnamed: 5',]
        df1_rename_cols = ['Name','OPP','FINAL_SCORE','PROJECTION']
        df1_rename_dict = dict(zip(df1_extract_cols,df1_rename_cols))
        df1 = dfpt1[df1_extract_cols]
        df1.rename(columns=df1_rename_dict, inplace=True)
        df1 = df1[df1.index>0]

        groups = df1.groupby(np.arange(len(df1.index))//3)
        extracted_dicts = []
        for group_p in groups:
            extract_cols = ['p_name', 'p_team', 'OPP', 'FINAL_SCORE', 'PROJECTION']
            extract_dict = dict(zip(extract_cols, [{}]*5))

            extract_dict['p_name'] = group_p[1]['Name'].to_list()[1]
            extract_dict['p_team'] = group_p[1]['Name'].to_list()[2]
            extract_dict['OPP'] = group_p[1]['OPP'].to_list()[0]
            extract_dict['FINAL_SCORE'] = group_p[1]['FINAL_SCORE'].to_list()[0]
            extract_dict['PROJECTION'] = group_p[1]['PROJECTION'].to_list()[0]
            extracted_dicts.append(extract_dict)
        
        df1_extracted = pd.DataFrame(extracted_dicts)

        # get all dfs to same length and merge
        print(len(df1_extracted),len(df2), len(df3))
        df1_extracted['P_TOT_PTS'] = df3.to_list()
        for df2_col in df2_rename_cols:
            df1_extracted[df2_col] = df2[df2_col].to_list()
        print(df1_extracted.head())

        # drop empty rows
        df1_extracted = df1_extracted[df1_extracted['P_TOT_PTS'] != '--']

        # def get opponent, convert to full name, extract home or away
        raw_OPPS = df1_extracted['OPP'].to_list()
        game_status = ['away' if '@' in el else 'home' for el in raw_OPPS]
        opps_short = [el if el.find('@') == -1 else el[1:] for el in raw_OPPS]
        opps_short_to_full_dict = dict(zip(
            ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 'MIA', 'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'LV', 'PHI', 'PIT', 'LAC', 'SF', 'SEA', 'LAR', 'TB', 'TEN', 'WSH'],
            ['Arizona Cardinals', 'Atlanta Falcons', 'Baltimore Ravens', 'Buffalo Bills', 'Carolina Panthers', 'Chicago Bears', 'Cincinnati Bengals', 'Cleveland Browns', 'Dallas Cowboys', 'Denver Broncos', 'Detroit Lions', 'Green Bay Packers', 'Houston Texans', 'Indianapolis Colts', 'Jacksonville Jaguars', 'Kansas City Chiefs', 'Miami Dolphins', 'Minnesota Vikings', 'New England Patriots', 'New Orleans Saints', 'New York Giants', 'New York Jets', 'Las Vegas Raiders', 'Philadelphia Eagles', 'Pittsburgh Steelers', 'Los Angeles Chargers', 'San Francisco 49ers', 'Seattle Seahawks', 'Los Angeles Rams', 'Tampa Bay Buccaneers', 'Tennessee Titans', 'Washington Commanders']
        ))
        opp_full_name = [opps_short_to_full_dict[el.upper()] for el in opps_short]
        df1_extracted['OPP'] = opp_full_name
        df1_extracted['game_status'] = game_status

        # convert p_team to fullname
        p_team_full = [opps_short_to_full_dict[el.upper()[:-1]] for el in df1_extracted['p_team'].to_list()]
        df1_extracted['p_team'] = p_team_full

        # add week index
        df1_extracted['week'] = week_i

        # weeklydata
        all_dfs.append(df1_extracted)


all_df = pd.concat(all_dfs, ignore_index=True)


'''
################################################################################
# merge team stats onto punter weekly data
################################################################################
'''

# add columns for p_team's and opp's OFF/DEF
############################################
# create col names
team_stats_cols = ['p_team_' + acol for acol in team_sheet.columns[1:]]
team_stats_cols.extend(['OPP_' + acol for acol in team_sheet.columns[1:]])
for acol in team_stats_cols:
    all_df[acol] = ''   

# merge the data
for rowi, row in all_df.iterrows():
    # get p_team's and opp's ranks
    p_OPP_teams_cols = ['p_team', 'OPP']
    p_OPP_teams_names = [row[acol] for acol in p_OPP_teams_cols]
    p_OPP_teams_ranks = [team_sheet[team_sheet['TEAM'] == team_name]  for team_name in p_OPP_teams_names]

    for tri, team_ranking in enumerate(p_OPP_teams_ranks):
        for acol in team_ranking.columns[1:]:
            aval = team_ranking[acol].values[0]
            merge_col_name = p_OPP_teams_cols[tri] + '_' + acol
            # merge
            all_df.loc[rowi, merge_col_name] = aval



all_df.to_excel(out_path, index=False)