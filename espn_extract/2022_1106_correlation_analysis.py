import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

input_dir = r'C:\Users\pasca\Desktop\2022_FF_Punters'
input_fn = '2022_PunterWeeklyStats_extracted.xlsx'
fn = os.path.join(input_dir, input_fn)

input_fn2 = 'TeamRanksOFF-DEF.xlsx'
fn2 = os.path.join(input_dir, input_fn2)

schedule_fn = '2022_NFL_Schedule.xlsx'
schedule_path = os.path.join(input_dir, schedule_fn)

df_w = pd.read_excel(fn)
df_tr = pd.read_excel(fn2)
schedule = pd.read_excel(schedule_path)

# add punter ranks to each weekly game
p_s_ranks = df_w.groupby('p_team')['P_TOT_PTS'].mean().sort_values(ascending=False,)
p_s_ranks_p_names = df_w.groupby('p_name')['P_TOT_PTS'].mean().sort_values(ascending=False,)
p_rank_df = pd.DataFrame({'p_team':p_s_ranks.index, 'p_name':p_s_ranks_p_names.index, 'P_AVG_PTS':p_s_ranks.values, 'P_rank':list(range(1,33))})
df_w['P_rank'] = 0
for rowi, row in df_w.iterrows():
    this_team = row['p_team']
    p_rank = p_rank_df[p_rank_df['p_team'] == this_team]['P_rank'].values[0]
    df_w.loc[rowi,'P_rank'] = p_rank


# create a table of allowed pts for each team
# this team allowed punters to score this many points
pts_allowed_all = []
pts_scored_all = []
played_p_ranks = []
teams = df_w['p_team'].unique().tolist()
for team in teams:
    # get pts allowed
    team_df = df_w[df_w['OPP'] == team]
    pts_allowed = team_df['P_TOT_PTS'].mean()
    pts_allowed_all.append(pts_allowed)
    # get pts scored
    this_team_df = df_w[df_w['p_team'] == team]
    pts_scored = this_team_df['P_TOT_PTS'].mean()
    pts_scored_all.append(pts_scored)
    # get punter rank this team was playing
    played_p_ranks.append(int(team_df['P_rank'].mean()))

pts_allowed_df = pd.DataFrame({
    'team':teams, 
    'pts_allowed':pts_allowed_all,
    'faced_P_ranks':played_p_ranks,
    'P_pts_scored':pts_scored_all,
    # correlation_columns

})
pts_allowed_df.sort_values(['pts_allowed'], ascending=False, inplace=True)



    



'''
plot heatmap vars vs vars
# seems to indicate the following trends
    positive correlations
        OPP_RUSH_YDS2 - OPP rush yds allowed
        p_team_RUSH_YDS
    negative correlations
        p_team_TOT_YDS
        p_team_OFF_PTS
        OPP_OFF_PTS
        OPP_PASS_YDS2
'''
# get mean data
data = df_w.groupby('p_team').mean()
corr = data.corr()
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
plt.show()



# normalize team ranks df
df_tr.sort_values('OFF_PTS', inplace=True)
df = df_tr.loc[:, df_tr.columns != 'TEAM']
df_tr_N = (df-df.min())/(df.max()-df.min())
df_tr_N['TEAM'] = df_tr['TEAM']
# get correlated stats for a given matchup
P_TEAM = 'Las Vegas Raiders'
OPP_TEAM = 'Jacksonville Jaguars'

# TODO apply correlation weighting
def get_matchup_stats(P_TEAM, OPP_TEAM, df_tr_N):
    p_team_stats = ['RUSH_YDS', 'TOT_YDS', 'OFF_PTS']
    opp_team_stats = ['RUSH_YDS2','OFF_PTS','PASS_YDS2']

    maximize = ['OPP_RUSH_YDS2', 'p_team_RUSH_YDS']
    minimize = ['p_team_TOT_YDS',
            'p_team_OFF_PTS',
            'OPP_OFF_PTS',
            'OPP_PASS_YDS2']
    min_max_names = ['maximize', 'minimize']
    stats_min_max = {'p_team':P_TEAM, 'maximize':0, 'minimize':0, 'p_rank':}
    for i, groupp in enumerate([maximize, minimize]):
        stats_group = []
        for st in groupp:
            # determine whether get from p_team or OPP
            if st.startswith('OPP'):
                get_team = OPP_TEAM
                stf = st[4:]
            else:
                get_team = P_TEAM
                stf = st[7:]
            
            # get stat
            val = df_tr_N[df_tr_N['TEAM'] == get_team][stf].values[0]
            stats_group.append(val)

            print(get_team, st, val)
        
        stats_group_average = sum(stats_group)/len(stats_group)
        stats_min_max[min_max_names[i]] = stats_group_average
    return stats_min_max

this_week = 10
this_schedule = schedule[schedule['week'] == this_week]
all_matchups = []
for i, matchup in this_schedule.iterrows():
    home, away = matchup['home'], matchup['away']
    all_matchups.append(get_matchup_stats(home, away, df_tr_N))
    all_matchups.append(get_matchup_stats(away, home, df_tr_N))

match_up_df = pd.DataFrame(all_matchups).sort_values(['minimize', 'maximize',])

fig, ax = plt.subplots(figsize=(10,10))
g = sns.scatterplot(x='minimize', y='maximize', hue='p_team', data=match_up_df, ax=ax)
ax.plot([0, 1], [1, 0], c='k', alpha=0.3, ls='--')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)


g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=2)

plt.show()





        


