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

df_w = pd.read_excel(fn)
df_tr = pd.read_excel(fn2)

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
})
pts_allowed_df.sort_values(['pts_allowed'], ascending=False, inplace=True)

# plot pts_allowed vs faced p rank
# see if strength of p played aligns with pts_allowed
# teams playing the best punters should have most pts allowed
fig, ax = plt.subplots(figsize=(10,5))
g = sns.scatterplot(x='faced_P_ranks', y='pts_allowed', hue='team', data=pts_allowed_df, 
                ax=ax, alpha=0.3)
ax.plot(np.arange(0, 32), list(np.arange(0, 32))[::-1], c='k', alpha=0.3, ls='--')
# ax.set_xlim(pts_allowed_df['faced_P_ranks'].min()-1, pts_allowed_df['faced_P_ranks'].max()+1)
# ax.set_ylim(pts_allowed_df['pts_allowed'].min()-1, pts_allowed_df['pts_allowed'].max()+1)
ax.legend([],[])
plt.tight_layout()
plt.show()


# plot projections vs real
# TODO
proj_diff = df_w['P_TOT_PTS'] - df_w['PROJECTION']
fig, ax = plt.subplots(figsize=(10,5))
g = sns.scatterplot(x='PROJECTION', y='P_TOT_PTS', hue='p_team', data=df_w, 
                ax=ax, alpha=0.3)
# plot linear line
ax.plot(np.arange(-20, 60), np.arange(-20, 60), c='k', alpha=0.3, ls='--')

ax.set_xlim(df_w['P_TOT_PTS'].min()-1, df_w['P_TOT_PTS'].max()+1)
ax.set_ylim(df_w['P_TOT_PTS'].min()-1, df_w['P_TOT_PTS'].max()+1)
g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=2)
plt.tight_layout()
plt.show()

# extract best predictors of p score



