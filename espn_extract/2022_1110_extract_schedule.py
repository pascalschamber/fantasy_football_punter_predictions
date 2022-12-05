import os
import pandas as pd

adir = r'C:\Users\pasca\Desktop\2022_FF_Punters'
fn = 'NFL-Schedule-2022-FPv2.xlsx'
schedule_path = os.path.join(adir, fn)
lookup_fn = 'NFL_team-name_lookup-table.xlsx'
lookup_path = os.path.join(adir, lookup_fn)

schedule = pd.read_excel(schedule_path, sheet_name='Raw')
lookup = pd.read_excel(lookup_path)

class matchup:
    def __init__(self):
        pass

def convert_abr_to_fullname(astr):
    return lookup[lookup['3-letter']==astr]['FullName'].values[0]

teams = schedule['Team'].to_list()
game_dicts = []
for week_i in range(1,19):
    this_week = schedule[week_i].to_list()
    
    for team_i in range(len(teams)):
        this_team = teams[team_i].strip()
        plays = this_week[team_i].strip()
        # skip bye weeks
        if plays == 'BYE':
            continue

        # determine home or away
        HOME = False if '@' in plays else True
        if HOME:
            home_team = this_team
            away_team = plays
        else:
            home_team = plays[1:]
            away_team = this_team
        
        # convert to full name
        home_team, away_team = convert_abr_to_fullname(home_team), convert_abr_to_fullname(away_team)
        # save to dict
        this_game_dict = {'week':week_i, 'home':home_team, 'away':away_team}

        # append if not already in list
        if this_game_dict not in game_dicts:
            game_dicts.append(this_game_dict)
        else:
            print('duplicate')


schedule_df = pd.DataFrame(game_dicts)
schedule_df.to_excel(os.path.join(adir, '2022_NFL_Schedule.xlsx'), index=False)








