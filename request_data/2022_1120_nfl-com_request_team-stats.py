import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time


def all_data_to_excel(all_tables):
    ''' convert all fetched tables to single excel file with sheet for each '''
    writer = pd.ExcelWriter('all_nflcom_historical_data.xlsx', engine='xlsxwriter')
    for sheetname,df in all_tables.items():
        df.to_excel(writer, sheet_name=sheetname)
    writer.save()

def convert_nflcom_team_name(ateam, input_mode='nflcom'):
    df = pd.read_excel('../data/NFL_team-name_lookup-table.xlsx')
    input_str = ateam.lower()

    #find abrv. in df and return FullName
    if input_mode == 'nflcom':
        search_cols = ['nfl-com_name',	'nfl-com_alt',	'nfl-com_alt2']
        for col in search_cols:
            search_result = df[df[col].str.lower()==input_str]
            if len(search_result) == 1:
                return search_result['FullName'].values[0]
        raise IndexError(f'search for {ateam} returned {len(search_result)} results')

'''
##########################################################################################
HTML structure
##########################################################################################
    data fields are listed before all teams
    each team is bounded by a <tr>
        first <td> holds club information
        each stat is bounded by a subsequent <td>

    
'''
class HTMLparser:
    def __init__(self, date_range=(1999,2023)):
        self.date_range = date_range

        

    def get_stats_table(self,asoup):
        '''returns a bs4.element.Tag object'''
        table_list = asoup.find_all('table')
        assert len(table_list) == 1
        return table_list[0]

    def get_table_elements(self,atable):
        self.stat_cols = atable.find('thead')
        self.stat_columns = [stc.text for stc in self.stat_cols.find_all('th')]
        self.tbody = atable.find('tbody')
        self.team_tags = self.tbody.find_all('tr')
        if len(self.team_tags) == 32:
            pass # normal
        elif len(self.team_tags) == 31:
            self.missing_team_tag_flag = True
        else:
            raise ValueError(f'team tags len unexpected, {len(self.team_tags)} should be 32 or 31')

        return self.stat_columns, self.team_tags

    def get_team_stat_vals(self,team_stats):
        team_divs = team_stats.find_all('td')
        team_name_div = team_divs[0]
        team_stat_divs = team_divs[1:]
        tn_tags = team_name_div.find_all("div", {"class": 'd3-o-club-fullname'})
        assert len(tn_tags) == 1
        raw_team_name = tn_tags[0].text
        team_name = convert_nflcom_team_name(raw_team_name)#fu.get_team_name(raw_team_name)
        team_stat_vals = [team_stat_div.text for team_stat_div in team_stat_divs]
        to_return = [team_name] + team_stat_vals
        return to_return

    def format_stats(self,stat_columns, team_stat_vals):
        assert len(stat_columns) == len(team_stat_vals)
        return dict(zip(stat_columns, team_stat_vals))

    def get_historical_data(self):
        
        base_url = 'http://nfl.com/stats/team-stats'
        suffix = 'reg/all'
        sub_urls = ['offense', 'defense', 'special-teams']
        sub2_urls = [
            ['passing','Rushing','Receiving','Scoring','Downs'], 
            ['Passing','Rushing','Scoring','Downs','Fumbles','Interceptions'], 
            ['Field-Goals','Scoring','Kickoffs','Kickoff-Returns','Punting','Punt-Returns']]

        all_tables = {}
        for sub_url_i in range(len(sub_urls)):
            for sub2_url_i in range(len(sub2_urls[sub_url_i])):
                table_stats = []
                url1 = sub_urls[sub_url_i].lower()
                url2 = sub2_urls[sub_url_i][sub2_url_i].lower()
                table_name = f'{url1}_{url2}'

                for season in range(*self.date_range):
                    time.sleep(2.0) # limit requests to avoid spamming

                    req_url = f'{base_url}/{url1}/{url2}/{season}/{suffix}'
                    req = requests.get(req_url)
                    self.req_url = req_url
                    print(req_url)
                    print(req.status_code)
                    print(req.headers['content-type'])

                    content = req.content
                    soup = BeautifulSoup(content, 'html.parser')
                    self.soup = soup
                    table = self.get_stats_table(soup)
                    self.missing_team_tag_flag = False # used for catching when 1999-2002 texans didn't exist
                    self.missing_team_tag_name = None # which if occurs, is set  in get_table_elements
                    stat_columns, team_tags = self.get_table_elements(table)
                    
                    for team_stats in team_tags:
                        team_stat_vals = self.get_team_stat_vals(team_stats)
                        stats_dict = self.format_stats(stat_columns, team_stat_vals)
                        stats_dict['season'] = season
                        self.stats_dict = stats_dict
                        table_stats.append(stats_dict)

                    
                table_df = pd.DataFrame(table_stats)
                all_tables[table_name] = table_df
        
        # save data to single excel file with sheet for each stat group
        all_data_to_excel(all_tables)

        return all_tables


# if __name__ == '__main__':
parser = HTMLparser(date_range=(1999,2023))
parser.get_historical_data()



