
import sqlite3
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import pickle 

"""### Load data from SQLite"""

con = sqlite3.connect("/database.sqlite")
cursor = con.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())

df = pd.read_sql('SELECT * FROM football_data', con)

"""#### Drop columns relatedt o betting
These columns won't be used in this project

"""

df = df.drop(columns=['B365H', 'B365D', 'B365A', 'BSH', 'BSD', 'BSA', 'BWH', 'BWD', 'BWA', 'GBH',
                     'GBD', 'GBA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'PSH', 'PH', 'PSD', 'PD', 'PSA',
                     'PA', 'SOH', 'SOD', 'SOA', 'SBH', 'SBD', 'SBA', 'SJH', 'SJD', 'SJA', 'SYH', 'SYD', 'SYA',
                     'VCH', 'VCD', 'VCA', 'WHH', 'WHD', 'WHA', 'Bb1X2', 'BbMxH', 'BbAvH', 'BbMxD', 'BbAvD', 'BbMxA',
                     'BbAvA', 'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA', 'BbOU', 'BbMx>2.5', 'BbAv>2.5', 'BbMx<2.5',
                     'BbAv<2.5', 'GB>2.5', 'GB<2.5', 'B365>2.5', 'B365<2.5', 'P>2.5', 'P<2.5', 'Max>2.5', 'Max<2.5',
                     'Avg>2.5', 'Avg<2.5', 'BbAH', 'BbAHh', 'AHh', 'BbMxAHH', 'BbAvAHH', 'BbMxAHA', 'BbAvAHA', 'GBAHH',
                     'GBAHA', 'GBAH', 'LBAHH', 'LBAHA', 'LBAH', 'B365AHH', 'B365AHA', 'B365AH', 'PAHH', 'PAHA', 'MaxAHH',
                     'MaxAHA', 'AvgAHH', 'AvgAHA','ABP','AHCh', 'AvgC<2.5', 'AvgC>2.5', 'AvgCA', 'AvgCAHA', 'AvgCAHH', 
                     'AvgCD', 'AvgCH', 'B365C<2.5', 'B365C>2.5', 'B365CA', 'B365CAHA','B365CAHH', 'B365CD', 'B365CH', 
                     'BWCA', 'BWCD', 'BWCH','IWCA', 'IWCD', 'IWCH', 'MaxC<2.5', 'MaxC>2.5', 'MaxCA', 'MaxCAHA', 
                     'MaxCAHH','MaxCD', 'MaxCH', 'PC<2.5', 'PC>2.5', 'PCAHA', 'PCAHH', 'PSCA', 'PSCD','PSCH',
                     'VCCA', 'VCCD', 'VCCH', 'WHCA', 'WHCD', 'WHCH', 'HBP'])



""" Data Explorer
"""
## Countries ##

list_of_countries = df['Country'].unique()

df_plot = df.groupby('Country').count()

df_plot_countries = df['Country'].value_counts()

df_plot = pd.DataFrame(df_plot_countries).reset_index().rename(columns={"index": "Country", "Country": "Total"})
df_plot.to_csv('world_countries.csv')

fig = go.Figure(data=go.Choropleth(
        locations = df_plot['Country'],
        z = df_plot['Total'].astype(int),
        locationmode = 'country names',
        colorscale = 'algae'))
fig.show()

## Total games per country ##

fig = plt.figure(figsize = (15,12))
plt.barh(df_plot_countries.index, df_plot_countries, align='center')
plt.title('Total games per country')
plt.xlabel('Countries')
plt.ylabel('Total games')
plt.show()

## Different leagues ##

df_plot_Leagues = df['League'].value_counts().reset_index().rename(columns={"index": "League", "League": "Total Games"})

fig = plt.figure(figsize = (15,12))
plt.bar(df_plot_Leagues['League'], df_plot_Leagues['Total Games'])
plt.title('Total games per League')
plt.xlabel('League')
plt.xticks(rotation=90)
plt.ylabel('Total games')
plt.show()

"""#### Games per year"""

df['Datetime'] = pd.to_datetime(df['Datetime'])

df_plot_years = df['Datetime'].dt.year.value_counts().rename_axis('Year').to_frame('Games_count').reset_index().sort_values('Year')

df_plot_years

sns.lineplot(data=df_plot_years, x="Year", y="Games_count")

## Missing Values ## 

total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data

#Dropping columns with over 70% missing values
df_dropped = df.drop(columns=['AT','HT','AFKC','AHW','HFKC','HHW','AO', 'HO', 'Attendance','Referee'] )


#Dropping columns with duplicat information
df_dropped = df_dropped.drop(columns=['Date','Time'])


total = df_dropped.isnull().sum().sort_values(ascending=False)
percent = (df_dropped.isnull().sum()/df_dropped.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

## Full data - All Years ##

df_nan_finding = df_dropped.loc[(df_dropped['HF'].isnull())]

df_comparing = df_nan_finding.groupby('League').count()['HomeTeam']
df_comparing = pd.DataFrame(df_comparing).reset_index().rename(columns={"index": "League",'HomeTeam':'Total_Games_null'})

df_total_games = df_dropped.groupby('League').count()['HomeTeam']
df_total_games  = pd.DataFrame(df_total_games).reset_index().rename(columns={"index": "League",'HomeTeam':'Total_Games'})

df_total_compraing = df_total_games.merge(df_comparing,left_on='League', right_on='League')



"""## Deleting leagues
These leagues will not be part of this project. 
Project will only be prediucting the top 8 European leagues
"""
# Reseting DataFrame index for League
df_full_leagues_index = df_dropped.set_index('League')

# Deleting leagues 
df_full_leagues_index = df_full_leagues_index.drop(['Liga MX', 'La Liga Segunda Division', 'Liga 1', 'Superliga', 'Ethniki Katigoria','Conference',
                                          'Futbol Ligi 1', 'Serie B', 'Jupiler League', 'Ekstraklasa', 'Super League', 'Division 1',
                                          'Eredivisie', 'Division 2','Bundesliga', 'Primera Division', 'Division 3', 'Eliteserien' , 
                                          'Allsvenskan', 'MLS', 'Veikkausliiga', 'J-League','Premier Division', 'Liga I', 'Bundesliga 2']).reset_index()

#  Reseting DataFrame index for country, easier to delete leagues that has the same name as others like Serie A
df_full_leagues_index = df_full_leagues_index.set_index('Country')
df_full_leagues_index = df_full_leagues_index.drop(['Brazil','Russia', 'Scotland']).reset_index( )

## Checking for mising values
total = df_full_leagues_index.isnull().sum().sort_values(ascending=False)
percent = (df_full_leagues_index.isnull().sum()/df_full_leagues_index.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print (missing_data.head(15))


# Delete rows with missing data, it is only 8% of the data.
df_full_leagues_index = df_full_leagues_index.dropna()

## Filtering Engineering##

# creating columns for year, month and day, weekday and deleting the full date
df_full_leagues_index['Year'] = pd.DatetimeIndex(df_full_leagues_index['Datetime']).year
df_full_leagues_index['Month'] = pd.DatetimeIndex(df_full_leagues_index['Datetime']).month
df_full_leagues_index['Day'] = pd.DatetimeIndex(df_full_leagues_index['Datetime']).day
df_full_leagues_index['Week-Day'] = pd.DatetimeIndex(df_full_leagues_index['Datetime']).weekday
df_full_leagues_index = df_full_leagues_index.drop(columns=['Datetime'])

# Dropping Div feature
df_full_leagues_index = df_full_leagues_index.drop(columns=['Div'])


## Creating list of season
list_of_season = df_full_leagues_index['Season'].unique()
list_of_season

## Feature creation, Total goals the home team scored playing home during the season
def totalgoalhome(X):
    for season in list_of_season:
        Z = df_full_leagues_index.loc[(df_full_leagues_index['HomeTeam']==X) & (df_full_leagues_index['Season'] ==season) ]
        return  Z.FTHG.sum()      
df_full_leagues_index['Total_goal_home_season'] = df_full_leagues_index['HomeTeam'].apply(totalgoalhome)

## Feature creation, Total goals the away team scored playing away during this season
def totalgoalaway(X):
    for season in list_of_season:
        Z = df_full_leagues_index.loc[(df_full_leagues_index['AwayTeam']==X) & (df_full_leagues_index['Season'] ==season) ]
        return Z.FTAG.sum()
df_full_leagues_index['Total_goal_Away_season'] = df_full_leagues_index['AwayTeam'].apply(totalgoalhome)

## Feature creation, Total goals the home team suffered playing home during the season
def totalgoalsufferedhome(X):
    for season in list_of_season:
        Z = df_full_leagues_index.loc[(df_full_leagues_index['HomeTeam']==X) & (df_full_leagues_index['Season'] ==season) ]
        return Z.FTAG.sum()
df_full_leagues_index['Total_goal_suf_home_season'] = df_full_leagues_index['HomeTeam'].apply(totalgoalsufferedhome)

## Feature creation, Total goals the away team suffered playing away during this season
def totalgoalsufferedaway(X):
    for season in list_of_season:
        Z = df_full_leagues_index.loc[(df_full_leagues_index['AwayTeam']==X) & (df_full_leagues_index['Season'] ==season) ]
        return Z.FTHG.sum()
df_full_leagues_index['Total_goal_suf_away_season'] = df_full_leagues_index['AwayTeam'].apply(totalgoalhome)


# Feature creation, Gols differenc from home team playing at home and goals difference away team playing away
df_full_leagues_index['TL_goals_Diff_HomeT_PlayHome_season'] = df_full_leagues_index['Total_goal_home_season'] - df_full_leagues_index['Total_goal_suf_home_season']
df_full_leagues_index['TL_goals_Diff_AwayT_PlayAway_season'] = df_full_leagues_index['Total_goal_Away_season'] - df_full_leagues_index['Total_goal_suf_away_season']

# Creating columns with the last game result --> H: win, D: Draw, A:lost
df_full_leagues_index['HomeTeam_Last_Status'] = df_full_leagues_index.groupby(['HomeTeam']).FTR.shift(-1)

# Creating columns with the last game result --> A: win, D: Draw, H:lost
df_full_leagues_index['AwayTeam_Last_Status'] = df_full_leagues_index.groupby(['AwayTeam']).FTR.shift(-1)

# create a columns thats looks into homeTeam and AwayTeam en see if they won, if yes 1 , no 0
def lastgame_away(X):
    if X == 'A':
        return 2
    elif X == 'D':
        return 1
    else: return 0
df_full_leagues_index['AwayTeam_Last_Status'] = df_full_leagues_index['AwayTeam_Last_Status'].apply(lastgame_away)

# Replace H,D or A to numbers
def lastgame(X):
    if X == 'H':
        return 2
    elif X == 'D':
        return 1
    else: return 0
df_full_leagues_index['HomeTeam_Last_Status'] = df_full_leagues_index['HomeTeam_Last_Status'].apply(lastgame)

# Total points home team won per game
def pointwon(x):
    if x == 'H':
        return 3
    elif x == 'D':
        return 1
    else: return 0
    
df_full_leagues_index['Pointswon_HT'] = df_full_leagues_index['FTR'].apply(pointwon)

# Total points Away team won per game
def pointwon(x):
    if x == 'A':
        return 3
    elif x == 'D':
        return 1
    else: return 0

df_full_leagues_index['Pointswon_AT'] = df_full_leagues_index['FTR'].apply(pointwon)

df_full_leagues_index['HAttack_effic'] = df_full_leagues_index['FTHG']/df_full_leagues_index['HS']
df_full_leagues_index['AAttack_effic'] = df_full_leagues_index['FTAG']/df_full_leagues_index['AS']
df_full_leagues_index['HDefense_effic'] = df_full_leagues_index['FTAG']/df_full_leagues_index['AS']
df_full_leagues_index['ADefense_effic'] = df_full_leagues_index['FTHG']/df_full_leagues_index['HS']

# Feature creation, Creating a dictonary with all teams and a corresponding number
list_teams = df_full_leagues_index['HomeTeam'].unique()
count = 0
teams_dicts = {}
for i in list_teams:
    teams_dicts[i] = count
    count+=1
df_full_leagues_index = df_full_leagues_index.replace({"HomeTeam": teams_dicts, 'AwayTeam':teams_dicts})

# Feature alteratin, Alterar o FTR para numerical - Labor encoder 

LE = LabelEncoder()
df_full_leagues_index['FTR'] = LE.fit_transform(df_full_leagues_index['FTR'])

df_full_leagues_index['HTR'] = LE.fit_transform(df_full_leagues_index['HTR'])

df_full_leagues_index['Country'] = LE.fit_transform(df_full_leagues_index['Country'])

df_full_leagues_index['League'] = LE.fit_transform(df_full_leagues_index['League'])


# transforming Season into numerical --> 2020/2021 to 2020
df_full_leagues_index['Season'] = df_full_leagues_index['Season'].str[5:]

# Checking for missing values after the feature engineering 
total = df_full_leagues_index.isnull().sum().sort_values(ascending=False)
percent = (df_full_leagues_index.isnull().sum()/df_full_leagues_index.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(5))

# Missing data , replace Nan with 0
df_full_leagues_index["AAttack_effic"] = df_full_leagues_index["AAttack_effic"].fillna(0)
df_full_leagues_index["HDefense_effic"] = df_full_leagues_index["HDefense_effic"].fillna(0)
df_full_leagues_index["ADefense_effic"] = df_full_leagues_index["ADefense_effic"].fillna(0)
df_full_leagues_index["HAttack_effic"] = df_full_leagues_index["HAttack_effic"].fillna(0)

# Alter columns to categorical 
df_full_leagues_index['Country'] = df_full_leagues_index['Country'].astype('category')
df_full_leagues_index['League'] = df_full_leagues_index['League'].astype('category')
df_full_leagues_index['Season'] = df_full_leagues_index['Season'].astype('category')
df_full_leagues_index['Season'] = df_full_leagues_index['Season'].astype('category')
df_full_leagues_index['HomeTeam'] = df_full_leagues_index['HomeTeam'].astype('category')
df_full_leagues_index['AwayTeam'] = df_full_leagues_index['AwayTeam'].astype('category')
df_full_leagues_index['FTR'] = df_full_leagues_index['FTR'].astype('category')
df_full_leagues_index['HTR'] = df_full_leagues_index['HTR'].astype('category')
df_full_leagues_index['Year'] = df_full_leagues_index['Year'].astype('category')
df_full_leagues_index['Month'] = df_full_leagues_index['Month'].astype('category')
df_full_leagues_index['Day'] = df_full_leagues_index['Day'].astype('category')
df_full_leagues_index['Week-Day'] = df_full_leagues_index['Week-Day'].astype('category')
df_full_leagues_index['New_date'] = df_full_leagues_index['New_date'].astype('category')

#### Exporting dataset to csv file

df_full_leagues_index.to_css('Final_dataset_top8leagues_LV-22-03.csv', index=False)


# import to pickle

df_full_leagues_index.to_pickle('Final_dataset_top8leagues.pickle')

