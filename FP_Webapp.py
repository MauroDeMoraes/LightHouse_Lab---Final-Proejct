import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import datetime
import pickle
from sklearn.preprocessing import MinMaxScaler



### Load df ### 
df = pd.read_csv('Final_dataset_top8leagues_LV-22-03.csv')
### Load Image ###
image = Image.open('stadium.jpeg')

st.image(image,use_column_width=True, caption='Soccer Stadium')

st.title('European Soccer League Prediction')

st.markdown(
"""
Leagues availabe for prediction **Premier League, La Liga Primera Division, Le Championnat, Serie A, Bundesliga 1, Championship, League 1 and League 2**

This application predicts the probability of a event to occurs in a future match between two soccer teams. The output of this predict is the change of home team and away team win the match also the probability of tie game. 

* **Data** source: [World Soccer DB: archive of odds [09.02.21]](https://www.kaggle.com/sashchernuh/european-football).

""")
############ TEAMS SELECTION ###############
Team_list_2 = {'Leeds': 0, 'Ath Madrid': 1, 'Marseille': 2, 'Betis': 3, 'Lazio': 4, 'Sheffield United': 5,\
    'Osasuna': 6, 'Parma': 7, 'Liverpool': 8, 'Nantes': 9, 'Ath Bilbao': 10, 'Hoffenheim': 11, 'Montpellier': 12, \
    'St Etienne': 13, 'Milan': 14, 'Wolves': 15, 'Nice': 16, 'Udinese': 17, 'Nimes': 18, 'Sociedad': 19, 'Tottenham': 20, \
    'Brest': 21, 'Benevento': 22, 'Sevilla': 23, 'Lens': 24, 'Man United': 25, 'Genoa': 26, 'Lyon': 27, 'Elche': 28, \
    'Fulham': 29, "M'gladbach": 30, 'Juventus': 31, 'Lorient': 32, 'Huesca': 33, 'Preston': 34, 'Bournemouth': 35, \
    'Bristol City': 36, 'Luton': 37, 'Middlesbrough': 38, 'Millwall': 39, 'QPR': 40, 'Stoke': 41, 'Wycombe': 42, \
    'Newcastle': 43, 'Burnley': 44, 'Burton': 45, 'Tranmere': 46, 'Stevenage': 47, 'Newport County': 48, 'Leyton Orient': 49, \
    'Crawley Town': 50, 'Barrow': 51, 'Accrington': 52, 'Ipswich': 53, 'Peterboro': 54, 'Swindon': 55, 'Milton Keynes Dons': 56, 'Rochdale': 57, \
    'Portsmouth': 58, 'Fleetwood Town': 59, 'Doncaster': 60, 'Wigan': 61, 'Freiburg': 62, 'Leverkusen': 63, \
    'Mainz': 64, 'Schalke 04': 65, 'Augsburg': 66, 'Sassuolo': 67, 'Atalanta': 68, 'Exeter': 69, 'Levante': 70, \
    'Coventry': 71, 'Aston Villa': 72, 'Swansea': 73, 'Alaves': 74, 'Fiorentina': 75, 'Hertha': 76, 'Gillingham': 77, 'Paris SG': 78,\
    'Monaco': 79, 'Dijon': 80, 'Brentford': 81, 'Rotherham': 82, 'Rennes': 83, 'Strasbourg': 84, 'Reims': 85, 'Metz': 86, 'Bordeaux': 87, \
    'Charlton': 88, 'Shrewsbury': 89, 'Blackpool': 90, 'Carlisle': 91, 'Watford': 92, 'Barcelona': 93, 'Roma': 94, 'Brighton': 95, \
    'Granada': 96, 'Napoli': 97, 'Wolfsburg': 98, 'West Ham': 99, 'Lille': 100, 'Cadiz': 101, 'FC Koln': 102, 'Cagliari': 103, 'Leicester': 104,\
    'Angers': 105, 'Crotone': 106, 'Getafe': 107, 'Chelsea': 108, 'Spezia': 109, 'Southampton': 110, 'Villarreal': 111, 'Inter': 112, 'Valencia': 113,\
    'RB Leipzig': 114, 'Arsenal': 115, 'Sampdoria': 116, 'Real Madrid': 117, 'Bradford': 118, 'Forest Green': 119, 'Bolton': 120, 'Grimsby': 121, \
    'Harrogate': 122, 'West Brom': 123, 'Cambridge': 124, 'AFC Wimbledon': 125, 'Bristol Rvs': 126, 'Crewe': 127, 'Hull': 128, 'Lincoln': 129, \
    'Oxford': 130, 'Plymouth': 131, 'Crystal Palace': 132, 'Sheffield Weds': 133, 'Morecambe': 134, 'Port Vale': 135, "Nott'm Forest": 136, 'Oldham': 137, \
    'Man City': 138, 'Blackburn': 139, 'Huddersfield': 140, 'Derby': 141, 'Cardiff': 142, 'Birmingham': 143, 'Union Berlin': 144, 'Dortmund': 145, \
    'Bayern Munich': 146, 'Werder Bremen': 147, 'Ein Frankfurt': 148, 'Bologna': 149, 'Sunderland': 150, 'Eibar': 151, 'Everton': 152, 'Norwich': 153, \
    'Reading': 154, 'Valladolid': 155, 'Torino': 156, 'Stuttgart': 157, 'Colchester': 158, 'Barnsley': 159, 'Southend': 160, 'Scunthorpe': 161, \
    'Cheltenham': 162, 'Celta': 163, 'Verona': 164, 'Bielefeld': 165, 'Salford': 166, 'Walsall': 167, 'Mansfield': 168, 'Northampton': 169, 'Lecce': 170, \
    'Spal': 171, 'Brescia': 172, 'Espanol': 173, 'Leganes': 174, 'Mallorca': 175, 'Fortuna Dusseldorf': 176, 'Paderborn': 177, 'Toulouse': 178, 'Amiens': 179, 'Macclesfield': 180, \
    'Frosinone': 181, 'Caen': 182, 'Chievo': 183, 'Empoli': 184, 'Guingamp': 185, 'Girona': 186, 'Vallecano': 187, 'Hannover': 188, 'Nurnberg': 189, 'Yeovil': 190, 'Bury': 191, \
    'Notts County': 192, 'Malaga': 193, 'Las Palmas': 194, 'Troyes': 195, 'La Coruna': 196, 'Hamburg': 197, 'Barnet': 198, 'Chesterfield': 199, 'Palermo': 200, 'Pescara': 201, 'Nancy': 202, \
    'Ingolstadt': 203, 'Sp Gijon': 204, 'Bastia': 205, 'Darmstadt': 206, 'Hartlepool': 207, 'Carpi': 208, 'Ajaccio GFCO': 209, 'Dag and Red': 210, 'York': 211, 'Cesena': 212, 'Almeria': 213, \
    'Cordoba': 214, 'Evian Thonon Gaillard': 215, 'Catania': 216, 'Sochaux': 217, 'Livorno': 218, 'Valenciennes': 219, 'Ajaccio': 220, 'Braunschweig': 221, \
    'Torquay': 222, 'Zaragoza': 223, 'Siena': 224, 'Greuther Furth': 225, 'Aldershot': 226, 'Auxerre': 227, 'Santander': 228, 'Novara': 229, 'Hereford': 230, \
    'Kaiserslautern': 231, 'Arles': 232, 'Hercules': 233, 'Bari': 234, 'St Pauli': 235, 'Stockport': 236, 'Boulogne': 237, 'Darlington': 238, 'Bochum': 239, \
    'Xerez': 240, 'Grenoble': 241, 'Le Mans': 242, 'Tenerife': 243, 'Numancia': 244, 'Reggina': 245, 'Recreativo': 246, 'Le Havre': 247, 'Cottbus': 248, \
    'Karlsruhe': 249, 'Chester': 250, 'Murcia': 251, 'Hansa Rostock': 252, 'Duisburg': 253, 'Wrexham': 254, 'Gimnastic': 255, 'Ascoli': 256, 'Messina': 257, \
    'Aachen': 258, 'Boston': 259, 'Treviso': 260, 'Rushden & D': 261, 'Kidderminster': 262, 'Wimbledon': 263, 'Munich 1860': 264, 'Halifax': 265, 'Unterhaching': 266
    }

teams_List = []
for team in Team_list_2.keys():
    teams_List.append(team)
teams_list_sort = sorted(teams_List)


team_list = [ 1,  2, 3]
####### SideBar########
st.sidebar.header('User Input Home and Away teams')
today = datetime.date.today()

############ LEAGUE SELECTION ###############
league_dict = {'Bundesliga 1': 0, 'Championship': 1, 'La Liga Primera Division': 2, 'Le Championnat': 3, 'League 1': 4, 
                'League 2': 5, 'Premier League': 6, 'Serie A': 7}

league_List = []
for team in league_dict.keys():
    league_List.append(team)
league_List_sort = sorted(league_List)

############ Country SELECTION ###############
Country_dict = {'England': 0, 'France': 1, 'Germany': 2, 'Italy': 3, 'Spain': 4}
country_List = []
for team in Country_dict.keys():
    country_List.append(team)
Country_dict_sort = sorted(country_List)

def user_input_features():
    country_ = st.sidebar.selectbox('Country', Country_dict_sort)
    country_num = Country_dict.get(country_)

    League_ = st.sidebar.selectbox('League', league_List_sort)
    league_num = league_dict.get(League_)

    home_team = st.sidebar.selectbox('Home Team', teams_list_sort)
    home_team_num = Team_list_2.get(home_team)

    away_team = st.sidebar.selectbox('Away Team', teams_list_sort)
    away_team_num = Team_list_2.get(away_team)

    season = st.sidebar.selectbox('Season', list(reversed(range(2000,2022))))
    Game_Date = st.sidebar.date_input('Game Date', today)
    year_ = Game_Date.year
    day_ = Game_Date.day
    month_ = Game_Date.month
    weekday_ = Game_Date.weekday()

    #### Loop for values into the df ###
    HomeTeam_Last_Status_find = df.loc[df['HomeTeam']==home_team_num].HomeTeam_Last_Status.reset_index().HomeTeam_Last_Status[0]
    AwayTeam_Last_Status_find = df.loc[df['AwayTeam']==away_team_num].HomeTeam_Last_Status.reset_index().HomeTeam_Last_Status[0]
    Total_goal_home_season_find = df.loc[df['HomeTeam']==home_team_num].Total_goal_home_season.reset_index().Total_goal_home_season[0]
    Total_goal_away_season_find = df.loc[df['AwayTeam']==away_team_num].Total_goal_Away_season.reset_index().Total_goal_Away_season[0]
    Total_goal_suf_home_season_find = df.loc[df['HomeTeam']==home_team_num].Total_goal_suf_home_season.reset_index().Total_goal_suf_home_season[0]
    Total_goal_suf_away_season_find = df.loc[df['AwayTeam']==away_team_num].Total_goal_suf_away_season.reset_index().Total_goal_suf_away_season[0]
    TL_goals_Diff_HomeT_PlayHome_season_find = df.loc[df['HomeTeam']==home_team_num].TL_goals_Diff_HomeT_PlayHome_season.reset_index().TL_goals_Diff_HomeT_PlayHome_season[0]
    TL_goals_Diff_AwayT_PlayAway_season_find = df.loc[df['AwayTeam']==away_team_num].TL_goals_Diff_AwayT_PlayAway_season.reset_index().TL_goals_Diff_AwayT_PlayAway_season[0]
    AC = df.loc[(df['AwayTeam']==away_team_num) & (df['Season']==season)].mean()['AC']
    AF = df.loc[(df['AwayTeam']==away_team_num) & (df['Season']==season)].mean()['AF']
    AR = df.loc[(df['AwayTeam']==away_team_num) & (df['Season']==season)].mean()['AR']
    AS = df.loc[(df['AwayTeam']==away_team_num) & (df['Season']==season)].mean()['AS']
    AST = df.loc[(df['AwayTeam']==away_team_num) & (df['Season']==season)].mean()['AST']
    AY = df.loc[(df['AwayTeam']==away_team_num) & (df['Season']==season)].mean()['AY']
    HC = df.loc[(df['HomeTeam']==home_team_num) & (df['Season']==season)].mean()['HC']
    HF = df.loc[(df['HomeTeam']==home_team_num) & (df['Season']==season)].mean()['HF']
    HR = df.loc[(df['HomeTeam']==home_team_num) & (df['Season']==season)].mean()['HR']
    HS = df.loc[(df['HomeTeam']==home_team_num) & (df['Season']==season)].mean()['HS']
    HST = df.loc[(df['HomeTeam']==home_team_num) & (df['Season']==season)].mean()['HST']
    HY = df.loc[(df['HomeTeam']==home_team_num) & (df['Season']==season)].mean()['HY']
    
    data = {'AC' : AC, 'AF': AF, 'AR': AR, 'AS' : AS , 'AST': AST, 'AY' : AY,
            'HC': HC , 'HF': HF, 'HR': HR, 'HS': HS, 'HST': HST , 'HY': HY, 
            'HomeTeam_Last_Status': HomeTeam_Last_Status_find,
            'AwayTeam_Last_Status': AwayTeam_Last_Status_find ,
            'Total_goal_home_season': Total_goal_home_season_find, 
            'Total_goal_Away_season': Total_goal_away_season_find,
            'Total_goal_suf_home_season' : Total_goal_suf_home_season_find, 
            'Total_goal_suf_away_season': Total_goal_suf_away_season_find, 
            'TL_goals_Diff_HomeT_PlayHome_season': TL_goals_Diff_HomeT_PlayHome_season_find, 
            'TL_goals_Diff_AwayT_PlayAway_season':TL_goals_Diff_AwayT_PlayAway_season_find , 
            'Country': country_num, 
            'League': league_num, 
            'Season':season,
            'HomeTeam' : home_team_num, 
            'AwayTeam':away_team_num, 
            'Year': year_, 
            'Month':month_, "Day":day_, 'Week-Day':weekday_ 
            }

    features = pd.DataFrame(data, index=[0])
    return features

df_to_pred = user_input_features()

def scalling(X):
    scaler = MinMaxScaler()
    # Save the variable you don't want to scale
    name_var = X[['Country', 'League', 'Season', 'HomeTeam', 'AwayTeam',  'Year', 'Month', 'Day', 'Week-Day']]

    # Fit scaler to your data
    scaler.fit(X.drop(['Country', 'League', 'Season', 'HomeTeam', 'AwayTeam',  'Year', 'Month', 'Day', 'Week-Day'], axis = 1))

    # Calculate scaled values and store them in a separate object
    scaled_values = scaler.transform(X.drop(['Country', 'League', 'Season', 'HomeTeam', 'AwayTeam',  'Year', 'Month', 'Day', 'Week-Day'], axis = 1))

    data_scl = pd.DataFrame(scaled_values, index = X.index, columns = X.drop(['Country', 'League', 'Season', 'HomeTeam', 'AwayTeam',  'Year', 'Month', 'Day', 'Week-Day'], axis = 1).columns)
    data_scl[['Country', 'League', 'Season', 'HomeTeam', 'AwayTeam',  'Year', 'Month', 'Day', 'Week-Day']] = name_var

    return data_scl

df_scl = scalling(df_to_pred)


# Load and read pickle
load_model = pickle.load(open('Model_xgboosting_03-24.p','rb'))

# predict
prediction = load_model.predict(df_scl)

prediction_prob =  load_model.predict_proba(df_scl)

st.subheader('Result Categories')
st.markdown(
"""
|   Result   |      Category    |
|   - - - -  |       - - -      |
|     0      |   Away Team wins |
|     1      |   Tie Game       |
|     2      |   Home Team Wins |

""")

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_prob)

st.markdown(
"""
### This application has the accuracy of 65.18%

""")																			
