import pandas as pd
import numpy as np
import joblib  # For loading the XGBoost model
import streamlit as st
from tensorflow.keras.models import load_model

# Load the datasets
train_data = pd.read_csv("custom_encoded_train_data.csv")
test_data = pd.read_csv("merged_custom_encoded_test_data.csv")

# Load trained XGBoost model
model = joblib.load("final_xgb_model2.pkl")  # Replace with "final_xgb_model.json" if using XGBoost's native method
scaler = joblib.load("scaler.pkl")

# Extract feature columns from train data (excluding non-predictive columns)
feature_columns = [
    "Seed1", "Seed2", 
    "SeedDiff", 
    "PPG_T1", "PAPG_T1", "PPG_T2", "PAPG_T2", 
    "ConfTourneyStage_T1", "ConfTourneyStage_T2",
    "FG_Percentage_T1", "FG3_Percentage_T1", "FT_Percentage_T1", 
    "Ast_Per_Game_T1", "TO_Per_Game_T1", "Stl_Per_Game_T1", "Blk_Per_Game_T1", "Reb_Per_Game_T1",
    "FG_Percentage_T2", "FG3_Percentage_T2", "FT_Percentage_T2", 
    "Ast_Per_Game_T2", "TO_Per_Game_T2", "Stl_Per_Game_T2", "Blk_Per_Game_T2", "Reb_Per_Game_T2",
    "PPG_Diff", "PAPG_Diff", "OffensiveRating_T1", "OffensiveRank_T1", "DefensiveRating_T1", "DefensiveRank_T1",
    "OffensiveRating_T2", "OffensiveRank_T2", "DefensiveRating_T2", "DefensiveRank_T2", "SOS_T1", "SOS_T2",
    # New rolling features for Team1
    "Last5_PPG_T1", "Win_Perc_Last5_T1", "Last5_PointDiff_Diff",
    "Last10_PPG_T1", "Win_Perc_Last10_T1", "Last10_PointDiff_Diff",
    # New rolling features for Team2
    "Last5_PPG_T2", "Win_Perc_Last5_T2", 
    "Last10_PPG_T2", "Win_Perc_Last10_T2", 
]

# Function to properly capitalize team names
def capitalize_team_name(team_name):
    return ' '.join(word.capitalize() for word in team_name.split())

# Create a dictionary of team names and their IDs with capitalized names
team_dict = {capitalize_team_name(name): id for name, id in zip(test_data['TeamName'], test_data['TeamID'])}
team_names = sorted(team_dict.keys())  # Sorted list of team names for the dropdown

def get_team_id(team_name):
    """Get team ID from team name"""
    return team_dict.get(team_name)

def get_team_stats(team_id):
    """ Extracts the stats of a given team from the test dataset """
    team_stats = test_data[test_data["TeamID"] == team_id]
    if team_stats.empty:
        return None  # If no match found
    stats = team_stats.iloc[0].copy()  # Return as a Series
    stats["TeamName"] = capitalize_team_name(stats["TeamName"])  # Capitalize the team name
    return stats

def create_matchup_row(team1_id, team2_id, season):
    """ Combines two team stats into a row formatted like the training data """
    
    # Extract stats
    team1_stats = get_team_stats(team1_id)
    team2_stats = get_team_stats(team2_id)
    
    
    teamName1 = team1_stats["TeamName"]
    teamName2 = team2_stats["TeamName"]


    if team1_stats is None or team2_stats is None:
        print("❌ Error: One or both teams not found in test data.")
        return None
    
    # Create a dictionary with formatted keys
    matchup_data = {
        "Seed1": team1_stats.get("Seed", 0),
        "Seed2": team2_stats.get("Seed", 0),
        "SeedDiff": team1_stats.get("Seed", 0) - team2_stats.get("Seed", 0),
        "PPG_Diff": (team1_stats.get("PPG", 0) - team2_stats.get("PPG", 0)).round(2),
        "PAPG_Diff": (team1_stats.get("PAPG", 0) - team2_stats.get("PAPG", 0)).round(2),
        "PPG_T1": team1_stats.get("PPG", 0),
        "PPG_T2": team2_stats.get("PPG", 0),

    }
    
    # Add stats with "_T1" and "_T2" suffixes
    for col in test_data.columns:
        if col not in ["TeamID", "TeamName", "Season", "Seed", "PPG"]:
            matchup_data[f"{col}_T1"] = team1_stats.get(col, 0)
            matchup_data[f"{col}_T2"] = team2_stats.get(col, 0)

    return matchup_data, teamName1, teamName2

def predict_game(team1_id, team2_id, season):
    """ Predicts the outcome of a matchup between two teams using XGBoost """
    
    # Create a matchup row
    matchup_data1, teamName1, teamName2 = create_matchup_row(team1_id, team2_id, season)
    matchup_data2, dummy, dummy2 = create_matchup_row(team2_id, team1_id, season)
    if matchup_data1 is None:
        return None
    
    # Convert to DataFrame
    matchup_df = pd.DataFrame([matchup_data1])
    matchup_df2 = pd.DataFrame([matchup_data2])

    # # 🔍 Debugging - Print column names before reindexing
    # print("\nColumns in matchup_df before filtering:", matchup_df.columns.tolist())
    # print("Expected feature_columns:", feature_columns)

    # ✅ Ensure correct feature order and fill missing columns with 0
    matchup_df = matchup_df.reindex(columns=feature_columns, fill_value=0)
    matchup_df = scaler.transform(matchup_df)
    matchup_df2 = matchup_df2.reindex(columns=feature_columns, fill_value=0)
    matchup_df2 = scaler.transform(matchup_df2)


    print(matchup_df)

    # Convert to NumPy array for model input

    # Predict using XGBoost
    prediction1 = model.predict(matchup_df)
    prediction2 = model.predict(matchup_df2)

    print(prediction1)
    print(prediction2)
    # Initialize result variable
    result = ""

    # Convert probability to binary win/loss if using probability output
    if hasattr(model, "predict_proba"):  # If model supports probability output
        prediction_prob1 = model.predict_proba(matchup_df)
        prediction_prob2 = model.predict_proba(matchup_df2)
        print(prediction_prob1)
        print(prediction_prob2)
        pred1_1 = prediction_prob1[0][0]
        pred2_1 = prediction_prob1[0][1] # Probability of Team 1 winning
        pred1_2 = prediction_prob2[0][0]
        pred2_2 = prediction_prob2[0][1] # Probability of Team 1 winning

        total1 = (pred2_1 + pred1_2) / 2
        total2 = (pred1_1 + pred2_2) / 2
        
        if total1 > total2:
            print(total1)
            result = f'{capitalize_team_name(teamName1)} has a {total1 * 100:.2f}% probability to beat {capitalize_team_name(teamName2)}'
        else:
            print(total2)
            result = f'{capitalize_team_name(teamName2)} has a {total2 * 100:.2f}% probability to beat {capitalize_team_name(teamName1)}'
    else:
        # Handle case where model doesn't provide probabilities
        if prediction1 == 1:
            result = f'{capitalize_team_name(teamName1)} is predicted to win against {capitalize_team_name(teamName2)}'
        else:
            result = f'{capitalize_team_name(teamName2)} is predicted to win against {capitalize_team_name(teamName1)}'

    return result

# UI with Streamlit
st.title("🏀 Basketball Game Predictor")

st.write("Select the teams to predict the game outcome:")

col1, col2 = st.columns(2)
with col1:
    team1_name = st.selectbox(
        "Team 1",
        options=team_names,
        key="team1"
    )
with col2:
    team2_name = st.selectbox(
        "Team 2",
        options=team_names,
        key="team2"
    )

if st.button("Predict"):
    try:
        team1_id = get_team_id(team1_name)
        team2_id = get_team_id(team2_name)
        
        if team1_id == team2_id:
            st.error("Please select different teams")
        else:
            result = predict_game(team1_id, team2_id, 2025)
            
            if result:
                st.success("🎯 Prediction Result:")
                st.write(result)
                
                # Get team stats for display
                team1_stats = get_team_stats(team1_id)
                team2_stats = get_team_stats(team2_id)
                
                # Display team statistics
                st.subheader("Team Statistics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**{team1_stats['TeamName']}**")
                    st.write(f"Seed: {team1_stats['Seed'] if team1_stats['Seed'] != 0 else 'Not in Tournament'}")
                    st.write(f"PPG: {(team1_stats['PPG']):.1f}")
                    st.write(f"FG%: {(team1_stats['FG_Percentage'] * 100):.1f}%")
                    st.write(f"3P%: {(team1_stats['FG3_Percentage'] * 100):.1f}%")
                    
                with col2:
                    st.write(f"**{team2_stats['TeamName']}**")
                    st.write(f"Seed: {team2_stats['Seed'] if team2_stats['Seed'] != 0 else 'Not in Tournament'}")
                    st.write(f"PPG: {(team2_stats['PPG']):.1f}")
                    st.write(f"FG%: {(team2_stats['FG_Percentage'] * 100):.1f}%")
                    st.write(f"3P%: {(team2_stats['FG3_Percentage'] * 100):.1f}%")
            else:
                st.error("Could not make prediction. Please check if the teams are valid.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        

