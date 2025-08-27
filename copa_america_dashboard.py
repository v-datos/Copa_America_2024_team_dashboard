import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mplsoccer import Pitch, VerticalPitch, Radar
from matplotlib.font_manager import FontProperties
import warnings
warnings.filterwarnings('ignore')

# Import all functions from the existing analysis script
from Team_Analysis_Dashboard_Copa_America_2024 import *

# Load custom fonts
font_path="Play-Regular.ttf"
font_play = FontProperties(fname=font_path)
font_play_bold = FontProperties(fname=font_path, weight='bold')

# Set page configuration
st.set_page_config(
    page_title="Team Analysis Dashboard\nCopa America 2024",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Play:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Play';
    }
    
    .main-header {
        font-family: 'Play';
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        font-family: 'Play';
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .tab-header {
        font-family: 'Play';
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    
    /* Apply Play font to all Streamlit elements */
    .stMetric, .stMetric > div, .stMetric label {
        font-family: 'Play' !important;
    }
    
    .stSubheader, h1, h2, h3, h4, h5, h6 {
        font-family: 'Play' !important;
    }
    
    .stSelectbox, .stSelectbox > div, .stSelectbox label, .stSelectbox select {
        font-family: 'Play' !important;
    }
    
    .stMarkdown, .stMarkdown > div {
        font-family: 'Play' !important;
    }
    
    /* Apply Play font to tabs and make them bold */
    .stTabs [data-baseweb="tab-list"] button {
        font-family: 'Play' !important;
        font-weight: bold !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-family: 'Play' !important;
        font-weight: bold !important;
    }
    
    /* Apply Play font to sidebar */
    .css-1d391kg, .css-1d391kg .stSelectbox, .css-1d391kg .stSelectbox > div {
        font-family: 'Play' !important;
    }
    
    /* Apply Play font to sidebar headers and labels */
    .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg .stSelectbox label {
        font-family: 'Play' !important;
    }
    
    /* Apply Play font to all sidebar text elements */
    section[data-testid="stSidebar"] * {
        font-family: 'Play' !important;
    }
    
    /* Apply Play font to dropdown options */
    div[data-baseweb="select"] > div {
        font-family: 'Play' !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all Copa America 2024 data"""
    try:
        events_df = pd.read_csv('copa_america_events.csv')
        copa_america_games = pd.read_csv('copa_america_games_2024.csv')
        block_df = pd.read_csv('copa_america_blocks_2024.csv')
        dribbled_pasts_df = pd.read_csv('copa_america_dribbled_pasts_2024.csv')
        duels_df = pd.read_csv('copa_america_duels_2024.csv')
        foul_committed_df = pd.read_csv('copa_america_foul_committeds_2024.csv')
        interceptions_df = pd.read_csv('copa_america_interceptions_2024.csv')
        df_50_50 = pd.read_csv('copa_america_50_50s_2024.csv')
        pressure_df = pd.read_csv('copa_america_pressure_2024.csv')
        counterpress_df = pd.read_csv('copa_america_counterpress_2024.csv')
        pass_df = pd.read_csv('copa_america_pass_2024.csv')
        shot_df = pd.read_csv('copa_america_shots_2024.csv')
        
        return {
            'events': events_df,
            'games': copa_america_games,
            'blocks': block_df,
            'dribbled_pasts': dribbled_pasts_df,
            'duels': duels_df,
            'fouls': foul_committed_df,
            'interceptions': interceptions_df,
            'fifty_fifty': df_50_50,
            'pressure': pressure_df,
            'counterpress': counterpress_df,
            'passes': pass_df,
            'shots': shot_df
        }
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def get_team_metrics(pass_df, shot_df, team):
    """Get comprehensive team metrics"""
    return analyze_team_metrics(pass_df, shot_df, team, return_dict=True)

def create_xg_timeline(shot_df, team):
    """Create xG timeline visualization using Plotly"""
    # Validate that the team exists in the data
    if team not in copa_america_games['home_team'].unique() and \
       team not in copa_america_games['away_team'].unique():
        raise ValueError(f"Team '{team}' not found in the dataset.")

    # Get all match IDs and dates for the selected team
    team_matches_df = copa_america_games[
        (copa_america_games['home_team'] == team) | (copa_america_games['away_team'] == team)
    ].copy()
    
    # Convert match_date to datetime and sort chronologically
    team_matches_df['match_date'] = pd.to_datetime(team_matches_df['match_date'])
    team_matches_df = team_matches_df.sort_values('match_date')
    
    team_matches = team_matches_df['match_id'].tolist()
    match_dates = team_matches_df['match_date'].tolist()
    
    match_xg = []
    match_xg_against = []
    
    # Calculate xG for and against for each match
    for match in team_matches:
        match_shots = shot_df[shot_df['match_id'] == match]
        
        # Calculate xG for (team's non-penalty shots)
        team_shots = match_shots[
            (match_shots['team'] == team) & (match_shots['shot_type'] != 'Penalty')
        ]
        xg_for = team_shots['shot_statsbomb_xg'].mean() if not team_shots.empty else 0
        
        # Calculate xG against (opponent's non-penalty shots)
        opponent_shots = match_shots[
            (match_shots['team'] != team) & (match_shots['shot_type'] != 'Penalty')
        ]
        xg_against = opponent_shots['shot_statsbomb_xg'].mean() if not opponent_shots.empty else 0
        
        match_xg.append(xg_for)
        match_xg_against.append(xg_against)

    # Calculate averages for dotted lines
    avg_xg_for = np.mean(match_xg) if match_xg else 0
    avg_xg_against = np.mean(match_xg_against) if match_xg_against else 0

    fig = go.Figure()
    
    # Add main lines
    fig.add_trace(go.Scatter(
        x=match_dates,
        y=match_xg,
        mode='lines+markers',
        name=f'{team} Average xG',
        line=dict(color='green', width=3),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=match_dates,
        y=match_xg_against,
        mode='lines+markers',
        name=f'{team} Average xG Conceded',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    # Add average dotted lines
    fig.add_trace(go.Scatter(
        x=match_dates,
        y=[avg_xg_for] * len(match_dates),
        mode='lines',
        name=f'Avg xG For ({avg_xg_for:.3f})',
        line=dict(color='green', width=2, dash='dot'),
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=match_dates,
        y=[avg_xg_against] * len(match_dates),
        mode='lines',
        name=f'Avg xG Against ({avg_xg_against:.3f})',
        line=dict(color='#1f77b4', width=2, dash='dot'),
        showlegend=True
    ))
    
    fig.update_layout(
        title=f'{team} xG per Game',
        xaxis_title="Match Date",
        yaxis_title='Average xG',
        height=400,
        showlegend=True,
    )
    
    return fig

def create_shot_map(shot_df, team):
    """Create shot map visualization"""
    team_shots = shot_df[shot_df['team'] == team].copy()
    
    fig, ax = plt.subplots(figsize=(10,6))
    pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white')
    pitch.draw(ax=ax)
    
    # Plot shots
    goals = team_shots[team_shots['shot_outcome'] == 'Goal']
    other_shots = team_shots[team_shots['shot_outcome'] != 'Goal']
    
    # Plot non-goals
    if len(other_shots) > 0:
        pitch.scatter(other_shots.x, other_shots.y, s=other_shots.shot_statsbomb_xg * 300,
                     c='red', alpha=0.6, ax=ax, label='Shots')
    
    # Plot goals
    if len(goals) > 0:
        pitch.scatter(goals.x, goals.y, s=goals.shot_statsbomb_xg * 300, marker='football',
                     c='black', alpha=0.8, ax=ax, label='Goals',)
    
    ax.set_title(f'{team} - Shot Map (Size = xG)', fontsize=22, pad=20, fontproperties=font_play_bold)
    ax.legend(loc='upper right')
    
    return fig

def main():
    # Load data
    data = load_data()
    if data is None:
        st.stop()
    
    # Main header
    st.markdown('<h1 class="main-header">Team Analysis Dashboard<br>⚽ Copa America 2024</h1>', unsafe_allow_html=True)
    
    # Sidebar for team selection
    st.sidebar.header("Team Selection")
    teams = sorted(data['shots']['team'].unique())
    selected_team = st.sidebar.selectbox("Select a team:", teams)
    
    # Get team metrics
    team_metrics = get_team_metrics(data['passes'], data['shots'], selected_team)
    
    if team_metrics is None:
        st.error(f"Could not load metrics for {selected_team}")
        st.stop()
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Main KPIs", 
        "🎯 Playing Style", 
        "⚽ Passes & Shots", 
        "🛡️ Defensive Metrics", 
        "⚔️ Team Comparison"
    ])
    
    # Tab 1: Main KPIs
    with tab1:
        st.markdown(f'<div class="tab-header">Metrics for {selected_team}ß</div>', unsafe_allow_html=True)
        
        # xG Timeline
        st.subheader("Expected Goals Timeline")
        xg_timeline = create_xg_timeline(data['shots'], selected_team)
        st.plotly_chart(xg_timeline, use_container_width=True)

        st.subheader("Main KPIs and Performance Metrics")
        

        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            matches_played = team_metrics['defensive']['matches_played']
            st.metric("Matches Played", matches_played)
            
            goals = team_metrics['shooting']['goals']
            st.metric("Goals Scored", goals)
        
        with col2:
            pass_completion = team_metrics['passing']['pass_completion_rate']
            st.metric("Pass Completion Rate", f"{pass_completion:.1f}%")
            
            total_passes = team_metrics['passing']['total_passes']
            st.metric("Total Passes", f"{total_passes:,}")
        
        with col3:
            shots_on_target_pct = team_metrics['shooting']['shots_on_target_percentage']
            st.metric("Shots on Target %", f"{shots_on_target_pct:.1f}%")
            
            total_shots = team_metrics['shooting']['total_shots']
            st.metric("Total Shots", total_shots)
        
        with col4:
            avg_xg = team_metrics['shooting']['non_penalty_avg_xG']
            st.metric("Average xG", f"{avg_xg:.3f}")
            
            avg_xg_per_shot = team_metrics['shooting']['xg_per_shot']
            st.metric("Avg xG/Shot", f"{avg_xg_per_shot:.3f}")
         
        # Additional metrics
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            goals_conceded = team_metrics['defensive']['goals_conceded']
            st.metric("Goals Conceded", goals_conceded)
        
        with col6:
            shot_assist_passes = team_metrics['passing']['shot_assist_passes']
            st.metric("Passes to Shot", shot_assist_passes)
        
        with col7:
            shots_against = team_metrics['defensive']['total_shots_against']
            st.metric("Total Shots Against", shots_against)
        
        with col8:
            avg_xg_per_shot_against = team_metrics['defensive']['avg_xg_per_shot_against']
            st.metric("Avg xG/Shot Against", f"{avg_xg_per_shot_against:.3f}")
         
    # Tab 2: Playing Style Analysis
    with tab2:
        st.markdown('<div class="tab-header">Playing Style Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Team Performance Radar")
            
            # Get radar stats
            radar_stats = get_team_radar_stats(team_metrics)
            
            # Create radar chart using matplotlib
            fig, ax = create_team_radar_chart(
                radar_stats, 
                font_play, 
                team_name=selected_team,
                team_color='#1f77b4'
            )
            st.pyplot(fig)
        
        with col2:
            st.subheader("Key Style Metrics")
            
            # Split metrics into two columns
            metric_col1, metric_col2 = st.columns(2)
            
            with metric_col1:
                st.metric("Non-Penalty xG", f"{team_metrics['shooting']['non_penalty_avg_xG']:.2f}")
                st.metric("Shots on Target %", f"{team_metrics['shooting']['shots_on_target_percentage']:.2f}%")
                st.metric("Shots per Game", f"{team_metrics['shooting']['shots_per_match']:.2f}")
                st.metric("Counter Shots/Game", f"{team_metrics['shooting']['counter_shots_per_match']:.2f}")
                st.metric("Avg xG Set Piece", f"{team_metrics['shooting']['avg_xG_set_piece']:.2f}")
            
            with metric_col2:
                st.metric("Under Pressure %", f"{team_metrics['passing']['under_pressure_percentage']:.2f}%")
                st.metric("Through Ball %", f"{team_metrics['passing']['through_ball_percentage']:.2f}%")
                st.metric("GK Pass Length (avg)", f"{team_metrics['passing']['goalkeeper_pass_avg_length']:.1f}m")
                st.metric("Cross %", f"{team_metrics['passing']['cross_percentage']:.2f}%")
        
        # Distribution plots
        st.subheader(f"Metric Distributions for {selected_team}")
        
        dist_col1, dist_col2 = st.columns(2)
        
        with dist_col1:
            st.write("**xG Distribution**")
            try:
                plot_xg_distribution(data['shots'], selected_team, color='#1f77b4')
                st.pyplot(plt.gcf())
                plt.close()
            except Exception as e:
                st.error(f"Error creating xG distribution plot: {str(e)}")
        
        with dist_col2:
            st.write("**Goalkeeper Pass Length Distribution**")
            try:
                plot_gk_pass_length_distribution(data['passes'], selected_team, color='#ff7f0e')
                st.pyplot(plt.gcf())
                plt.close()
            except Exception as e:
                st.error(f"Error creating GK pass length plot: {str(e)}")

        # let's add another row just like the previous one to include two more distributions: plot_shot_xg_by_type and plot_xg_per_shot_distribution
        dist_col3, dist_col4 = st.columns(2)
        
        with dist_col3:
            st.write("**xG Distribution by Shot Type**")
            try:
                plot_shot_xg_by_type(data['shots'], selected_team, aspect=2.2)
                st.pyplot(plt.gcf())
                plt.close()
            except Exception as e:
                st.error(f"Error creating shot xG by type plot: {str(e)}")
        
        with dist_col4:
            st.write("**xG/Shot**")
            try:
                plot_xg_per_shot_distribution(data['shots'], selected_team, color='purple')
                st.pyplot(plt.gcf())
                plt.close()
            except Exception as e:
                st.error(f"Error creating xG per shot plot: {str(e)}")
    
    # Tab 3: Passes and Shots Location
    with tab3:
        st.markdown('<div class="tab-header">Passes and Shots Location Analysis</div>', unsafe_allow_html=True)

        st.subheader("Metrics")

        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Shot per Game", f"{team_metrics['shooting']['shots_per_match']:.1f}")
        
        with col2:
            # Calculate passes per game manually if not available
            total_passes = team_metrics['passing']['total_passes']
            matches_played = team_metrics['defensive']['matches_played']
            passes_per_game = total_passes / matches_played if matches_played > 0 else 0
            st.metric("Passes per Game", f"{passes_per_game:.1f}")
        
        with col3:
            # Calculate pressure actions if available
            if selected_team in data['pressure']['team'].values:
                pressure_actions = len(data['pressure'][data['pressure']['team'] == selected_team])
                st.metric("Pressure Actions", pressure_actions)
            else:
                st.metric("Pressure Actions", "N/A")
        
        with col4:
            # Calculate penalties taken
            if selected_team in data['events']['team'].values:
                penalties_taken = len(data['events'][(data['events']['team'] == selected_team) & (data['events']['shot_type'] == 'Penalty')])
                st.metric("Penalties Taken", penalties_taken)
            else:
                st.metric("Penalties Taken", "N/A")
        
        # Shot map
        st.subheader("Shot Map")
        shot_map_fig = create_shot_map(data['shots'], selected_team)
        st.pyplot(shot_map_fig)
        
        # Attacking passes
        st.subheader("Attacking Passes")
        attacking_passes_fig, _ = plot_attacking_passes(data['passes'], selected_team)
        st.pyplot(attacking_passes_fig)

        # Passes completed vs passes under pressure
        st.subheader("Passes Completed vs Passes Under Pressure")
        try:
            under_pressure_result = plot_player_passing_under_pressure(data['passes'], selected_team)
            # Handle case where function returns tuple (fig, ax)
            if isinstance(under_pressure_result, tuple):
                under_pressure_fig = under_pressure_result[0]
            else:
                under_pressure_fig = under_pressure_result
            st.pyplot(under_pressure_fig)
        except Exception as e:
            st.error(f"Error creating under pressure plot: {str(e)}")
    
    # Tab 4: Defensive Metrics
    with tab4:
        st.markdown(f'<div class="tab-header">Defensive Metrics for {selected_team}</div>', unsafe_allow_html=True)
        
# Additional defensive metrics
        st.subheader("Defensive Performance")
        
        def_col1, def_col2, def_col3, def_col4 = st.columns(4)
        
        with def_col1:
            if selected_team in data['blocks']['team'].values:
                blocks = len(data['blocks'][data['blocks']['team'] == selected_team])
                st.metric("Blocks", blocks)
            else:
                st.metric("Blocks", "N/A")
        
        with def_col2:
            if selected_team in data['events']['team'].values:
                clearances = len(data['events'][(data['events']['type'] == 'Clearance') & (data['events']['team'] == selected_team)])
                st.metric("Clearances", clearances)
            else:
                st.metric("Clearances", "N/A")
        
        with def_col3:
            if selected_team in data['pressure']['team'].values:
                pressure_events = len(data['pressure'][data['pressure']['team'] == selected_team])
                st.metric("Pressure Events", pressure_events)
            else:
                st.metric("Pressure Events", "N/A")
        
        with def_col4:
            if selected_team in data['dribbled_pasts']['team'].values:
                dribbled_pasts = len(data['dribbled_pasts'][data['dribbled_pasts']['team'] == selected_team])
                st.metric("Dribbled Past", dribbled_pasts)
            else:
                st.metric("Dribbled Past", "N/A")
        
        # Second row of defensive metrics
        def_col5, def_col6, def_col7, def_col8 = st.columns(4)
        
        with def_col5:
            if selected_team in data['interceptions']['team'].values:
                interceptions = len(data['interceptions'][data['interceptions']['team'] == selected_team])
                st.metric("Interceptions", interceptions)
            else:
                st.metric("Interceptions", "N/A")
        
        with def_col6:
            if selected_team in data['fouls']['team'].values:
                fouls_committed = len(data['fouls'][data['fouls']['team'] == selected_team])
                st.metric("Fouls Committed", fouls_committed)
            else:
                st.metric("Fouls Committed", "N/A")
        
        with def_col7:
            # Fouls against (fouls committed by opponents against this team)
            if selected_team in data['fouls']['team'].values and len(data['fouls']) > 0:
                # Get all matches where the team played
                team_matches = data['shots'][data['shots']['team'] == selected_team]['match_id'].unique()
                # Count fouls committed by opponents in those matches
                fouls_against = len(data['fouls'][
                    (data['fouls']['match_id'].isin(team_matches)) & 
                    (data['fouls']['team'] != selected_team)
                ])
                st.metric("Fouls Against", fouls_against)
            else:
                st.metric("Fouls Against", "N/A")
        
        with def_col8:
            if selected_team in data['events']['team'].values:
                cards_received = len(data['events'][(data['events']['foul_committed_card'].notnull()) & (data['events']['team'] == selected_team)])
                st.metric("Cards Received", cards_received)
            else:
                st.metric("Cards Received", "N/A")
        
        # plot_shots_against
        st.subheader("Shots Against")
        try:
            plot_shots_against(selected_team)
            st.pyplot(plt.gcf())
            plt.close()
        except Exception as e:
            st.error(f"Error creating shots against plot: {str(e)}")

        # plot_progression_against_team
        st.subheader("Progression Against")
        try:
            plot_progression_against_team(selected_team)
            st.pyplot(plt.gcf())
            plt.close()
        except Exception as e:
            st.error(f"Error creating progression against plot: {str(e)}")  

        # plot_pressure_heatmap
        st.subheader("Pressure Heatmap")
        try:
            plot_pressure_heatmap(data['pressure'], selected_team)
            st.pyplot(plt.gcf())
            plt.close()
        except Exception as e:
            st.error(f"Error creating pressure heatmap plot: {str(e)}")    


    # Tab 5: Team Comparison
    with tab5:
        st.markdown('<div class="tab-header">Team Comparison</div>', unsafe_allow_html=True)
        
        # Team selection for comparison
        comparison_team = st.selectbox("Select team to compare with:", 
                                     [team for team in teams if team != selected_team])
        
        if comparison_team:
            # Get comparison team metrics
            comparison_metrics = get_team_metrics(data['passes'], data['shots'], comparison_team)
            
            if comparison_metrics:
                st.subheader(f"Comparing {selected_team} and {comparison_team}")
                
                # Comparison metrics
                comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
                
                with comp_col1:
                    st.write("**Pass Completion Rate**")
                    team1_pcr = team_metrics['passing']['pass_completion_rate']
                    team2_pcr = comparison_metrics['passing']['pass_completion_rate']
                    st.write(f"{selected_team}: {team1_pcr:.1f}%")
                    st.write(f"{comparison_team}: {team2_pcr:.1f}%")
                
                with comp_col2:
                    st.write("**Shots on Target %**")
                    team1_sot = team_metrics['shooting']['shots_on_target_percentage']
                    team2_sot = comparison_metrics['shooting']['shots_on_target_percentage']
                    st.write(f"{selected_team}: {team1_sot:.1f}%")
                    st.write(f"{comparison_team}: {team2_sot:.1f}%")
                
                with comp_col3:
                    st.write("**Average xG**")
                    team1_xg = team_metrics['shooting']['non_penalty_avg_xG']
                    team2_xg = comparison_metrics['shooting']['non_penalty_avg_xG']
                    st.write(f"{selected_team}: {team1_xg:.3f}")
                    st.write(f"{comparison_team}: {team2_xg:.3f}")
                
                with comp_col4:
                    st.write("**Goals Scored**")
                    team1_goals = team_metrics['shooting']['goals']
                    team2_goals = comparison_metrics['shooting']['goals']
                    st.write(f"{selected_team}: {team1_goals}")
                    st.write(f"{comparison_team}: {team2_goals}")
                
                # xG Distribution comparison
                st.subheader("xG Distribution Comparison")
                
                # Create xG distribution comparison
                try:
                    teams_to_compare = [selected_team, comparison_team]
                    colors = ['#1f77b4', '#ff7f0e']
                    plot_multiple_team_xg_distributions(data['shots'], teams_to_compare, colors)
                    st.pyplot(plt.gcf())
                    plt.close()
                except Exception as e:
                    st.error(f"Error creating xG distribution comparison: {str(e)}")

                # Team touch comparison (plot_team_touch_comparison)
                st.subheader("Team Touch Comparison")

                try:
                    plot_team_touch_comparison(selected_team, comparison_team, data['events'])
                    st.pyplot(plt.gcf())
                    plt.close()
                except Exception as e:
                    st.error(f"Error creating team touch comparison: {str(e)}")
                
                # Pass Pressure comparison (plot_team_pressure_passing_comparison   )
                st.subheader("Pass Pressure Analysis Comparison")

                try:
                    plot_team_pressure_passing_comparison(data['passes'])
                    st.pyplot(plt.gcf())
                    plt.close()
                except Exception as e:
                    st.error(f"Error creating pass pressure comparison: {str(e)}")

                # Pressure events comparison (plot_pressure_events)
                st.subheader("Pressure Events Comparison")

                try:
                    plot_pressure_events(data['pressure'])
                    st.pyplot(plt.gcf())
                    plt.close()
                except Exception as e:
                    st.error(f"Error creating pressure events comparison: {str(e)}")



                # # Match passes comparison (plot_match_passes) 

                # ### Error creating match passes comparison: name 'match_id' is not defined

                # st.subheader("Match Passes Comparison")

                # try:
                #     plot_match_passes(match_id, events_df)
                #     st.pyplot(plt.gcf())
                #     plt.close()
                # except Exception as e:
                #     st.error(f"Error creating match passes comparison: {str(e)}")


if __name__ == "__main__":
    main()
