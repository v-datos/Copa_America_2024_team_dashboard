# Team Analisis Dashboard for Copa America 2024
# Import necessary libraries
from statsbombpy import sb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from highlight_text import ax_text, fig_text
import plotly.express as px
from mplsoccer import Pitch, VerticalPitch
from mplsoccer import Radar, FontManager # grid
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
from typing import Dict, Union, List


# Load custom fonts
font_path="Play-Regular.ttf"
font_play = FontProperties(fname=font_path)
font_play_bold = FontProperties(fname=font_path, weight='bold')

# Load Copa America 2024 data
events_df = pd.read_csv('https://drive.google.com/uc?export=download&id=1hD7P5uCluMfOnM1q--7e4XSEbxZT3rnU', low_memory=False)
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

# Functions to calculate and print metrics for each team.

def analyze_team_passes(pass_df: pd.DataFrame, team: str) -> Dict[str, Union[int, float]]:
    """
    Analyze passing statistics for a given team.
    
    Parameters
    ----------
    pass_df : pd.DataFrame
        DataFrame containing passing data with columns:
        ['team', 'pass_outcome', 'pass_cross', 'pass_cut_back', 'pass_switch',
         'pass_through_ball', 'pass_shot_assist', 'pass_goal_assist',
         'under_pressure', 'position', 'pass_length']
    team : str
        Name of the team to analyze
    
    Returns
    -------
    Dict[str, Union[int, float]]
        Dictionary containing the following metrics:
        - total_passes: Total number of passes attempted
        - completed_passes: Number of completed passes
        - pass_completion_rate: Percentage of passes completed
        - cross_passes: Number of crosses
        - cutback_passes: Number of cutback passes
        - switch_passes: Number of switch passes
        - through_ball_passes: Number of through balls
        - shot_assist_passes: Number of shot assists
        - goal_assist_passes: Number of goal assists
        - under_pressure_passes: Number of passes under pressure
        - under_pressure_percentage: Percentage of passes under pressure
        - cross_percentage: Percentage of passes that are crosses
        - cutback_percentage: Percentage of passes that are cutbacks
        - through_ball_percentage: Percentage of through balls
        - goalkeeper_pass_avg_length: Average length of goalkeeper passes
    
    Raises
    ------
    ValueError
        If team is not found in the dataset
    """
    # Validate input
    if team not in pass_df['team'].unique():
        raise ValueError(f"Team '{team}' not found in the dataset")
    
    # Filter passes for the given team
    team_passes = pass_df[pass_df['team'] == team].copy()
    
    # Calculate basic metrics
    total_passes = len(team_passes)
    
    # Handle edge case where team has no passes
    if total_passes == 0:
        return {
            'error': f"No passes found for team {team}"
        }
    
    # Calculate all metrics
    metrics = {
        'total_passes': total_passes,
        'passes_per_match': total_passes / team_passes['match_id'].nunique(),
        'completed_passes': len(team_passes[team_passes['pass_outcome'] == 'Completed']),
        'completed_passes_per_match': len(team_passes[team_passes['pass_outcome'] == 'Completed']) / team_passes['match_id'].nunique(),
        'cross_passes': len(team_passes[team_passes['pass_cross'] == True]),
        'cutback_passes': len(team_passes[team_passes['pass_cut_back'] == True]),
        'switch_passes': len(team_passes[team_passes['pass_switch'] == True]),
        'through_ball_passes': len(team_passes[team_passes['pass_through_ball'] == True]),
        'shot_assist_passes': len(team_passes[team_passes['pass_shot_assist'] == True]),
        'goal_assist_passes': len(team_passes[team_passes['pass_goal_assist'] == True]),
        'under_pressure_passes': len(team_passes[team_passes['under_pressure'] == True])
    }
    
    # Calculate percentages
    metrics.update({
        'pass_completion_rate': (metrics['completed_passes'] / total_passes) * 100,
        'under_pressure_percentage': (metrics['under_pressure_passes'] / total_passes) * 100,
        'cross_percentage': (metrics['cross_passes'] / total_passes) * 100,
        'cutback_percentage': (metrics['cutback_passes'] / total_passes) * 100,
        'through_ball_percentage': (metrics['through_ball_passes'] / total_passes) * 100
    })
    
    # Calculate goalkeeper metrics
    goalkeeper_passes = team_passes[team_passes['position'] == 'Goalkeeper']
    metrics['goalkeeper_pass_avg_length'] = goalkeeper_passes['pass_length'].mean() \
        if not goalkeeper_passes.empty else np.nan
    
    return metrics

def print_team_pass_analysis(metrics: Dict[str, Union[int, float]], team: str) -> None:
    """
    Print formatted pass analysis metrics for a team.
    
    Parameters
    ----------
    metrics : Dict[str, Union[int, float]]
        Dictionary of passing metrics from analyze_team_passes
    team : str
        Name of the team
    """
    print(f"\nPass metrics for {team}:")
    print(f"Total passes: {metrics['total_passes']}")
    print(f"Passes per match: {metrics['passes_per_match']:.2f}")
    print(f"Completed passes: {metrics['completed_passes']}")
    print(f"Completed passes per match: {metrics['completed_passes_per_match']:.2f}")
    print(f"Pass completion rate: {metrics['pass_completion_rate']:.2f}%")
    print(f"Cross passes: {metrics['cross_passes']}")
    print(f"Cut back passes: {metrics['cutback_passes']}")
    print(f"Cross percentage: {metrics['cross_percentage']:.2f}%")
    print(f"Cutback percentage: {metrics['cutback_percentage']:.2f}%")
    print(f"Switch passes: {metrics['switch_passes']}")
    print(f"Through ball passes: {metrics['through_ball_passes']}")
    print(f"Through ball percentage: {metrics['through_ball_percentage']:.2f}%")
    print(f"Shot assist passes: {metrics['shot_assist_passes']}")
    print(f"Goal assist passes: {metrics['goal_assist_passes']}")
    print(f"Passes under pressure: {metrics['under_pressure_passes']}")
    print(f"Percentage of passes under pressure: {metrics['under_pressure_percentage']:.2f}%")
    if not np.isnan(metrics['goalkeeper_pass_avg_length']):
        print(f"Goalkeeper pass length average: {metrics['goalkeeper_pass_avg_length']:.3f} m")

# Example usage:
# metrics = analyze_team_passes(pass_df, 'Argentina')
# print_team_pass_analysis(metrics, 'Argentina')


def analyze_team_shots(shot_df: pd.DataFrame, team: str) -> Dict[str, Union[int, float]]:
    """
    Analyze shooting statistics for a given team.
    
    Parameters
    ----------
    shot_df : pd.DataFrame
        DataFrame containing shot data with columns:
        ['team', 'shot_type', 'shot_outcome', 'play_pattern', 'under_pressure',
         'shot_statsbomb_xg']
    team : str
        Name of the team to analyze
    
    Returns
    -------
    Dict[str, Union[int, float]]
        Dictionary containing the following metrics:
        - total_shots: Total number of non-penalty shots
        - shots_on_target: Number of shots on target
        - goals: Number of goals scored
        - counter_shots: Number of shots from counter attacks
        - shots_under_pressure: Number of shots under pressure
        - shots_on_target_percentage: Percentage of shots on target
        - goals_percentage: Percentage of shots resulting in goals
        - counter_shots_percentage: Percentage of shots from counter attacks
        - shots_under_pressure_percentage: Percentage of shots under pressure
        - total_xG: Total expected goals (excluding penalties)
        - avg_xG_open_play: Average xG for open play shots
        - non_penalty_avg_xG: Average xG excluding penalties
        - avg_xG_set_piece: Average xG for set pieces
        - xg_per_shot: Average xG per shot
    
    Raises
    ------
    ValueError
        If team is not found in the dataset
    """
    # Validate input
    if team not in shot_df['team'].unique():
        raise ValueError(f"Team '{team}' not found in the dataset")
    
    # Filter shots for the given team
    team_shots = shot_df[shot_df['team'] == team].copy()

    
    # Filter non-penalty shots
    non_penalty_shots = team_shots[team_shots['shot_type'] != 'Penalty']
    open_play_shots = team_shots[team_shots['shot_type'] == 'Open Play']
    
    # Handle edge case where team has no shots
    if len(non_penalty_shots) == 0:
        return {
            'error': f"No shots found for team {team}"
        }
    
    # Basic shot metrics
    metrics = {
        'total_shots': len(non_penalty_shots),
        'shots_per_match': len(non_penalty_shots) / team_shots['match_id'].nunique(),
        'shots_on_target': len(team_shots[
            (team_shots['shot_outcome'].isin(['Saved', 'Goal']))
        ]),
        'goals': len(team_shots[team_shots['shot_outcome'] == 'Goal']),
        'counter_shots': len(open_play_shots[
            open_play_shots['play_pattern'] == 'From Counter'
        ]),
        'counter_shots_per_match': len(open_play_shots) / team_shots['match_id'].nunique(),
        'shots_under_pressure': len(open_play_shots[
            open_play_shots['under_pressure'] == True
        ]),
        'shots_under_pressure_per_match': len(open_play_shots) / team_shots['match_id'].nunique()                                  
    }
    
    # Calculate percentages
    total_shots = metrics['total_shots']
    metrics.update({
        'shots_on_target_percentage': (metrics['shots_on_target'] / total_shots) * 100,
        'goals_percentage': (metrics['goals'] / total_shots) * 100,
        'counter_shots_percentage': (metrics['counter_shots'] / total_shots) * 100,
        'shots_under_pressure_percentage': (metrics['shots_under_pressure'] / total_shots) * 100
    })
    
    # Expected goals (xG) metrics
    metrics.update({
        'total_xG': non_penalty_shots['shot_statsbomb_xg'].sum(),
        'avg_xG_open_play': open_play_shots['shot_statsbomb_xg'].mean(),
        'non_penalty_avg_xG': non_penalty_shots['shot_statsbomb_xg'].mean(),
        'avg_xG_set_piece': team_shots[
            team_shots['shot_type'] == 'Free Kick'
        ]['shot_statsbomb_xg'].mean(),
    })
    
    # Calculate xG per shot
    metrics['xg_per_shot'] = metrics['non_penalty_avg_xG'] / total_shots if total_shots > 0 else 0
    
    # Replace NaN values with 0
    metrics = {k: 0 if pd.isna(v) else v for k, v in metrics.items()}
    
    return metrics

def print_team_shot_analysis(metrics: Dict[str, Union[int, float]], team: str) -> None:
    """
    Print formatted shot analysis metrics for a team.
    
    Parameters
    ----------
    metrics : Dict[str, Union[int, float]]
        Dictionary of shooting metrics from analyze_team_shots
    team : str
        Name of the team
    """
    print(f"\nShot metrics for {team}:")
    print(f"Total shots (not including penalties): {metrics['total_shots']}")
    print(f"Shots per match: {metrics['shots_per_match']:.2f}")
    print(f"Shots on target: {metrics['shots_on_target']}")
    print(f"Shots on target percentage: {metrics['shots_on_target_percentage']:.2f}%")
    print(f"Goals: {metrics['goals']} ({metrics['goals_percentage']:.2f}%)")
    print(f"Shots from counter: {metrics['counter_shots']}")
    print(f"Shots from counter per match: {metrics['counter_shots_per_match']:.2f}")
    print(f"Shots from counter percentage: {metrics['counter_shots_percentage']:.2f}%")
    print(f"Shots under pressure: {metrics['shots_under_pressure']}")
    print(f"Shots under pressure per match: {metrics['shots_under_pressure_per_match']:.2f}")
    print(f"Shots under pressure percentage: {metrics['shots_under_pressure_percentage']:.2f}%")
    print(f"Total xG: {metrics['total_xG']:.3f}")
    print(f"Average xG (open play only): {metrics['avg_xG_open_play']:.3f}")
    print(f"Average xG: {metrics['non_penalty_avg_xG']:.3f}")
    print(f"Average xG when shot is a set piece: {metrics['avg_xG_set_piece']:.3f}")
    print(f"Average xG/shot: {metrics['xg_per_shot']:.3f}")

# Example usage:
# shot_metrics = analyze_team_shots(shot_df, 'Argentina')
# print_team_shot_analysis(shot_metrics, 'Argentina')

# Funtions to analyze team shots

def analyze_team_shots_against(shot_df: pd.DataFrame, team: str) -> Dict[str, Union[int, float]]:
    """
    Analyze defensive shooting statistics against a given team.
    
    Parameters
    ----------
    shot_df : pd.DataFrame
        DataFrame containing shot data with columns:
        ['team', 'match_id', 'shot_type', 'shot_outcome', 'shot_statsbomb_xg']
    team : str
        Name of the team to analyze
    
    Returns
    -------
    Dict[str, Union[int, float]]
        Dictionary containing the following defensive metrics:
        - matches_played: Number of matches played
        - total_shots_against: Total number of non-penalty shots faced
        - goals_conceded: Number of goals conceded
        - goals_conceded_percentage: Percentage of shots that resulted in goals
        - total_xg_against: Total expected goals against
        - avg_xg_per_shot_against: Average xG per shot faced
        - avg_xg_goals_conceded: Average xG of conceded goals
    
    Raises
    ------
    ValueError
        If team is not found in the dataset or if no matches are found for the team
    """
    # Validate input
    if team not in shot_df['team'].unique():
        raise ValueError(f"Team '{team}' not found in the dataset")
    
    # Get all match IDs where this team played
    team_matches = shot_df[shot_df['team'] == team]['match_id'].unique()
    
    if len(team_matches) == 0:
        raise ValueError(f"No matches found for team {team}")
    
    # Get shots against this team (only in matches they played)
    shots_against = shot_df[
        (shot_df['match_id'].isin(team_matches)) & # Only in matches they played
        (shot_df['team'] != team) & # By opponent team
        (shot_df['shot_type'] != 'Penalty') # Excluding penalties
    ].copy()
    
    # Initialize metrics dictionary
    metrics = {
        'matches_played': len(team_matches),
        'total_shots_against': len(shots_against),
        'shots_against_per_match': len(shots_against) / len(team_matches) if len(team_matches) > 0 else 0
    }
    
    # Handle edge case where team has no shots against
    if metrics['total_shots_against'] == 0:
        return {
            **metrics,
            'goals_conceded': 0,
            'goals_conceded_percentage': 0,
            'total_xg_against': 0,
            'avg_xg_per_shot_against': 0,
            'avg_xg_goals_conceded': 0
        }
    
    # Calculate goals conceded metrics
    goals_against = shots_against[shots_against['shot_outcome'] == 'Goal']
    metrics.update({
        'goals_conceded': len(goals_against),
        'goals_conceded_percentage': (len(goals_against) / metrics['total_shots_against']) * 100,
    })
    
    # Calculate xG metrics
    metrics.update({
        'total_xg_against': shots_against['shot_statsbomb_xg'].sum(),
        'avg_xg_per_shot_against': shots_against['shot_statsbomb_xg'].mean(),
        'avg_xg_goals_conceded': goals_against['shot_statsbomb_xg'].mean() 
            if len(goals_against) > 0 else 0
    })
    
    # Round floating point values
    for key in ['avg_xg_per_shot_against', 'avg_xg_goals_conceded']:
        metrics[key] = round(metrics[key], 3)
    
    return metrics

def print_team_defensive_analysis(metrics: Dict[str, Union[int, float]], team: str) -> None:
    """
    Print formatted defensive analysis metrics for a team.
    
    Parameters
    ----------
    metrics : Dict[str, Union[int, float]]
        Dictionary of defensive metrics from analyze_team_shots_against
    team : str
        Name of the team
    """
    print(f"\nDefensive Metrics for {team}:")
    print(f"Matches played: {metrics['matches_played']}")
    print(f"Total shots against: {metrics['total_shots_against']}")
    print(f"Shots against per match: {metrics['shots_against_per_match']:.2f}")
    print(f"Goals conceded: {metrics['goals_conceded']} "
          f"({metrics['goals_conceded_percentage']:.2f}%)")
    print(f"Total xG conceded: {metrics['total_xg_against']:.3f}")
    print(f"Average xG per shot against: {metrics['avg_xg_per_shot_against']:.3f}")
    print(f"Average xG of conceded goals: {metrics['avg_xg_goals_conceded']:.3f}")

# Example usage:
# defensive_metrics = analyze_team_shots_against(shot_df, 'Argentina')
# print_team_defensive_analysis(defensive_metrics, 'Argentina')

from typing import Dict, Union, List
import pandas as pd

def analyze_team_metrics(pass_df: pd.DataFrame, 
                        shot_df: pd.DataFrame, 
                        team: str,
                        return_dict: bool = False) -> Union[Dict[str, Dict], None]:
    """
    Comprehensive analysis of a team's performance metrics including passing,
    shooting, and defensive statistics.
    
    Parameters
    ----------
    pass_df : pd.DataFrame
        DataFrame containing passing data
    shot_df : pd.DataFrame
        DataFrame containing shooting data
    team : str
        Name of the team to analyze
    return_dict : bool, optional
        If True, returns a dictionary with all metrics, by default False
    
    Returns
    -------
    Union[Dict[str, Dict], None]
        If return_dict is True, returns a dictionary containing:
        - 'passing': Dictionary of passing metrics
        - 'shooting': Dictionary of shooting metrics
        - 'defensive': Dictionary of defensive metrics
        If return_dict is False, returns None (prints results instead)
    
    Raises
    ------
    ValueError
        If team is not found in either dataset
    """
    # Validate team exists in both datasets
    if team not in pass_df['team'].unique() or team not in shot_df['team'].unique():
        raise ValueError(f"Team '{team}' not found in one or both datasets")
    
    try:
        # Collect all metrics
        metrics = {
            'passing': analyze_team_passes(pass_df, team),
            'shooting': analyze_team_shots(shot_df, team),
            'defensive': analyze_team_shots_against(shot_df, team)
        }
        
        if not return_dict:
            # Print all metrics with clear section headers
            print(f"\n{'='*50}")
            print(f"COMPLETE ANALYSIS FOR {team.upper()}")
            print(f"{'='*50}\n")
            
            # Print passing metrics
            print("\nPASSING METRICS")
            print("-" * 20)
            print_team_pass_analysis(metrics['passing'], team)
            
            # Print shooting metrics
            print("\nSHOOTING METRICS")
            print("-" * 20)
            print_team_shot_analysis(metrics['shooting'], team)
            
            # Print defensive metrics
            print("\nDEFENSIVE METRICS")
            print("-" * 20)
            print_team_defensive_analysis(metrics['defensive'], team)
            
            return None
        
        return metrics
    
    except Exception as e:
        print(f"Error analyzing {team}: {str(e)}")
        return None


def analyze_all_teams(pass_df: pd.DataFrame, 
                     shot_df: pd.DataFrame, 
                     return_dict: bool = False) -> Union[Dict[str, Dict], None]:
    """
    Analyze metrics for all teams in the dataset.
    
    Parameters
    ----------
    pass_df : pd.DataFrame
        DataFrame containing passing data
    shot_df : pd.DataFrame
        DataFrame containing shooting data
    return_dict : bool, optional
        If True, returns a dictionary with all teams' metrics, by default False
    
    Returns
    -------
    Union[Dict[str, Dict], None]
        If return_dict is True, returns a dictionary with team names as keys
        and their respective metrics as values
    """
    teams = sorted(shot_df['team'].unique())
    
    if return_dict:
        return {
            team: analyze_team_metrics(pass_df, shot_df, team, return_dict=True)
            for team in teams
        }
    
    for team in teams:
        analyze_team_metrics(pass_df, shot_df, team)
        print("\n" + "="*70 + "\n")  # Separator between teams

# Example usage:
# Basic usage - print all metrics
# analyze_all_teams(pass_df, shot_df)

# Get metrics as dictionary for further analysis
# all_team_metrics = analyze_all_teams(pass_df, shot_df, return_dict=True)

# Analyze single team
# team_metrics = analyze_team_metrics(pass_df, shot_df, 'Argentina', return_dict=True)

# Funtions to create plots and visualizations for each team.

def create_team_radar_chart(team_stats: list[float], 
                           font_play, 
                           team_name: str = "",
                           team_color: str = '#aa65b2',
                           ring_color: str = '#66d8ba',
                           inner_ring_face: str = '#ffb2b2',
                           inner_ring_edge: str = '#fc5f5f') -> tuple:
    """
    Create a radar chart for team statistics.
    
    Parameters
    ----------
    team_stats : List[float]
        List of team statistics in the following order:
        [Average xG, Average xG/shot, Total shots, Shots from counter,
         Average xG set piece, Shots under pressure, Through ball percentage,
         Goalkeeper pass length average, Cross percentage]
    font_play : font
        Font to use for labels
    team_name : str, optional
        Name of the team for title, by default ""
    team_color : str, optional
        Color for team statistics area, by default '#aa65b2'
    ring_color : str, optional
        Color for outer rings, by default '#66d8ba'
    inner_ring_face : str, optional
        Face color for inner rings, by default '#ffb2b2'
    inner_ring_edge : str, optional
        Edge color for inner rings, by default '#fc5f5f'
    
    Returns
    -------
    tuple
        (fig, ax) - matplotlib figure and axis objects
    """
    # Define parameters and their ranges
    params = [
        "Non_Penalty xG",
        "Shots on Target %",
        "Shots per Game",
        "Counter Attacking Shots per Game",
        "Set Piece xG",
        "Shots Under Pressure per Game",
        "Through ball %",
        "Goalkeeper Pass Length (avg)",
        "Cross %"
    ]
    
    # Define the lower and upper boundaries
    low = [0.053, 22.22, 3.00, 0, 0.000, 0.33, 0.00, 26.402, 0.55]
    high = [0.141, 44.71, 19.00, 5, 0.058, 7.67, 0.86, 54.922, 3.33]
    
    # Initialize the Radar object
    radar = Radar(
        params,
        low,
        high,
        round_int=[False] * len(params),
        num_rings=4,
        ring_width=1,
        center_circle_radius=1
    )
    
    # Create the figure and axis
    fig, ax = radar.setup_axis()
    
    # Draw the circles
    rings_inner = radar.draw_circles(
        ax=ax,
        facecolor=inner_ring_face,
        edgecolor=inner_ring_edge
    )
    
    # Draw the radar
    radar_output = radar.draw_radar(
        team_stats,
        ax=ax,
        kwargs_radar={'facecolor': team_color},
        kwargs_rings={'facecolor': ring_color}
    )
    radar_poly, rings_outer, vertices = radar_output
    
    # Add the labels
    range_labels = radar.draw_range_labels(
        ax=ax,
        fontsize=14,
        zorder=2.5,
        fontproperties=font_play
    )
    
    param_labels = radar.draw_param_labels(
        ax=ax,
        fontsize=14,
        fontproperties=font_play
    )
    
    # Add title if team name is provided
    if team_name:
        ax.set_title(f"{team_name} Attacking Radar - Copa America 2024", 
                    fontproperties=font_play_bold, 
                    pad=15,
                    fontsize=24)
    
    return fig, ax

def get_team_radar_stats(team_metrics: dict) -> list[float]:
    """
    Extract radar chart statistics from team metrics dictionary.
    
    Parameters
    ----------
    team_metrics : dict
        Dictionary containing team metrics from analyze_team_metrics function
    
    Returns
    -------
    List[float]
        List of statistics in the order required by the radar chart
    """
    shooting = team_metrics['shooting']
    passing = team_metrics['passing']
    
    return [
        shooting['non_penalty_avg_xG'],
        shooting['shots_on_target_percentage'],
        shooting['shots_per_match'],
        shooting['counter_shots_per_match'],
        shooting['avg_xG_set_piece'],
        shooting['shots_under_pressure_per_match'],
        passing['through_ball_percentage'],
        passing['goalkeeper_pass_avg_length'],
        passing['cross_percentage']
    ]

# Example usage:
"""
# For a single team
team_metrics = analyze_team_metrics(pass_df, shot_df, 'Jamaica', return_dict=True)
team_stats = get_team_radar_stats(team_metrics)
fig, ax = create_team_radar_chart(
    team_stats,
    font_play,
    team_name='Jamaica',
    team_color='#aa65b2'  # Custom color for Jamaica
)
plt.show()

# To compare multiple teams
teams_to_compare = ['Jamaica', 'Brazil', 'Argentina']
fig, axes = plt.subplots(1, len(teams_to_compare), figsize=(20, 6))
for team, ax in zip(teams_to_compare, axes):
    metrics = analyze_team_metrics(pass_df, shot_df, team, return_dict=True)
    stats = get_team_radar_stats(metrics)
    create_team_radar_chart(stats, font_play, team_name=team, ax=ax)
plt.tight_layout()
plt.show()
"""

def plot_xg_distribution(shot_df: pd.DataFrame, 
                        team: str,
                        color: str = "purple",
                        height: int = 4,
                        aspect: float = 2.5,
                        font_size: dict = {"title": 20, "labels": 15}) -> None:
    """
    Plot the Expected Goals (xG) distribution for a given team's shots,
    excluding penalties.

    
    Parameters
    ----------
    shot_df : pd.DataFrame
        DataFrame containing shot data with columns:
        ['team', 'shot_type', 'shot_statsbomb_xg']
    team : str
        Name of the team to analyze
    color : str, optional
        Color for the distribution plot, by default "purple"
    height : int, optional
        Height of the plot in inches, by default 4
    aspect : float, optional
        Aspect ratio of the plot (width/height), by default 2.5
    font_size : dict, optional
        Dictionary containing font sizes for title and labels,
        by default {"title": 20, "labels": 15}
    
    Returns
    -------
    g : seaborn.FacetGrid
        The seaborn distribution plot object
    
    Raises
    ------
    ValueError
        If team is not found in the dataset or if no shots are found
    KeyError
        If required columns are missing from the DataFrame
    """
    # Validate input DataFrame has required columns
    required_columns = ['team', 'shot_type', 'shot_statsbomb_xg']
    if not all(col in shot_df.columns for col in required_columns):
        raise KeyError(f"DataFrame missing one or more required columns: {required_columns}")
    
    # Validate team exists in dataset
    if team not in shot_df['team'].unique():
        raise ValueError(f"Team '{team}' not found in the dataset")
    
    # Filter shots for the given team
    team_shots = shot_df[shot_df['team'] == team]
    
    # Filter out penalties
    non_penalty_shots = team_shots[team_shots['shot_type'] != 'Penalty']
    
    # Check if there are any non-penalty shots
    if len(non_penalty_shots) == 0:
        raise ValueError(f"No non-penalty shots found for team {team}")
    
    try:
        # Get xG values
        xg = non_penalty_shots['shot_statsbomb_xg']
        
        # Create the distribution plot
        g = sns.displot(
            data=xg,
            fill=True,
            common_norm=False,
            color=color,
            alpha=0.5,
            linewidth=1,
            kind="kde",
            rug=True,
            height=height,
            aspect=aspect
        )
        
        # Customize the plot using the displot object
        g.fig.suptitle(f'Non-Penalty xG Distribution for {team}', 
                      size=font_size["title"], 
                      y=1.05, fontproperties=font_play_bold)
        g.set_axis_labels('Non-Penalty xG', 'Density', 
                         fontproperties=font_play, fontsize=font_size["labels"])
        
        # Add summary statistics annotations
        stats_text = (
            f'Mean xG: {xg.mean():.3f}\n'
            f'Median xG: {xg.median():.3f}\n'
            f'Std Dev: {xg.std():.3f}\n'
            f'Total Shots: {len(xg)}'
        )
        plt.annotate(stats_text,
                    xy=(0.95, 0.95),
                    xycoords='axes fraction',
                    ha='right',
                    va='top', fontproperties=font_play_bold,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.show()
        return g
        
    except Exception as e:
        plt.close()  # Close any partial figure in case of error
        raise Exception(f"Error creating plot for {team}: {str(e)}")


def plot_xg_per_shot_distribution(shot_df: pd.DataFrame, 
                                team: str,
                                color: str = "red",
                                height: int = 4,
                                aspect: float = 2.5,
                                font_size: dict = {"title": 20, "labels": 15}) -> None:
    """
    Plot the Expected Goals (xG) per shot distribution for a given team.
    
    Parameters
    ----------
    shot_df : pd.DataFrame
        DataFrame containing shot data with columns:
        ['team', 'shot_type', 'shot_statsbomb_xg']
    team : str
        Name of the team to analyze
    color : str, optional
        Color for the distribution plot, by default "red"
    height : int, optional
        Height of the plot in inches, by default 4
    aspect : float, optional
        Aspect ratio of the plot (width/height), by default 2.5
    font_size : dict, optional
        Dictionary containing font sizes for title and labels,
        by default {"title": 20, "labels": 15}
    
    Returns
    -------
    None
        Displays the plot using matplotlib
    
    Raises
    ------
    ValueError
        If team is not found in the dataset or if no shots are found for the team
    KeyError
        If required columns are missing from the DataFrame
    """
    # Validate input DataFrame has required columns
    required_columns = ['team', 'shot_type', 'shot_statsbomb_xg']
    if not all(col in shot_df.columns for col in required_columns):
        raise KeyError(f"DataFrame missing one or more required columns: {required_columns}")
    
    # Validate team exists in dataset
    if team not in shot_df['team'].unique():
        raise ValueError(f"Team '{team}' not found in the dataset")
    
    # Filter shots for the given team
    team_shots = shot_df[shot_df['team'] == team]
    
    # Filter out penalties
    non_penalty_shots = team_shots[team_shots['shot_type'] != 'Penalty']
    
    # Check if there are any non-penalty shots
    if len(non_penalty_shots) == 0:
        raise ValueError(f"No non-penalty shots found for team {team}")
    
    try:
        # Calculate xG per shot
        avg_xg_per_shot = team_shots['shot_statsbomb_xg'] / len(team_shots)
        
        plt.figure(figsize=(12, 6))

        # Create the distribution plot
        g = sns.displot(
            data=avg_xg_per_shot,
            fill=True,
            common_norm=False,
            color=color,
            alpha=0.5,
            linewidth=1,
            kind="kde",
            rug=True,
            height=height,
            aspect=aspect
        )
        
                            # --- Customization ---
        # Set the title using the displot object
        g.fig.suptitle(f'xG/Shot Distribution for {team}', 
                        size=font_size['title'], 
                        fontproperties=font_play_bold,
                        y=1.05)
        
        # Set axis labels using the displot object
        g.set_axis_labels('xG/Shot', 'Density', 
                            fontproperties=font_play, fontsize=font_size['labels'])

        # # Customize the plot
        # plt.title(f'xG/Shot Distribution for {team}', 
        #          size=font_size["title"], 
        #          y=1.05, fontproperties=font_play_bold)
        # plt.xlabel('xG/Shot', size=font_size["labels"], fontproperties=font_play)
        # plt.ylabel('Density', size=font_size["labels"], fontproperties=font_play)
        
        # Add summary statistics annotations
        stats_text = (
            f'Mean xG/shot: {avg_xg_per_shot.mean():.3f}\n'
            f'Median xG/shot: {avg_xg_per_shot.median():.3f}\n'
            f'Std Dev: {avg_xg_per_shot.std():.3f}'
        )
        plt.annotate(stats_text,
                    xy=(0.95, 0.95),
                    xycoords='axes fraction',
                    ha='right',
                    va='top', fontproperties=font_play,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.show()
        
        return g
    
    except Exception as e:
        plt.close()  # Close any partial figure in case of error
        raise Exception(f"Error creating plot for {team}: {str(e)}")

def plot_multiple_team_xg_over_shots_distributions(shot_df: pd.DataFrame, 
                                      teams: list[str],
                                      colors: list[str] = None) -> None:
    """
    Plot xG per shot distributions for multiple teams for comparison.
    
    Parameters
    ----------
    shot_df : pd.DataFrame
        DataFrame containing shot data
    teams : List[str]
        List of team names to compare
    colors : List[str], optional
        List of colors for each team's distribution.
        If None, default color palette will be used.
    
    Returns
    -------
    None
        Displays the plot using matplotlib
    """
    if colors is None:
        colors = sns.color_palette("husl", n_colors=len(teams))
    
    plt.figure(figsize=(12, 6))
    
    for team, color in zip(teams, colors):
        try:
            team_shots = shot_df[
                (shot_df['team'] == team) & 
                (shot_df['shot_type'] != 'Penalty')
            ]
            xg_per_shot = team_shots['shot_statsbomb_xg'] / len(team_shots)
            
            sns.kdeplot(
                data=xg_per_shot,
                label=team,
                color=color,
                fill=True,
                alpha=0.3
            )
        except Exception as e:
            print(f"Error plotting distribution for {team}: {str(e)}")
            continue


    plt.title('xG/Shot Distribution Comparison', size=20, y=1.05, fontproperties=font_play_bold)
    plt.xlabel('xG/Shot', size=15, fontproperties=font_play)
    plt.ylabel('Density', size=15, fontproperties=font_play)
    plt.legend()
    plt.show()


def plot_multiple_team_xg_distributions(shot_df: pd.DataFrame, 
                                      teams: list[str],
                                      colors: list[str] = None) -> None:
    """
    Plot xG distributions for multiple teams for comparison.
    
    Parameters
    ----------
    shot_df : pd.DataFrame
        DataFrame containing shot data
    teams : List[str]
        List of team names to compare
    colors : List[str], optional
        List of colors for each team's distribution.
        If None, default color palette will be used.
    
    Returns
    -------
    None
        Displays the plot using matplotlib
    """
    if colors is None:
        colors = sns.color_palette("husl", n_colors=len(teams))
    
    plt.figure(figsize=(12, 6))

    for team, color in zip(teams, colors):
        try:
            team_shots = shot_df[
                (shot_df['team'] == team) & 
                (shot_df['shot_type'] != 'Penalty')
            ]
            xg = team_shots['shot_statsbomb_xg']
            
            sns.kdeplot(
                data=xg,
                label=team,
                color=color,
                fill=True,
                alpha=0.3
            )
        except Exception as e:
            print(f"Error plotting distribution for {team}: {str(e)}")
            continue
    
    plt.suptitle('Non-Penalty xG Distribution Comparison', size=20, fontproperties=font_play_bold)
    plt.xlabel('xG', fontproperties=font_play, fontsize=15)
    plt.ylabel('Density', fontproperties=font_play, fontsize=15)
    plt.legend(loc='upper right', prop=font_play, fontsize=15)
    plt.show()


'''
def plot_multiple_team_xg_distributions(shot_df: pd.DataFrame, 
                                      teams: list[str],
                                      colors: list[str] = None) -> None:
    """
    Plot xG distributions for multiple teams for comparison.
    
    Parameters
    ----------
    shot_df : pd.DataFrame
        DataFrame containing shot data
    teams : List[str]
        List of team names to compare
    colors : List[str], optional
        List of colors for each team's distribution.
        If None, default color palette will be used.
    
    Returns
    -------
    None
        Displays the plot using matplotlib
    """
    if colors is None:
        colors = sns.color_palette("husl", n_colors=len(teams))
    
    
    
    for team, color in zip(teams, colors):
        try:
            team_shots = shot_df[
                (shot_df['team'] == team) & 
                (shot_df['shot_type'] != 'Penalty')
            ]
            xg = team_shots['shot_statsbomb_xg']
            
            plt.figure(figsize=(12, 6))

            g = sns.kdeplot(
                data=xg,
                label=team,
                color=color,
                fill=True,
                alpha=0.3
            )
    
            g.set_title('Non-Penalty xG Distribution Comparison', size=20, y=1.05, fontproperties=font_play_bold)
            g.set_xlabel('xG', size=15, fontproperties=font_play)
            g.set_ylabel('Density', size=15, fontproperties=font_play)
            g.legend()
            plt.show()

            return g
            
        except Exception as e:
            plt.close()  # Close any partial figure in case of error
            raise Exception(f"Error creating plot for {team}: {str(e)}")
'''

# Example usage:
# # Compare multiple teams
# teams_to_compare = ['Argentina', 'Brazil', 'Uruguay']
# colors = ['skyblue', 'yellow', 'lightblue']
# plot_multiple_team_xg_distributions(shot_df, teams_to_compare, colors)

def plot_gk_pass_length_distribution(pass_df: pd.DataFrame, 
                                   team: str,
                                   color: str = "blue",
                                   height: int = 4,
                                   aspect: float = 2.5,
                                   font_size: dict = {"title": 20, "labels": 15}) -> None:
    """
    Plot the distribution of goalkeeper pass lengths for a given team.
    
    Parameters
    ----------
    pass_df : pd.DataFrame
        DataFrame containing pass data with columns:
        ['team', 'position', 'pass_length']
    team : str
        Name of the team to analyze
    font_play : font
        Font object to use for text elements
    color : str, optional
        Color for the distribution plot, by default "blue"
    height : int, optional
        Height of the plot in inches, by default 4
    aspect : float, optional
        Aspect ratio of the plot (width/height), by default 2.5
    font_size : dict, optional
        Dictionary containing font sizes for title and labels,
        by default {"title": 20, "labels": 15}
    
    Returns
    -------
    g : seaborn.FacetGrid
        The seaborn distribution plot object
        Raises
    ------
    ValueError
        If team is not found in the dataset or if no goalkeeper passes are found
    KeyError
        If required columns are missing from the DataFrame
    """
    # Validate input DataFrame has required columns
    required_columns = ['team', 'position', 'pass_length']
    if not all(col in pass_df.columns for col in required_columns):
        raise KeyError(f"DataFrame missing one or more required columns: {required_columns}")
    
    # Validate team exists in dataset
    if team not in pass_df['team'].unique():
        raise ValueError(f"Team '{team}' not found in the dataset")
    
    # Filter passes for the given team's goalkeeper
    team_passes = pass_df[pass_df['team'] == team]
    gk_pass_lengths = team_passes[team_passes['position'] == 'Goalkeeper']['pass_length']
    
    # Check if there are any goalkeeper passes
    if len(gk_pass_lengths) == 0:
        raise ValueError(f"No goalkeeper passes found for team {team}")
    
    try:
        # Create the distribution plot
        g = sns.displot(
            data=gk_pass_lengths,
            fill=True,
            common_norm=False,
            color=color,
            alpha=0.5,
            linewidth=1,
            kind="kde",
            rug=True,
            height=height,
            aspect=aspect
        )
        
                # --- Customization ---
        # Set the title using the displot object
        g.fig.suptitle(f'Goalkeeper Pass Length Distribution for {team}', 
                        size=font_size['title'], 
                        fontproperties=font_play_bold,
                        y=1.05)
        
        # Set axis labels using the displot object
        g.set_axis_labels('Goalkeeper Pass Length (m)', 'Density', 
                            fontproperties=font_play, fontsize=font_size['labels'])
    
    # # Customize the plot with font_play
    #     plt.title(f'Goalkeeper Pass Length Distribution for {team}', 
    #               size=font_size["title"], 
    #               y=1.05,
    #               fontproperties=font_play)
        
    #     plt.xlabel('Goalkeeper Pass Length (m)', 
    #               size=font_size["labels"],
    #               fontproperties=font_play)
        
    #     plt.ylabel('Density', 
    #               size=font_size["labels"],
    #               fontproperties=font_play)
        
        # Add summary statistics annotations with font_play
        stats_text = (
            f'Mean Length: {gk_pass_lengths.mean():.1f}m\n'
            f'Median Length: {gk_pass_lengths.median():.1f}m\n'
            f'Std Dev: {gk_pass_lengths.std():.1f}m\n'
            f'Total Passes: {len(gk_pass_lengths)}'
        )
        plt.annotate(stats_text,
                    xy=(0.95, 0.95),
                    xycoords='axes fraction',
                    ha='right',
                    va='top',
                    fontproperties=font_play,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.show()
        return g
        
    except Exception as e:
        plt.close()
        raise Exception(f"Error creating plot for {team}: {str(e)}")

def plot_multiple_gk_pass_distributions(pass_df: pd.DataFrame, 
                                      teams: list[str],
                                      colors: list[str] = None) -> None:
    """
    Plot goalkeeper pass length distributions for multiple teams for comparison.
    
    Parameters
    ----------
    pass_df : pd.DataFrame
        DataFrame containing pass data
    teams : List[str]
        List of team names to compare
    font_play : font
        Font object to use for text elements
    colors : List[str], optional
        List of colors for each team's distribution.
        If None, default color palette will be used.
    """
    if colors is None:
        colors = sns.color_palette("husl", n_colors=len(teams))
    
    plt.figure(figsize=(12, 6))
    
    for team, color in zip(teams, colors):
        try:
            team_passes = pass_df[
                (pass_df['team'] == team) & 
                (pass_df['position'] == 'Goalkeeper')
            ]
            gk_pass_lengths = team_passes['pass_length']
            
            if len(gk_pass_lengths) > 0:
                g = sns.kdeplot(
                    data=gk_pass_lengths,
                    label=f"{team} (avg: {gk_pass_lengths.mean():.1f}m)",
                    color=color,
                    fill=True,
                    alpha=0.3
                )
        except Exception as e:
            print(f"Error plotting distribution for {team}: {str(e)}")
            continue
    
    # plt.title('Goalkeeper Pass Length Distribution Comparison', 
    #          size=20, 
    #          y=1.05,
    #          fontproperties=font_play)
    
    # plt.xlabel('Pass Length (m)', 
    #           size=15,
    #           fontproperties=font_play)
    
    # plt.ylabel('Density', 
    #           size=15,
    #           fontproperties=font_play)
    
    # # Add legend with custom font
    # leg = plt.legend(prop=font_play)
    
    # plt.show()

    # --- Customization ---
    # Set the title using the displot object
    g.fig.suptitle(f'Goalkeeper Pass Length Distribution Comparison {team}', 
                    size=font_size['title'], 
                    fontproperties=font_play_bold,
                    y=1.05)
    
    # Set axis labels using the displot object
    g.set_axis_labels('Pass Length (m)', 'Density', 
                        fontproperties=font_play, fontsize=font_size['labels'])
      
    return g.fig

# Example usage:
"""
# Single team analysis
plot_gk_pass_length_distribution(
    pass_df,
    'Argentina',
    font_play,
    color='skyblue',
    height=5,
    aspect=2.0,
    font_size={"title": 22, "labels": 16}
)

# Compare multiple teams
teams_to_compare = ['Argentina', 'Brazil', 'Uruguay']
colors = ['skyblue', 'yellow', 'lightblue']
plot_multiple_gk_pass_distributions(
    pass_df,
    teams_to_compare,
    font_play,
    colors
)
"""


def plot_attacking_passes(pass_df: pd.DataFrame, 
                          team: str,) -> tuple:
    """
    Plots a map of attacking passes (crosses, cutbacks, switches, and through balls) for a given team.

    Parameters
    ----------
    pass_df : pd.DataFrame
        DataFrame containing pass data with columns:
        ['team', 'pass_outcome', 'pass_cross', 'pass_cut_back', 'pass_switch', 
         'pass_through_ball', 'x', 'y', 'pass_end_x', 'pass_end_y']
    team : str
        The name of the team to analyze.

    Returns
    -------
    tuple
        A tuple containing the matplotlib figure and axes objects (fig, axs).
        
    Raises
    ------
    ValueError
        If the team is not found in the dataset.
    """
    if team not in pass_df['team'].unique():
        raise ValueError(f"Team '{team}' not found in the dataset")

    # Filter passes for the selected team
    team_passes = pass_df[pass_df['team'] == team].copy()

    # Filter different types of passes
    completed_passes = team_passes[team_passes['pass_outcome'] == 'Completed']
    cross_passes = team_passes[team_passes['pass_cross'] == True]
    cutback_passes = team_passes[team_passes['pass_cut_back'] == True]
    switch_passes = team_passes[team_passes['pass_switch'] == True]
    through_ball_passes = team_passes[team_passes['pass_through_ball'] == True]

    # Create the pitch
    pitch = VerticalPitch(
        pitch_type='statsbomb',
        line_color='black',
    )

    # Create the figure with subplots
    fig, axs = pitch.grid(ncols=4, endnote_height=0, axis=False)
    fig.set_facecolor("#f4f4f4")

    # Plot arrows for each pass type
    pitch.arrows(
        cross_passes.x, cross_passes.y, cross_passes.pass_end_x, cross_passes.pass_end_y,
        width=1, headwidth=8, headlength=8, color='#d73728', ax=axs['pitch'][0],
        label=f'{team} Cross Passes'
    )
    pitch.arrows(
        cutback_passes.x, cutback_passes.y, cutback_passes.pass_end_x, cutback_passes.pass_end_y,
        width=1, headwidth=8, headlength=8, color='blue', ax=axs['pitch'][1],
        label=f'{team} Cut Back Passes'
    )
    pitch.arrows(
        switch_passes.x, switch_passes.y, switch_passes.pass_end_x, switch_passes.pass_end_y,
        width=1, headwidth=10, headlength=8, color='green', ax=axs['pitch'][2],
        label=f'{team} Switch Passes'
    )
    pitch.arrows(
        through_ball_passes.x, through_ball_passes.y, through_ball_passes.pass_end_x, through_ball_passes.pass_end_y,
        width=1, headwidth=8, headlength=8, color='orange', ax=axs['pitch'][3],
        label=f'{team} Through ball Passes'
    )

    # Add titles and text
    axs['title'].text(
        0.55, 0.6, f'Attacking Passes for {team} - Copa America 2024', fontproperties=font_play_bold, 
        color='#000009', va='center', ha='center', fontsize=36
    )

    axs['title'].text(
        0.06, 0.8, f"Total Passes: {len(team_passes)}",
        color='#000009', va='center', ha='center', fontsize=18, fontproperties=font_play, 
    )

    axs['title'].text(
        0.08, 0.6, f"Completed Passes: {len(completed_passes)}",
        color='#000009', va='center', ha='center', fontsize=18, fontproperties=font_play, 
    )

    axs['title'].text(
        0.08, 0.4, f"Completion Rate: {(len(completed_passes) / len(team_passes) * 100):.2f}%",
        color='#000009', va='center', ha='center', fontsize=18, fontproperties=font_play, 
    )

    axs['title'].text(
        0.12, 0.1, f'Cross Passes: {len(cross_passes)}',
        color='#d73728', va='center', ha='center', fontsize=18, fontproperties=font_play, 
    )

    axs['title'].text(
        0.37, 0.1, f'Cut Back Passes: {len(cutback_passes)}',
        color='blue', va='center', ha='center', fontsize=18, fontproperties=font_play,
    )

    axs['title'].text(
        0.63, 0.1, f'Switch Passes: {len(switch_passes)}',
        color='green', va='center', ha='center', fontsize=18, fontproperties=font_play,
    )

    axs['title'].text(
        0.89, 0.1, f'Through Balls: {len(through_ball_passes)}',
        color='orange', va='center', ha='center', fontsize=18, fontproperties=font_play,
    )


    return fig, axs

# Example usage:
# fig, axs = plot_attacking_passes(pass_df, 'Argentina')
# plt.show()


def plot_team_xg_trend(team: str):
    """
    Plots the trend of Expected Goals (xG) for and against a specific team 
    across all their matches in the tournament.

    This function relies on 'copa_america_games' and 'shot_df' DataFrames
    being available in the global scope.

    Parameters
    ----------
    team : str
        The name of the team to analyze.
        
    Raises
    ------
    ValueError
        If the specified team is not found in the match data.
    """
    # Validate that the team exists in the data
    if team not in copa_america_games['home_team'].unique() and \
       team not in copa_america_games['away_team'].unique():
        raise ValueError(f"Team '{team}' not found in the dataset.")

    # Get all match IDs and dates for the selected team
    team_matches_df = copa_america_games[
        (copa_america_games['home_team'] == team) | (copa_america_games['away_team'] == team)
    ]
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
    
    # --- Plotting ---
 
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot xG for and its average
    ax.plot(match_dates, match_xg, 'o-', color='green', label='xG', linewidth=2)
    ax.axhline(y=np.mean(match_xg), color='green', linestyle='--', alpha=0.5)
    
    # Plot xG against and its average
    ax.plot(match_dates, match_xg_against, 'o-', color='purple', label='xG Conceded', linewidth=2)
    ax.axhline(y=np.mean(match_xg_against), color='purple', linestyle='--', alpha=0.5)
    
    # --- Customization ---
    ax.set_title(f'xG Trend for {team} - Copa America 2024', fontsize=16, pad=20, weight='bold', fontproperties=font_play_bold)
    ax.set_xlabel('Match Date', fontsize=12, fontproperties=font_play)
    ax.set_ylabel('Expected Goals (xG)', fontsize=12, fontproperties=font_play)
    
    # Improve x-axis date formatting
    fig.autofmt_xdate(rotation=45)
    
    # Create a clear, custom legend
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='xG'),
        Line2D([0], [0], color='purple', lw=2, label='xG Conceded'),
        Line2D([0], [0], color='green', lw=2, linestyle='--', label='Avg xG', alpha=0.7),
        Line2D([0], [0], color='purple', lw=2, linestyle='--', label='Avg xG Conceded', alpha=0.7)
    ]
    ax.legend(handles=legend_elements, loc='upper center', ncol=4)
    
    # Ensure the layout is clean and labels are not cut off
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig

# Example usage:
# plot_team_xg_trend('Argentina')



def plot_shot_xg_by_type(shot_df: pd.DataFrame, 
                        team: str,
                        height: int = 4,
                        aspect: float = 2.5,
                        font_size: dict = {"title": 20, "labels": 15}) -> None:
    """
    Plots the distribution of Expected Goals (xG) for a given team, 
    categorized by the type of shot.

    This function creates a kernel density estimate (KDE) plot to show how xG values 
    are distributed across different shot types (e.g., Open Play, Free Kick, Penalty).

    This function relies on 'shot_df', 'font_play', and 'font_play_bold' 
    being available in the global scope.

    Parameters
    ----------
    team : str
        The name of the team to analyze.
        
    Raises
    ------
    ValueError
        If the specified team is not found in the shot data.
    """
    # Validate that the team exists in the DataFrame
    if team not in shot_df['team'].unique():
        raise ValueError(f"Team '{team}' not found in the dataset.")

    # Filter the DataFrame for the selected team
    team_shots = shot_df[shot_df['team'] == team]

    try:
        # Remove penalty shots from the data
        team_shots = team_shots[team_shots['shot_type'] != 'Penalty']
        
        # --- Plotting ---
        # Create a distribution plot using seaborn's displot
        g = sns.displot(
            data=team_shots,
            x="shot_statsbomb_xg",
            hue="shot_type",
            kind="kde",
            fill=True,
            common_norm=False,
            palette="icefire",
            alpha=0.5,
            linewidth=1,
            height=height,
            aspect=aspect,
            rug=True
        )
        
        # --- Customization ---
        # Set the title using the displot object
        g.fig.suptitle(f'xG Distribution by Shot Type for {team}', 
                      size=font_size['title'], 
                      fontproperties=font_play_bold,
                      y=1.05)
        
        # Set axis labels using the displot object
        g.set_axis_labels('Expected Goals (xG)', 'Density', 
                         fontproperties=font_play, fontsize=font_size['labels'])
        
        # Set legend title
        g.legend.set_title("Shot Type")
        
        return g.fig
        
    except Exception as e:
        plt.close()
        raise Exception(f"Error creating xG distribution plot: {str(e)}")

# Example usage:
# plot_shot_xg_by_type('Argentina')

def plot_player_passing_under_pressure(pass_df: pd.DataFrame, 
                                       team: str,) -> tuple:
    """
    Analyzes and plots player passing performance, comparing overall completion 
    rate with completion rate under pressure.

    This function generates a scatter plot where each point represents a player
    from the selected team, annotated with their name.

    Parameters
    ----------
    pass_df : pd.DataFrame
        DataFrame containing pass data. Must include columns: 
        ['team', 'player', 'pass_outcome', 'under_pressure'].
    team : str
        The name of the team to analyze.

    Returns
    -------
    tuple
        A tuple containing the matplotlib figure and axes objects (fig, ax).
        
    Raises
    ------
    ValueError
        If the specified team is not found in the pass data.
    """
    # Validate that the team exists in the DataFrame
    if team not in pass_df['team'].unique():
        raise ValueError(f"Team '{team}' not found in the dataset.")

    # Filter passes for the selected team
    team_passes = pass_df[pass_df['team'] == team].copy()

    # --- Data Processing using groupby for efficiency ---
    # Group by player to calculate passing metrics
    player_groups = team_passes.groupby('player')

    # Define a function to apply to each player group
    def get_pressure_metrics(group):
        total = len(group)
        completed = (group['pass_outcome'] == 'Completed').sum()
        
        pressure_total = (group['under_pressure'] == True).sum()
        pressure_completed = ((group['under_pressure'] == True) & (group['pass_outcome'] == 'Completed')).sum()
        
        # Calculate completion rates, handling division by zero
        completion_rate = (completed / total) * 100 if total > 0 else 0
        pressure_completion_rate = (pressure_completed / pressure_total) * 100 if pressure_total > 0 else 0
        
        return pd.Series({
            'pass_completion_rate': completion_rate,
            'pass_completion_rate_under_pressure': pressure_completion_rate
        })

    # Apply the function to get the metrics for each player
    player_pass_metrics = player_groups.apply(get_pressure_metrics).reset_index()

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=player_pass_metrics, 
        x='pass_completion_rate', 
        y='pass_completion_rate_under_pressure', 
        ax=ax,
        s=80, # Increase marker size
        alpha=0.7
    )

    # --- Customization ---
    ax.set_xlabel('Overall Pass Completion %', fontproperties=font_play_bold)
    ax.set_ylabel('Pass Completion % Under Pressure', fontproperties=font_play_bold)
    ax.set_title(f'Passing Accuracy vs. Accuracy Under Pressure ({team})', fontproperties=font_play_bold, fontsize=16)
    
    # Annotate each point with the player's name
    for i, row in player_pass_metrics.iterrows():
        ax.text(
            row['pass_completion_rate'] + 0.2, 
            row['pass_completion_rate_under_pressure'] + 0.2, 
            row['player'], 
            fontproperties=font_play, 
            fontsize=9
        )
        
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    return fig, ax

# Example usage:
#fig, ax = plot_player_passing_under_pressure(pass_df, 'Argentina')

def plot_team_pressure_passing_comparison(pass_df: pd.DataFrame,) -> tuple:
    """
    Analyzes and plots a comparison of team passing performance, showing 
    overall completion rate vs. completion rate under pressure.

    This function generates a scatter plot where each point represents a team,
    annotated with its name.

    Parameters
    ----------
    pass_df : pd.DataFrame
        DataFrame containing pass data. Must include columns: 
        ['team', 'pass_outcome', 'under_pressure'].
    font_play : FontProperties
        Font for regular text elements in the plot.
    font_play_bold : FontProperties
        Font for bold text elements in the plot.

    Returns
    -------
    tuple
        A tuple containing the matplotlib figure and axes objects (fig, ax).
    """
    # --- Data Processing using groupby for efficiency ---
    # Group by team to calculate passing metrics
    team_groups = pass_df.groupby('team')

    # Define a function to apply to each team group
    def get_pressure_metrics(group):
        total = len(group)
        completed = (group['pass_outcome'] == 'Completed').sum()
        
        pressure_total = (group['under_pressure'] == True).sum()
        pressure_completed = ((group['under_pressure'] == True) & (group['pass_outcome'] == 'Completed')).sum()
        
        # Calculate completion rates, handling division by zero
        completion_rate = (completed / total) * 100 if total > 0 else 0
        pressure_completion_rate = (pressure_completed / pressure_total) * 100 if pressure_total > 0 else 0
        
        return pd.Series({
            'Overall passing %': completion_rate,
            'Overall pass % when under pressure': pressure_completion_rate
        })

    # Apply the function to get the metrics for each team
    team_pass_metrics = team_groups.apply(get_pressure_metrics).reset_index()

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=team_pass_metrics, 
        x='Overall passing %', 
        y='Overall pass % when under pressure', 
        s=100, # Set marker size
        ax=ax,
        alpha=0.8,
        legend=False # Hide the default legend
    )

    # --- Customization ---
    ax.set_xlabel('Overall Pass Completion %', fontproperties=font_play_bold)
    ax.set_ylabel('Pass Completion % Under Pressure', fontproperties=font_play_bold)
    ax.set_title('Team Passing Accuracy vs. Accuracy Under Pressure', fontproperties=font_play_bold, fontsize=16)
    
    # Annotate each point with the team's name
    for i, row in team_pass_metrics.iterrows():
        ax.text(
            row['Overall passing %'] + 0.1, 
            row['Overall pass % when under pressure'] + 0.1, 
            row['team'], 
            fontproperties=font_play, 
            fontsize=10
        )
        
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    return fig, ax

# Example usage:
#fig, ax = plot_team_pressure_passing_comparison(pass_df)

def plot_pressure_heatmap(pressure_df: pd.DataFrame, 
                          team: str,) -> tuple:
    """
    Generates and plots a heatmap of pressure events for a specified team.

    This function visualizes the locations on the pitch where a team most
    frequently applies pressure. It overlays a scatter plot of individual
    pressure events on top of the heatmap.

    Parameters
    ----------
    pressure_df : pd.DataFrame
        DataFrame containing pressure event data. Must include columns:
        ['team', 'x', 'y', 'duration'].
    team : str
        The name of the team to analyze.
    font_play : FontProperties
        Font for regular text elements in the plot.
    font_play_bold : FontProperties
        Font for bold text elements in the plot.

    Returns
    -------
    tuple
        A tuple containing the matplotlib figure and axes objects (fig, ax).
        
    Raises
    ------
    ValueError
        If the specified team is not found in the pressure data.
    """
    # Validate that the team exists in the DataFrame
    if team not in pressure_df['team'].unique():
        raise ValueError(f"Team '{team}' not found in the dataset.")

    # Filter pressure events for the selected team
    team_pressure = pressure_df[pressure_df['team'] == team].copy()
    
    # Calculate summary statistics
    pressure_time = team_pressure['duration'].sum()
    total_pressure_events = len(team_pressure)
    pressure_time_per_match_min = (pressure_time / len(team_pressure['match_id'].unique())) / 60
    pressure_events_per_match = total_pressure_events / len(team_pressure['match_id'].unique())

    # --- Plotting ---
    # Create the pitch
    pitch = Pitch(
        pitch_type='statsbomb',
        line_color='#c7d5cc',
        pitch_color='#22312b',
        line_zorder=2,
    )
    fig, ax = pitch.draw(figsize=(10, 7))
    fig.set_facecolor('#22312b')

    # Calculate binned statistics for the heatmap
    bin_statistics = pitch.bin_statistic(
        team_pressure.x,
        team_pressure.y,
        statistic='count',
        bins=(6, 4),  # Divide the pitch into a 6x4 grid
        normalize=False # Use raw counts
    )

    # Plot the heatmap
    pitch.heatmap(
        bin_statistics,
        ax=ax,
        cmap='Reds',
        edgecolors='#22312b',
        linewidth=0.1,
        alpha=0.7
    )

    # Overlay individual pressure events as a scatter plot
    pitch.scatter(
        team_pressure.x,
        team_pressure.y,
        ax=ax,
        s=20,
        edgecolors='#22312b',
        c='white',
        alpha=0.7
    )

    # --- Customization ---
    # Add title and subtitle
    fig.text(
        0.5, 0.94, f'{team} Pressure Events\nCopa America 2024',
        fontsize=18, fontproperties=font_play_bold, color='white', ha='center'
    )
    fig.text(
        0.5, 0.04, f'Pressure Events per Game: {pressure_events_per_match:.2f} | Pressure Time per Game: {pressure_time_per_match_min:.2f} minutes',
        fontsize=12, fontproperties=font_play, color='white', ha='center'
    )
    
    plt.tight_layout()
    plt.show()

    return fig, ax

# Example usage:
#fig, ax = plot_pressure_heatmap(pressure_df, 'Argentina', font_play, font_play_bold)

def plot_pressure_events(pressure_df: pd.DataFrame):
    # setup a mplsoccer pitch
    pitch = Pitch(line_zorder=2, line_color='black', pad_top=20)

    # mplsoccer calculates the binned statistics usually from raw locations, such as touches events
    # for this example we will create a binned statistic dividing
    # the pitch into thirds for one point (0, 0)
    # we will fill this in a loop later with each team's statistics from the dataframe
    bin_statistic = pitch.bin_statistic([0], [0], statistic='count', bins=(6, 1), 
        normalize=True)

    GRID_HEIGHT = 0.8
    CBAR_WIDTH = 0.03
    fig, axs = pitch.grid(nrows=4, ncols=5, figheight=20,
                          # leaves some space on the right hand side for the colorbar
                          grid_width=0.88, left=0.025,
                          endnote_height=0.03, endnote_space=0,
                          # Turn off the endnote/title axis. I usually do this after
                          # I am happy with the chart layout and text placement
                          axis=False,
                          title_space=0.02, title_height=0.06, grid_height=GRID_HEIGHT)
    fig.set_facecolor('white')

    teams = pressure_df['team'].unique()
    for i, ax in enumerate(axs['pitch'].flat[:len(teams)]):
        # the top of the pitch is zero
        # plot the title half way between zero and -20 (the top padding)
        ax.text(60, -10, teams[i],
                ha='center', va='center', fontsize=50,
                fontproperties=font_play)

        # fill in the bin statistics from df and plot the heatmap

        bin_statistic['statistic'] = pitch.bin_statistic(
            pressure_df[pressure_df['team'] == teams[i]]['x'],
            pressure_df[pressure_df['team'] == teams[i]]['y'],
            statistic='count',
            bins=(6, 1),
            normalize=True
        )['statistic']
        
        heatmap = pitch.heatmap(bin_statistic, ax=ax, cmap='Reds', alpha=0.8)
        # annotate = pitch.label_heatmap(bin_statistic, color='white', fontproperties=font_play,
        #                                fontsize=50, ax=ax,
        #                                str_format='{:.0f}%', ha='center', va='center')

    # if its the Copa America remove the four spare pitches
    if len(teams) == 16:
        for ax in axs['pitch'][-1, 1:]:
            ax.remove()

    # add cbar axes
    cbar_bottom = axs['pitch'][-1, 0].get_position().y0
    cbar_left = axs['pitch'][0, -1].get_position().x1 + 0.01
    ax_cbar = fig.add_axes((cbar_left, cbar_bottom, CBAR_WIDTH,
                            # take a little bit off the height because of padding
                            GRID_HEIGHT - 0.036))
    cbar = plt.colorbar(heatmap, cax=ax_cbar)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(font_play)
        label.set_fontsize(40)

    # # title and endnote
    # add_image(sp_logo, fig,
    #           left=axs['endnote'].get_position().x0,
    #           bottom=axs['endnote'].get_position().y0,
    #           height=axs['endnote'].get_position().height)
    title = axs['title'].text(0.5, 0.5, 'Pressure events by Team, Copa America 2024',
                              ha='center', va='center', fontsize=60, fontproperties=font_play_bold)

    return fig

# Example use
# fig = plot_pressure_events(pressure_df)

# PLot pass into the final 3rd part of the pitch against by pass_height
def plot_progression_against_team(team):
    """
    Plot progression against a team by pass height.

    Parameters:
    team (str): The team to plot progression against.
    pass_df (pd.DataFrame): The passes dataframe.
    events_df (pd.DataFrame): The events dataframe.

    Returns:
    fig (matplotlib.figure.Figure): The figure object.
    """
    # Get all match IDs where this team played
    team_matches = pass_df[pass_df['team'] == team]['match_id'].unique()

    # Get passes against this team (only in matches they played)
    passes = pass_df[
        (pass_df['match_id'].isin(team_matches)) & 
        (pass_df['team'] != team) &     
        (pass_df['pass_outcome'] == 'Completed')  # Successful passes only    
    ]

    carry_df = events_df[
        (events_df['match_id'].isin(team_matches)) & 
        (events_df['team'] != team) &     
        (events_df['type'] == 'Carry')  # Carries only    
    ]

    # Filter passes landing into the last 3rd of the pitch by pass height
    low_passes = passes[(passes['pass_height'] == 'Low Pass') & (passes['x'] < 80) & (passes['pass_end_x'] >= 80)]
    high_passes = passes[(passes['pass_height'] == 'High Pass') & (passes['x'] < 80) & (passes['pass_end_x'] >= 80)]
    ground_passes = passes[(passes['pass_height'] == 'Ground Pass') & (passes['x'] < 80) & (passes['pass_end_x'] >= 80)]
    carries = carry_df[(carry_df['x'] < 80) & (carry_df['carry_end_x'] >= 80)]

    # Create pitch
    pitch = VerticalPitch(
        pitch_type='statsbomb',
        #pitch_color='grass', 
        line_color='black', 
        #stripe=True
    )

    # Create figure with 4 subplots
    fig, axs = pitch.grid(ncols=4, endnote_height=0, axis=False)
    # fig, axs = pitch.grid(figheight=8, title_height=0.08, endnote_space=0, title_space=0,
    #                       # Turn off the endnote/title axis. I usually do this after
    #                       # I am happy with the chart layout and text placement
    #                       axis=False,
    #                       grid_height=0.82, endnote_height=0.03)


    # Plot arrows for each type of pass against the team
    # Ground Passes
    pitch.arrows(
        ground_passes.x,
        ground_passes.y,
        ground_passes.pass_end_x,
        ground_passes.pass_end_y,
        width=1, headwidth=8, headlength=8,
        color='#d73728', ax=axs['pitch'][0],
        label=f'{team} Ground Passes'
    )

    # Low Passes
    pitch.arrows(
        low_passes.x,
        low_passes.y,
        low_passes.pass_end_x,
        low_passes.pass_end_y,
        width=1, headwidth=8, headlength=8,
        color='blue', ax=axs['pitch'][1],
        label=f'{team} Low Passes'
    )

    # High Passes
    pitch.arrows(
        high_passes.x,
        high_passes.y,
        high_passes.pass_end_x,
        high_passes.pass_end_y,
        width=1, headwidth=10, headlength=8,
        color='green', ax=axs['pitch'][2],
        label=f'{team} High Passes'
    )

    # Carries
    pitch.arrows(
        carries.x,
        carries.y,
        carries.carry_end_x,
        carries.carry_end_y,
        width=1, headwidth=8, headlength=8,
        color='orange', ax=axs['pitch'][3],
        label=f'{team} Carries'
    )

    # Add titles and text
    axs['title'].text(
        0.5, 0.6, f'Progression Against {team} - Copa America 2024', fontproperties=font_play_bold, 
        color='#000009', va='center', ha='center', fontsize=36
    )

    axs['title'].text(
        0.12, 0.1, f'Ground Passes: {len(ground_passes)}',
        color='#d73728', va='center', ha='center', fontsize=18, fontproperties=font_play, 
    )

    axs['title'].text(
        0.37, 0.1, f'Low Passes: {len(low_passes)}',
        color='blue', va='center', ha='center', fontsize=18, fontproperties=font_play,
    )

    axs['title'].text(
        0.63, 0.1, f'High Passes: {len(high_passes)}',
        color='green', va='center', ha='center', fontsize=18, fontproperties=font_play,
    )

    axs['title'].text(
        0.89, 0.1, f'Carries: {len(carries)}',
        color='orange', va='center', ha='center', fontsize=18, fontproperties=font_play,
    )
    return fig, axs

# Example usage:
# fig, axs = plot_progression_against_team('Argentina')

def plot_shots_against(team):
    """
    Plot shots against a team.

    Parameters:
    team (str): The team to plot shots against.
    shot_df (pd.DataFrame): The shots dataframe.

    Returns:
    fig (matplotlib.figure.Figure): The figure object.
    """

    # Get all match IDs where this team played
    team_matches = shot_df[shot_df['team'] == team]['match_id'].unique()

    # Get shots against this team (only in matches they played)
    shots_against = shot_df[
        (shot_df['match_id'].isin(team_matches)) &  # Only in matches they played
        (shot_df['team'] != team) &  # By opponent team
        (shot_df['shot_type'] != 'Penalty')  # Excluding penalties
    ]

    shots_goals_conceded = shots_against[shots_against['shot_outcome'] == 'Goal']

    # Calculate metrics for shots against
    total_shots_against = len(shots_against)
    goals_conceded = len(shots_against[shots_against['shot_outcome'] == 'Goal'])
    goals_conceded_percentage = (goals_conceded / total_shots_against * 100) if total_shots_against > 0 else 0
    total_xg_against = shots_against['shot_statsbomb_xg'].sum()
    avg_xg_per_shot_against = round(shots_against['shot_statsbomb_xg'].mean(), 3)
    avg_xg_goals_conceded = round(shots_against[shots_against['shot_outcome'] == 'Goal']['shot_statsbomb_xg'].mean(), 3)

    pitch = VerticalPitch(pad_bottom=0.5,  # pitch extends slightly below halfway line
                          half=True,  # half of a pitch
                          goal_type='box',
                          goal_alpha=0.8)  # control the goal transparency

    fig, axs = pitch.grid(figheight=8, title_height=0.08, endnote_space=0, title_space=0,
                          # Turn off the endnote/title axis. I usually do this after
                          # I am happy with the chart layout and text placement
                          axis=False,
                          grid_height=0.82, endnote_height=0.03)

    sc = pitch.scatter(shots_against.x, shots_against.y,
                       # size varies between 100 and 1000 (points squared)
                       s=(shots_against.shot_statsbomb_xg * 900) + 100,
                       c='#b94b75',  # color for scatter in hex format
                       edgecolors='#383838',  # give the markers a charcoal border
                       # for other markers types see: https://matplotlib.org/api/markers_api.html
                       marker='h',
                       ax=axs['pitch'])

    goles = pitch.scatter(shots_goals_conceded.x, shots_goals_conceded.y,
                          # size varies between 100 and 1000 (points squared)
                          s=(shots_goals_conceded.shot_statsbomb_xg * 900) + 100,
                          c='white',  # color for scatter in hex format
                          edgecolors='#383838',  # give the markers a charcoal border
                          # for other markers types see: https://matplotlib.org/api/markers_api.html
                          marker='football',
                          ax=axs['pitch'])

    txt = axs['title'].text(0.5, 0.7, s=f'Shots Against {team} - Copa America 2024',
                            size=25, fontproperties=font_play_bold, color='#383838',
                            va='center', ha='center')

    txt2 = axs['title'].text(0.5, 0.10, f"{total_shots_against} shots / Avg xG Conceded: {avg_xg_per_shot_against} / {goals_conceded} goals conceded", color='#383838',
                             va='center', ha='center', fontproperties=font_play, fontsize=16)

    return fig

# Example use
# fig = plot_shots_against(team)

def plot_team_shot_map(team):
    """
    Plot a team's shot map.

    Parameters:
    team (str): The team to plot.
    shot_df (pd.DataFrame): The shots dataframe.
    pass_df (pd.DataFrame): The passes dataframe.

    Returns:
    fig (matplotlib.figure.Figure): The figure object.
    """

    # Get team's data
    team_shots = shot_df[shot_df['team'] == team]
    team_passes = pass_df[pass_df['team'] == team]
    non_penalty_shots = team_shots[team_shots['shot_type'] != 'Penalty']
    team_goals = team_shots[(team_shots['shot_outcome'] == 'Goal') & (team_shots['shot_type'] != 'Penalty')]
    team_goals_shootout = team_shots[(team_shots['shot_outcome'] == 'Goal') & (team_shots['period'] == 5)]

    # Create pitch
    pitch = Pitch(pitch_type='statsbomb')
    fig, ax = pitch.draw(figsize=(12, 10))

    # Plot shots
    sc = pitch.scatter(team_shots.x, team_shots.y,
                       s=(team_shots.shot_statsbomb_xg * 900) + 100,
                       c='#b94b75',
                       edgecolors='#383838',
                       marker='h',
                       ax=ax)

    # Plot goals
    goles = pitch.scatter(team_goals.x, team_goals.y,
                          s=(team_goals.shot_statsbomb_xg * 900) + 100,
                          c='white',
                          edgecolors='#383838',
                          marker='football',
                          ax=ax)

    # Add title
    txt = ax.text(x=60, y=5, s=f'{team} Shot Map',
                  size=30,
                  fontproperties=font_play_bold,
                  color='#383838',
                  va='center', ha='center')

    # Add subtitle
    txt = ax.text(x=60, y=10,
                  s=f'Avg xG: {non_penalty_shots["shot_statsbomb_xg"].mean():.3f} | Shots: {len(non_penalty_shots)} | Goals: {len(team_goals) + len(team_goals_shootout)} ({len(team_goals_shootout)} from Penalty)',
                  size=16,
                  fontproperties=font_play,
                  color='darkblue',
                  va='center', ha='center')

    # Add description for size of markers
    txt = ax.text(x=60, y=82,
                  s='Marker size is proportional to xG value of each shot',
                  size=12,
                  fontproperties=font_play_bold,
                  color='#383838',
                  va='center', ha='center')

    return fig

# Example use
#fig = plot_team_shot_map(team)

def plot_match_passes(match_id, events_df, title="Copa América 2024"):
    """
    Plot pass maps for both teams in a match, split by halves.
    
    Parameters:
    match_id: str/int - The ID of the match to analyze
    events_df: DataFrame - The events dataframe containing all match events
    title: str - The title to display on the plot (default: "Copa América 2024")
    """
    # First, verify the match exists in the dataset
    match_events = events_df[events_df.match_id == match_id]
    if len(match_events) == 0:
        print(f"No events found for match_id: {match_id}")
        print("Available match_ids:", events_df.match_id.unique())
        return None
        
    # Get teams playing in this match
    teams = match_events.team.unique()
    if len(teams) < 2:
        print(f"Found only {len(teams)} team(s) for match_id {match_id}: {teams}")
        return None
    
    team1, team2 = teams[0], teams[1]
    
    # Filter passes for this match
    match_passes = events_df[
        (events_df.match_id == match_id) & 
        (events_df.type == 'Pass') & 
        (events_df.pass_outcome.isnull())  # Successful passes only
    ].copy()
    
    if len(match_passes) == 0:
        print(f"No successful passes found for match {match_id}")
        print("Event types in this match:", match_events.type.unique())
        return None
    
    # # Print debug info
    # print(f"Match: {team1} vs {team2}")
    # print(f"Total events in match: {len(match_events)}")
    # print(f"Total successful passes: {len(match_passes)}")
    # print(f"Passes by team: {match_passes.team.value_counts()}")
    # print(f"Passes by period: {match_passes.period.value_counts()}")
    
    # Create pitch
    pitch = VerticalPitch(
        pitch_type='statsbomb',
        line_color='#c7d5cc',
        pitch_color='#22312b',
        #pitch_length=105,
        #pitch_width=68,
    )
    
    # Create figure with 4 subplots
    fig, axs = pitch.grid(ncols=4, endnote_height=0, axis=False)
    
    # Create filters for each subplot
    team1_first_half = (match_passes.team == team1) & (match_passes.period == 1)
    team2_first_half = (match_passes.team == team2) & (match_passes.period == 1)
    team1_second_half = (match_passes.team == team1) & (match_passes.period == 2)
    team2_second_half = (match_passes.team == team2) & (match_passes.period == 2)
    
    # Plot arrows for each team and half
    # First half - Team 1
    pitch.arrows(
        match_passes[team1_first_half].x,
        match_passes[team1_first_half].y,
        match_passes[team1_first_half].pass_end_x,
        match_passes[team1_first_half].pass_end_y,
        width=1, headwidth=8, headlength=8,
        color='#ad993c', ax=axs['pitch'][0],
        label=f'{team1} First Half'
    )
    
    # First half - Team 2
    pitch.arrows(
        match_passes[team2_first_half].x,
        match_passes[team2_first_half].y,
        match_passes[team2_first_half].pass_end_x,
        match_passes[team2_first_half].pass_end_y,
        width=1, headwidth=8, headlength=8,
        color='#d73728', ax=axs['pitch'][1],
        label=f'{team2} First Half'
    )
    
    # Second half - Team 1
    pitch.arrows(
        match_passes[team1_second_half].x,
        match_passes[team1_second_half].y,
        match_passes[team1_second_half].pass_end_x,
        match_passes[team1_second_half].pass_end_y,
        width=1, headwidth=10, headlength=8,
        color='#ad993c', ax=axs['pitch'][2],
        label=f'{team1} Second Half'
    )
    
    # Second half - Team 2
    pitch.arrows(
        match_passes[team2_second_half].x,
        match_passes[team2_second_half].y,
        match_passes[team2_second_half].pass_end_x,
        match_passes[team2_second_half].pass_end_y,
        width=1, headwidth=8, headlength=8,
        color='#d73728', ax=axs['pitch'][3],
        label=f'{team2} Second Half'
    )
    
    # Add titles and text
    axs['title'].text(
        0.5, 0.8, f'{team1} vs {team2}\n{title}', fontproperties=font_play_bold, 
        color='#000009', va='center', ha='center', fontsize=20
    )
    
    axs['title'].text(
        0.5, 0.4, "Completed passes by team in first and second half",
        color='#000009', va='center', ha='center', fontsize=20, fontproperties=font_play, 
    )
    
    axs['title'].text(
        0.25, 0.08, "First Half",
        color='#000009', va='center', ha='center', fontsize=18, fontproperties=font_play
    )
    
    axs['title'].text(
        0.75, 0.08, "Second Half",
        color='#000009', va='center', ha='center', fontsize=18, fontproperties=font_play
    )
    
    # Add pass counts for each team/half with aligned positions
    # Format: (filter_condition, (x_position, y_position))
    count_positions = [
        (team1_first_half, (0.15, 0.08)),  # Team 1 First Half
        (team2_first_half, (0.35, 0.08)),  # Team 2 First Half
        (team1_second_half, (0.65, 0.08)), # Team 1 Second Half
        (team2_second_half, (0.85, 0.08))  # Team 2 Second Half
    ]
    
    # Add team labels
    axs['title'].text(0.05, 0.08, team1, color='#ad993c', va='center', ha='center', fontsize=14, fontproperties=font_play)
    axs['title'].text(0.45, 0.08, team2, color='#d73728', va='center', ha='center', fontsize=14, fontproperties=font_play)
    axs['title'].text(0.55, 0.08, team1, color='#ad993c', va='center', ha='center', fontsize=14, fontproperties=font_play)
    axs['title'].text(0.95, 0.08, team2, color='#d73728', va='center', ha='center', fontsize=14, fontproperties=font_play)
    
    # Add pass counts
    for filter_condition, (x, y) in count_positions:
        pass_count = len(match_passes[filter_condition])
        axs['title'].text(
            x, y,
            f'{pass_count} passes',
            color='#000009', va='center', ha='center', fontsize=12,
            fontproperties=font_play
        )
    
    #plt.show()
    return fig, axs

# Let's see what match IDs are available
#print("Available match IDs:")
#print(events_df.match_id.unique())

# Example usage:
# plot_match_passes(match_id=3943077, events_df=events_df)

def plot_team_touch_comparison(team1_name, team2_name, event_df, 
                             pitch_type='statsbomb', figheight=10,):
    """
    Compare two teams' touch heatmaps side by side.
    
    Parameters:
    - team1_name: str, name of first team
    - team2_name: str, name of second team
    - event_df: DataFrame containing event data
    - pitch_type: str, type of pitch to draw
    - figheight: int, height of figure
    """

    
    # Define actions to include as touches
    touches = ['Pass', 'Ball Receipt*', 'Carry', 'Dribble', 'Duel', 'Clearance',
               'Interception', 'Ball Recovery', 'Shield', 'Block', 'Miscontrol',
               'Foul Won', 'Shot']
    
    # Filter data for each team
    team1_df = event_df[(event_df.team == team1_name) & 
                        (event_df.type.isin(touches))]
    team2_df = event_df[(event_df.team == team2_name) & 
                        (event_df.type.isin(touches))]
    
    # Set up colors and effects
    color_1 = 'white'
    color_2 = '#c3c3c3'
    team1_color = '#e21017'  # You can customize team colors
    team2_color = '#d52b1e'  # You can customize team colors
    
    cmap_team1 = LinearSegmentedColormap.from_list('', [color_1, color_2, team1_color])
    cmap_team2 = LinearSegmentedColormap.from_list('', [color_1, color_2, team2_color])
    
    path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
                path_effects.Normal()]
    
    # Create pitch and grid
    pitch = VerticalPitch(pitch_type=pitch_type, line_zorder=2, 
                         line_color='#000000', linewidth=2, half=False)
    
    fig, axs = pitch.grid(nrows=1, ncols=2, figheight=figheight, grid_width=0.65,
                         endnote_height=0.03, endnote_space=0.05, axis=False,
                         title_space=0.02, title_height=0.06, grid_height=0.8)
    
    # Calculate bin statistics for both teams
    bin_statistics_1 = pitch.bin_statistic(team1_df.x, team1_df.y, 
                                         statistic='count', bins=(6, 4), normalize=True)
    bin_statistics_2 = pitch.bin_statistic(team2_df.x, team2_df.y, 
                                         statistic='count', bins=(6, 4), normalize=True)
    
    # Create heatmaps for both teams
    for i, (team_df, bin_stats, cmap) in enumerate(zip([team1_df, team2_df],
                                                      [bin_statistics_1, bin_statistics_2],
                                                      [cmap_team1, cmap_team2])):
        # Plot heatmap
        pitch.heatmap(bin_stats, ax=axs['pitch'][i], cmap=cmap, 
                     vmax=max(bin_statistics_1['statistic'].max(),
                             bin_statistics_2['statistic'].max()),
                     vmin=0)
        
        # Add percentage labels
        pitch.label_heatmap(bin_stats, color='white', path_effects=path_eff, 
                          fontsize=16, ax=axs['pitch'][i],
                          str_format='{:.0%}', ha='center', va='center', 
                          exclude_zeros=True)
        
        # Add team name and match stats
        team_name = team1_name if i == 0 else team2_name
        total_touches = len(team_df)
        touches_per_match = total_touches / len(team_df.match_id.unique())
        
        axs['pitch'][i].text(40, 125, 
                           f'{team_name}\nTouches: {total_touches}\n'
                           f'Touches per match: {touches_per_match:.1f}',
                           c='black', ha='center', va='center', fontsize=12,
                           fontproperties=font_play)
    
    # Add title with spacing
    fig.suptitle(f'How are {team1_name} and {team2_name} using the Pitch?\nCopa America 2024', 
                fontproperties=font_play_bold, fontsize=28, 
                color='black', y=0.98)
    
    return fig,axs

<<<<<<< HEAD
# Example usage:
plot_team_touch_comparison(
    team1_name='Venezuela',
    team2_name='Canada',
    event_df=events_df  # Your events DataFrame
)
=======
# # Example usage:
# plot_team_touch_comparison(
#     team1_name='Venezuela',
#     team2_name='Canada',
#     event_df=events_df
# )
>>>>>>> Load events via Google Drive link with low_memory=False; suppress warnings; fix xG timeline call signature
