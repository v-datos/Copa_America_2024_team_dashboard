# Copa America 2024 Team Analysis Dashboard

A comprehensive interactive dashboard for analyzing football team performance using Copa America 2024 event data.

## Features

### 📊 Main KPIs Tab
- **Key Performance Indicators**: Matches played, total passes, pass completion rate, total shots, shots on target percentage, goals, goals conceded
- **Advanced Metrics**: Average xG, average xG/shot, average xG per shot against
- **Interactive Timeline**: xG and xG conceded progression throughout the tournament

### 🎯 Playing Style Tab
- **Radar Chart**: Comprehensive performance comparison across 9 key metrics
- **Style Metrics**: Through ball %, cross %, under pressure %, counter shots per game, goalkeeper pass length
- **Distribution Analysis**: xG distribution and goalkeeper pass length distribution visualizations

### 📍 Passes & Shots Tab
- **Shot Maps**: Interactive shot location maps with xG-sized markers, highlighting goals vs other shots
- **Attacking Passes**: Visualization of crosses, cutbacks, switches, and through balls with their end locations
- **Spatial Analysis**: Understanding team's attacking patterns and shot quality

### 🛡️ Defensive Metrics Tab
- **Defensive KPIs**: Shots against per game, goals conceded percentage, pressure actions, interceptions
- **Defensive Actions**: Blocks, counter presses, and other defensive metrics
- **Performance Analysis**: Comprehensive defensive performance evaluation

### ⚔️ Team Comparison Tab
- **Head-to-Head Metrics**: Direct comparison of key performance indicators between two teams
- **Radar Comparison**: Side-by-side radar charts for visual performance comparison
- **Pressure Analysis**: Comparative analysis of playing under pressure

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure all CSV data files are in the same directory:
- copa_america_events.csv
- copa_america_games_2024.csv
- copa_america_blocks_2024.csv
- copa_america_dribbled_pasts_2024.csv
- copa_america_duels_2024.csv
- copa_america_foul_committeds_2024.csv
- copa_america_interceptions_2024.csv
- copa_america_50_50s_2024.csv
- copa_america_pressure_2024.csv
- copa_america_counterpress_2024.csv
- copa_america_pass_2024.csv
- copa_america_shots_2024.csv

## Usage

Run the dashboard:
```bash
streamlit run copa_america_dashboard.py
```

The dashboard will open in your default web browser. Use the sidebar to select different teams and navigate through the tabs to explore various aspects of team performance.

## Data Requirements

The dashboard expects CSV files with specific column structures:
- **Shots data**: team, shot_type, shot_outcome, shot_statsbomb_xg, x, y, match_id
- **Passes data**: team, pass_outcome, pass_cross, pass_cut_back, pass_switch, pass_through_ball, position, pass_length, x, y, pass_end_x, pass_end_y
- **Other event data**: team, match_id, and event-specific columns

## Technical Details

- Built with **Streamlit** for the web interface
- Uses **matplotlib** and **mplsoccer** for football pitch visualizations
- **Plotly** for interactive charts and timelines
- **Pandas** for data manipulation and analysis
- Custom fonts (Play-Regular.ttf, Play-Bold.ttf) for enhanced styling

## Key Metrics Explained

- **xG (Expected Goals)**: Probability that a shot will result in a goal
- **Pass Completion Rate**: Percentage of successful passes
- **Shots on Target %**: Percentage of shots that hit the target
- **Counter Attacking Shots**: Shots resulting from counter-attack situations
- **Under Pressure %**: Percentage of actions performed under opponent pressure

## Customization

The dashboard can be customized by:
- Modifying colors and styling in the CSS section
- Adding new metrics to the analysis functions
- Extending the radar chart with additional parameters
- Adding new visualization types for specific analysis needs
