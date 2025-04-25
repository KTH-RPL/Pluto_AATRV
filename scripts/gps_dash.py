#!/usr/bin/env python3

import warnings
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd # Using pandas for easy CSV reading
import os

# --- Configuration ---
# Path to the CSV file to read (MUST match the path used by the ROS node)
CSV_FILE = os.path.expanduser("~/gps_data.csv")
UPDATE_INTERVAL_MS = 2000  # How often to check the CSV file (milliseconds)

# --- Plotly Dash App ---
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("ROS GPS Visualization from CSV"),
    dcc.Graph(id='live-gps-map', animate=False),
    dcc.Interval(
        id='interval-component',
        interval=UPDATE_INTERVAL_MS,
        n_intervals=0
    ),
    html.Div(id='status-text', style={'padding': '10px'}) # Status display
])

@app.callback(
    [Output('live-gps-map', 'figure'),
     Output('status-text', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_graph_live(n):
    """Callback to read CSV and update the Plotly graph."""
    status_message = ""
    current_lats = []
    current_lons = []
    current_alts = []

    try:
        # Check if file exists before trying to read
        if not os.path.exists(CSV_FILE):
            status_message = f"Waiting for data file: {CSV_FILE}"
        else:
            # Read the entire CSV file using pandas
            # Using error_bad_lines=False / on_bad_lines='skip' might help if the file is being written during read
            # but ideally the overwrite is fast enough.
            # low_memory=False can sometimes help with mixed types, though unlikely here.
            df = pd.read_csv(CSV_FILE)

            if not df.empty:
                # Ensure columns exist (adjust names if your header is different)
                if 'latitude' in df.columns and 'longitude' in df.columns and 'altitude' in df.columns:
                    current_lats = df['latitude'].tolist()
                    current_lons = df['longitude'].tolist()
                    current_alts = df['altitude'].tolist()
                    status_message = f"Data loaded from CSV. Points: {len(current_lats)}"
                else:
                    status_message = "Error: CSV file missing required columns (latitude, longitude, altitude)."
            else:
                status_message = "CSV file is empty."

    except pd.errors.EmptyDataError:
        status_message = f"CSV file {CSV_FILE} is empty or being written."
    except FileNotFoundError:
         status_message = f"Error: CSV file not found at {CSV_FILE}. Is the ROS node running and writing?"
    except Exception as e:
        status_message = f"Error reading or processing CSV {CSV_FILE}: {e}"
        print(f"Error details: {e}") # Print detailed error to console

    # --- Plotting logic (same as before) ---
    if not current_lats:
        lat_center, lon_center = 34.0522, -118.2437 # Default center
        zoom_level = 10
        if not status_message: # Add message if needed
             status_message = "No GPS points received yet."
    else:
        lat_center, lon_center = current_lats[-1], current_lons[-1] # Center on last point
        zoom_level = 15

    trace = go.Scattermapbox(
        lat=current_lats,
        lon=current_lons,
        mode='markers+lines',
        marker=go.scattermapbox.Marker(
            size=9,
            color=current_alts,
            colorscale='Viridis',
            showscale=True,
            colorbar_title="Altitude (m)"
        ),
        line=dict(width=2, color='blue'),
        name='GPS Track',
        hovertext=[f"Alt: {alt:.2f}m" for alt in current_alts],
        hoverinfo='lat+lon+text'
    )

    layout = go.Layout(
        autosize=True,
        hovermode='closest',
        mapbox=dict(
            style='open-street-map',
            accesstoken=None,
            bearing=0,
            center=dict(lat=lat_center, lon=lon_center),
            pitch=0,
            zoom=zoom_level
        ),
        margin={"r":0,"t":5,"l":0,"b":0}
    )

    fig = go.Figure(data=[trace], layout=layout)
    fig.update_layout(uirevision='constant') # Keep map view stable

    return fig, status_message

# --- Main Execution ---
if __name__ == '__main__':
    print("\nDash server starting. Reading data from CSV.")
    print(f"Monitoring CSV file: {CSV_FILE}")
    print(f"Update interval: {UPDATE_INTERVAL_MS / 1000.0} seconds")
    print("Open your browser to http://127.0.0.1:8050/")
    warnings.filterwarnings(action="ignore")
    # Run the Dash app
    app.run(debug=False, host='0.0.0.0', port=8050)
    print("Dash server stopped.")
    