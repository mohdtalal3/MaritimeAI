import streamlit as st
import folium
from streamlit_folium import folium_static
import pandas as pd
from datetime import datetime, timedelta
import time
import searoute as sr
import plotly.graph_objects as go
from folium.plugins import BeautifyIcon
from glassflow import PipelineDataSink
from pymongo import MongoClient
import logging
import time
# Set up logging
logging.basicConfig(level=logging.INFO)

# MongoDB configuration
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["ais_eta_data"]
collection = db["ais_data"]

# Define your pipeline credentials
pipeline_id = "401cb58b-612e-478b-90c0-5f67f5ee485c"
pipeline_access_token = "4tfyFrqkrpn97Mww5r3PhAEynemEHPMatAEbBACZaGRSDM7WKTmdeVT7gb5ATw4hfMQAfyP97P4HHXwFbKRaA5pvJxpqcmKNFjBx5JPHkzKyqKYYgW88wFhZFK9X4EhB"

# Create a data sink
sink = PipelineDataSink(
    pipeline_id=pipeline_id,
    pipeline_access_token=pipeline_access_token
)

# Define color mapping for ship types
SHIP_TYPE_COLORS = {
    'Cargo': '#FF0000',  # Red
    'Fishing': '#0000FF',  # Blue
    'Pilot': '#00FF00',  # Green
    'Tanker': '#FFA500',  # Orange
    'Passenger': '#800080',  # Purple
    'Pleasure': '#FFFF00',  # Yellow
    'Military': '#808080',  # Gray
    'Tug': '#A52A2A',  # Brown
    'Other': '#000000'  # Black (default)
}

@st.cache_data
def load_port_data():
    try:
        ports_df = pd.read_csv('/home/talal/fyp/predictive model/ports_output.csv')
        logging.info(f"Loaded {len(ports_df)} ports from CSV.")
        
        # Check for duplicate port names
        duplicates = ports_df['Port Name'].duplicated()
        if duplicates.any():
            logging.warning(f"Found {duplicates.sum()} duplicate port names. Using first occurrence of each.")
        
        # Group by 'Port Name' and take the first occurrence of each port
        ports_df = ports_df.groupby('Port Name').first().reset_index()
        
        # Create a dictionary with port names as keys and coordinates as values
        ports_dict = ports_df.set_index('Port Name')[['Latitude', 'Longitude']].to_dict('index')
        
        logging.info(f"Created port dictionary with {len(ports_dict)} unique ports.")
        return ports_dict
    except Exception as e:
        logging.error(f"Error loading port data: {str(e)}")
        st.error("An error occurred while loading port data. Please check the logs for details.")
        return {}

def calculate_eta(current_position, destination_port, speed, ports_data):
    if destination_port not in ports_data:
        return None, None
    
    destination = ports_data[destination_port]
    origin = [current_position['longitude'], current_position['latitude']]
    dest = [destination['Longitude'], destination['Latitude']]
    
    try:
        route = sr.searoute(origin, dest)
        distance_km = route.properties['length']
        distance_nautical = distance_km / 1.852  # Convert km to nautical miles
        
        if speed is not None and speed > 0:
            time_to_destination = distance_nautical / speed  # in hours
            eta = datetime.now() + timedelta(hours=time_to_destination)
        else:
            eta = None
        
        return eta, distance_nautical
    except Exception as e:
        logging.error(f"Error calculating route: {e}")
        return None, None

def get_ship_color(ship_type):
    return SHIP_TYPE_COLORS.get(ship_type, SHIP_TYPE_COLORS['Other'])

def update_map(df, ports_data, selected_port, selected_ship_type, highlighted_mmsi):
    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=4)
    
    for _, ship in df.iterrows():
        if (selected_port == 'All' or ship['destination'] == selected_port) and \
           (selected_ship_type == 'All' or ship['ship_type'] == selected_ship_type):
            
            eta, distance_left = calculate_eta(ship, ship['destination'], ship['sog'], ports_data)
            eta_str = eta.strftime('%Y-%m-%d %H:%M:%S') if eta else "N/A"
            distance_str = f"{distance_left:.2f} nautical miles" if distance_left is not None else "N/A"
            
            ship_color = get_ship_color(ship['ship_type'])
            
            tooltip_content = f"""
            <table style="width:100%; border-collapse: collapse; font-family: Arial, sans-serif;">
                <tr style="background-color: #f2f2f2;">
                    <th style="padding: 10px; text-align: left; border-bottom: 1px solid #ddd;">Field</th>
                    <th style="padding: 10px; text-align: left; border-bottom: 1px solid #ddd;">Value</th>
                </tr>
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Name</strong></td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{ship['name']}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>MMSI</strong></td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{ship['mmsi']}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Time</strong></td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{ship['time']}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Latitude</strong></td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{ship['latitude']:.4f}°N</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Longitude</strong></td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{ship['longitude']:.4f}°W</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Navigational Status</strong></td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{ship['navigational_status']}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Heading</strong></td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{ship['heading']}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>SOG</strong></td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{ship['sog']} knots</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Destination</strong></td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{ship['destination']}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Ship Type</strong></td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{ship['ship_type']}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Distance to Destination</strong></td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">{distance_str}</td>
                </tr>
                <tr>
                    <td style="padding: 8px;"><strong>Estimated Time of Arrival</strong></td>
                    <td style="padding: 8px;">{eta_str}</td>
                </tr>
            </table>
            """
            
            # Determine icon size based on whether the ship matches the searched MMSI
            if str(ship['mmsi']) == str(highlighted_mmsi):
                icon = BeautifyIcon(
                    icon='ship',
                    icon_size=(40, 40),  # Increased size for highlighted ship
                    inner_icon_style=f'font-size:20px; color:{ship_color};',
                    background_color='transparent',
                    border_color='transparent',
                    text_color=ship_color,
                    icon_shape='marker'
                )
            else:
                icon = BeautifyIcon(
                    icon='ship',
                    icon_size=(20, 20),  # Normal size for other ships
                    inner_icon_style=f'font-size:14px; color:{ship_color};',
                    background_color='transparent',
                    border_color='transparent',
                    text_color=ship_color,
                    icon_shape='marker'
                )

            folium.Marker(
                [ship['latitude'], ship['longitude']],
                popup=folium.Popup(tooltip_content, max_width=400),
                tooltip="Click for more info",
                icon=icon
            ).add_to(m)
    
    return m

def create_donut_chart(df, selected_port):
    if selected_port != 'All':
        df = df[df['destination'] == selected_port]
    
    ship_type_counts = df['ship_type'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=ship_type_counts.index,
        values=ship_type_counts.values,
        hole=.6,
        textinfo='label+percent',
        insidetextorientation='radial',
        hoverinfo='label+value',
        textfont_size=12,
        pull=[0.1] * len(ship_type_counts),
        marker=dict(line=dict(color='#000000', width=2))
    )])
    
    fig.update_layout(
            annotations=[dict(text=f'Total Ships: {len(df)}', x=0.5, y=0.5, font_size=20, showarrow=False)],
            hoverlabel=dict(bgcolor="black", font_size=16),
            height=800,
            width=1000,  
            legend=dict(
                font=dict(size=20),
  
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
    
    return fig

@st.cache_data
def load_initial_data():
    data = list(collection.find({}, {'_id': 0})) 
    return pd.DataFrame(data)

def update_ship_data(new_data):
    if 'ship_data' not in st.session_state:
        st.session_state.ship_data = load_initial_data()
    
    # Update or add new data
    index = st.session_state.ship_data.index[st.session_state.ship_data['mmsi'] == new_data['mmsi']].tolist()
    if index:
        st.session_state.ship_data.loc[index[0]] = new_data
    else:
        st.session_state.ship_data = pd.concat([st.session_state.ship_data, pd.DataFrame([new_data])], ignore_index=True)
        st.rerun()  # Trigger a rerun when new data is added

def get_filter_options(df):
    return {
        'ports': ['All'] + sorted(df['destination'].unique().tolist()),
        'ship_types': ['All'] + sorted(df['ship_type'].unique().tolist())
    }

def eta_tab():
    st.header("ETA Calculator")

    # Load initial data from MongoDB
    if 'ship_data' not in st.session_state:
        st.session_state.ship_data = load_initial_data()

    # Load port data
    ports_data = load_port_data()

    df = st.session_state.ship_data

    # Get initial filter options
    if 'filter_options' not in st.session_state:
        st.session_state.filter_options = get_filter_options(df)

    # Filters and search
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_port = st.selectbox("Select Port", st.session_state.filter_options['ports'])
    with col2:
        selected_ship_type = st.selectbox("Select Ship Type", st.session_state.filter_options['ship_types'])
    with col3:
        searched_mmsi = st.text_input("Search Ship by MMSI")

    # Create two columns for map and chart
    col_map, col_chart = st.columns([0.8,0.2])

    # Create placeholders for map and chart
    with col_map:
        st.subheader("Ship Tracking Map")
        map_placeholder = st.empty()

    with col_chart:

        st.subheader("Ship Distribution")
        chart_placeholder = st.empty()

    # Initialize update counter
    update_counter = 0

    # Initial map and chart update
    m = update_map(df, ports_data, selected_port, selected_ship_type, searched_mmsi)
    with map_placeholder:
        folium_static(m, width=1300, height=800)

    donut_chart = create_donut_chart(df, selected_port)
    with chart_placeholder:
        st.plotly_chart(donut_chart, use_container_width=True, key=f"donut_chart_initial_{int(time.time())}")

    # Main loop
    while True:
        # Consume data from the sink
        response = sink.consume()
        
        if response.status_code == 200:
            data = response.json()
            logging.info("Consumed Data: %s", data)

            # Store only the latest entry in MongoDB
            latest_entry = {
                "time": data["Timestamp"],
                "mmsi": data["MMSI"],
                "latitude": data["Latitude"],
                "longitude": data["Longitude"],
                "navigational_status": data["Navigational status"],
                "heading": data["Heading"],
                "sog": data["SOG"],
                "destination": data["Destination"],
                "ship_type": data["Ship type"],
                "name": data["Name"]
            }

            # Update or insert the latest entry based on MMSI
            collection.update_one({"mmsi": latest_entry["mmsi"]}, {"$set": latest_entry}, upsert=True)
            logging.info("Stored/Updated in MongoDB: %s", latest_entry)

            # Update ship data in Streamlit session state
            update_ship_data(latest_entry)
            df = st.session_state.ship_data

            # Update filter options
            new_filter_options = get_filter_options(df)
            if new_filter_options != st.session_state.filter_options:
                st.session_state.filter_options = new_filter_options
                st.rerun()

            # Update map and chart only when new data is received
            m = update_map(df, ports_data, selected_port, selected_ship_type, searched_mmsi)
            with map_placeholder:
                folium_static(m, width=1300, height=800)

            donut_chart = create_donut_chart(df, selected_port)
            with chart_placeholder:
                st.plotly_chart(donut_chart, use_container_width=True, key=f"donut_chart_{update_counter}_{int(time.time())}")

            # Increment update counter
            update_counter += 1
        else:
            logging.error("Failed to consume data. Status code: %s", response.status_code)

        # Wait for 5 seconds before next update
        time.sleep(5)

    # Final message
    st.success("Journey completed!")