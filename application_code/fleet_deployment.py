import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Set the page configuration first
#st.set_page_config(layout="wide")  # Set to wide layout


def fleet():
    # Caching the models to prevent reloading every time
    @st.cache_resource
    def load_models():
        class_model = joblib.load('/home/talal/fyp/predictive model/multi_label_classification_model_new.joblib')
        reg_model = joblib.load('/home/talal/fyp/predictive model/regression_model_new.joblib')
        mlb = joblib.load('/home/talal/fyp/predictive model/destination_mlb_new.joblib')
        return class_model, reg_model, mlb

    # Load models at the start
    loaded_pipeline_class, loaded_pipeline_reg, loaded_mlb = load_models()

    def create_donut_chart(predictions_df, selected_port):
        """
        Creates a donut chart showing the distribution of predicted ship types for a given port.
        """
        # Summing up the total counts for each ship type at the selected destination
        yearly_totals = predictions_df.groupby('Destination')['TotalCount'].sum().reset_index()
        
        # Create the donut chart using Plotly
        fig = go.Figure(data=[go.Pie(
            labels=yearly_totals['Destination'],
            values=yearly_totals['TotalCount'],
            hole=.4,
            textposition='inside',
            textinfo='label+percent',
            insidetextorientation='radial',
            hoverinfo='label+value',
            textfont_size=12,
            pull=[0.1] * len(yearly_totals),
            marker=dict(line=dict(color='#000000', width=2))
        )])

        fig.update_layout(
            title_text=f"Predicted Ship Types Distribution for {selected_port}",
            annotations=[dict(text=f'Total Ships: {yearly_totals["TotalCount"].sum()}', 
                            x=0.5, y=0.5, font_size=20, showarrow=False)],
            hoverlabel=dict(bgcolor="black", font_size=16),
            height=600
        )

        return fig

    def create_bar_chart(predictions_df, selected_port, selected_year):
        """
        Creates an interactive bar chart for ship type deployments at the selected destination and year.
        """
        # Create the bar chart using Plotly
        fig = px.bar(predictions_df, 
                    x='Destination', 
                    y='TotalCount', 
                    labels={'Destination': 'Destination', 'TotalCount': 'Predicted Total Count'},
                    color='TotalCount', color_continuous_scale='Viridis')

      
        fig.update_layout(
            xaxis_title="Ship Type",
            yaxis_title="Predicted Total Count",
            barmode="group",
            xaxis_tickangle=-45,  
            showlegend=False,
            bargap=0.2, 
            xaxis={'categoryorder': 'total descending'},  
        )

        return fig

    def predict_destinations_and_counts(year_week, cn, class_model, reg_model, mlb):
        """
        Predict Destinations and their TotalCounts based on YearWeek and CN.
        
        Parameters:
        - year_week (str): Format 'YYYY-WW', e.g., '2024-05'
        - cn (str): Company Name
        
        Returns:
        - DataFrame: A DataFrame with 'Destination' and 'TotalCount' columns
        """
        # Split YearWeek
        try:
            year, week = map(int, year_week.split('-'))
        except ValueError:
            raise ValueError("year_week should be in 'YYYY-WW' format, e.g., '2024-05'")
        
        # Create a DataFrame for classification input
        input_class = pd.DataFrame({
            'Year': [year],
            'Week': [week],
            'CN': [cn]
        })
        
        # Predict Destinations (binary matrix)
        predicted_class = class_model.predict(input_class)
        
        # Convert binary matrix to list of destinations
        predicted_destinations = mlb.inverse_transform(predicted_class)
        
        # If no destinations predicted
        if not predicted_destinations or not predicted_destinations[0]:
            return pd.DataFrame(columns=['Destination', 'TotalCount'])
        
        # Extract destinations (since we have only one input)
        destinations = predicted_destinations[0]
        
        # Prepare DataFrame for regression input
        input_reg = pd.DataFrame({
            'Year': [year] * len(destinations),
            'Week': [week] * len(destinations),
            'CN': [cn] * len(destinations),
            'Destination': list(destinations)
        })
        
        # Predict TotalCount for each destination
        predicted_counts = reg_model.predict(input_reg)
        
        # Combine destinations with their predicted counts
        results = pd.DataFrame({
            'Destination': destinations,
            'TotalCount': predicted_counts.astype(int)
        })
        
        return results

    # Load unique destinations from CSV
    @st.cache_data
    def load_unique_destinations():
        df = pd.read_csv('/home/talal/fyp/predictive model/ship_compnay_name_new.csv')  
        return df['CN'].unique()

    # Load coordinates from the destination data CSV
    @st.cache_data
    def load_destination_coordinates():
        df = pd.read_csv('/home/talal/fyp/predictive model/ports_output.csv')
        return df[['Port Name', 'Latitude', 'Longitude']]

    # Week mapping: Mapping of month-week to week number (excluding the year part)
    week_mapping = {
        "January Week 1": "01", "January Week 2": "02", "January Week 3": "03", "January Week 4": "04",
        "February Week 1": "05", "February Week 2": "06", "February Week 3": "07", "February Week 4": "08",
        "March Week 1": "09", "March Week 2": "10", "March Week 3": "11", "March Week 4": "12",
        "April Week 1": "13", "April Week 2": "14", "April Week 3": "15", "April Week 4": "16",
        "May Week 1": "17", "May Week 2": "18", "May Week 3": "19", "May Week 4": "20",
        "June Week 1": "21", "June Week 2": "22", "June Week 3": "23", "June Week 4": "24",
        "July Week 1": "25", "July Week 2": "26", "July Week 3": "27", "July Week 4": "28",
        "August Week 1": "29", "August Week 2": "30", "August Week 3": "31", "August Week 4": "32",
        "September Week 1": "33", "September Week 2": "34", "September Week 3": "35", "September Week 4": "36",
        "October Week 1": "37", "October Week 2": "38", "October Week 3": "39", "October Week 4": "40",
        "November Week 1": "41", "November Week 2": "42", "November Week 3": "43", "November Week 4": "44",
        "December Week 1": "45", "December Week 2": "46", "December Week 3": "47", "December Week 4": "48"
    }

    # Streamlit UI components
    st.title("Number of Assets in Fleet Deployed")

    st.write("Select a year, week, and destination to see the predicted Number of Assets in Fleet Deployed.")

    # Create a row for selecting Year, Week, and Destination
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        year = st.selectbox("Select Year", [2024], index=0,key="company_year")

    with col2:
        # Create a list of options for weeks in the format: "Month Week X"
        months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]

        # Create a list of options for weeks in the format: "Month Week X"
        week_options = []
        for month_idx, month in enumerate(months, start=1):
            for week in range(1, 5):  # Assuming there are at least 4 weeks per month, modify as needed
                week_options.append(f"{month} Week {week}")

        selected_week_display = st.selectbox("Select Week", week_options,key="company_company_box")

    with col3:
        unique_ports = load_unique_destinations()
        cn = st.selectbox("Select Company Name", unique_ports)


    week_number = week_mapping.get(selected_week_display)

    # Combine the selected year with the mapped week number to form the 'YYYY-WW' format
    yearweek = f"{year}-{week_number}"

    if st.button("Prediction"):
        predicted_results = predict_destinations_and_counts(
            yearweek,
            cn,
            loaded_pipeline_class,
            loaded_pipeline_reg,
            loaded_mlb
        )

        # Show donut chart
        st.subheader("Ship Type Deployment Distribution")
        donut_chart = create_donut_chart(predicted_results, cn)
        st.plotly_chart(donut_chart)

        # Show bar chart
        st.subheader("Ship Type Deployment Counts")
        bar_chart = create_bar_chart(predicted_results, cn, year)
        st.plotly_chart(bar_chart)

        # Load destination coordinates
        destination_coords = load_destination_coordinates()

        # Merge the predicted destinations with their coordinates
        predicted_coords = pd.merge(predicted_results, destination_coords, left_on='Destination', right_on='Port Name', how='left')

        # Create a map plot
        st.subheader("Map of Predicted Destinations")
        fig_map = px.scatter_mapbox(predicted_coords, 
                                    lat="Latitude", 
                                    lon="Longitude", 
                                    color="TotalCount", 
                                    size="TotalCount", 
                                    hover_name="Destination",
                                    hover_data=["TotalCount"],
                                    color_continuous_scale="Viridis",
                                    size_max=40, 
                                    title="Predicted Ship Deployment Locations",
                                    zoom=6)
        
        fig_map.update_layout(mapbox_style="open-street-map", height=700)
        st.plotly_chart(fig_map)
