import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import plotly.express as px




def frequency():
    model_filename = "frequency_of_deployment_regressor.joblib"
    model = joblib.load(model_filename)

    # Array of all ship types
    ship_types = np.array([
        'Sailing', 'Undefined', 'Military', 'Tug', 'Fishing', 'Pilot',
        'Other', 'Port tender', 'Cargo', 'Pleasure', 'Passenger',
        'Reserved', 'Tanker', 'SAR', 'HSC', 'Dredging',
        'Not party to conflict', 'Law enforcement', 'Towing', 'Diving',
        'Anti-pollution', 'Medical', 'Spare 1', 'WIG', 'Towing long/wide',
        'Spare 2'
    ], dtype=object)

    # Function to generate predictions with ceiling applied
    def generate_predictions(yearweek, destination, model, ship_types):
        """
        Generates predictions for all ship types based on the provided YearWeek and Destination,
        applying the ceiling function to the predicted TotalCount.
        """
        # Create dataframe with all ship types
        new_data = pd.DataFrame({
            'Yearweek': [yearweek] * len(ship_types),
            'Ship_Type': ship_types,
            'Destination': [destination] * len(ship_types)  
        })

        # Split Yearweek into Year and Week
        new_data[['Year', 'Week']] = new_data['Yearweek'].str.split('-', expand=True)
        new_data['Year'] = new_data['Year'].astype(int)
        new_data['Week'] = new_data['Week'].astype(int)


        new_data = new_data.drop('Yearweek', axis=1)

        # Make predictions
        predicted_counts = model.predict(new_data)

        # Apply ceiling to predictions
        predicted_counts_ceiled = np.ceil(predicted_counts).astype(int)

        new_data['Predicted_TotalCount'] = predicted_counts_ceiled

        result_df = new_data[['Ship_Type', 'Predicted_TotalCount', 'Destination', 'Year', 'Week']]  # Include 'Destination'

        return result_df


    # Load unique destinations from CSV
    @st.cache_data
    def load_unique_destinations():
        df = pd.read_csv('/home/talal/fyp/ETA/unique_destinations.csv')  
        return df['Destination'].unique()

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

    st.title("Frequency of Deployments")

    st.write("Select a year, week, and destination to see the predicted Frequency of Deployments.")

    # Create a row for selecting Year, Week, and Destination
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        year = st.selectbox("Select Year", [2024], index=0)

    with col2:

        months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]


        week_options = []
        for month_idx, month in enumerate(months, start=1):
            for week in range(1, 5):  
                week_options.append(f"{month} Week {week}")

        selected_week_display = st.selectbox("Select Week", week_options)

    with col3:
        unique_ports = load_unique_destinations()
        destination = st.selectbox("Select Destination", unique_ports)

  
    week_number = week_mapping.get(selected_week_display)
    import plotly.graph_objects as go

    def create_donut_chart(predictions_df, selected_port):
        """
        Creates a donut chart showing the distribution of predicted ship types for a given port.
        """
        # Summing up the total counts for each ship type at the selected destination
        yearly_totals = predictions_df.groupby('Ship_Type')['Predicted_TotalCount'].sum().reset_index()
        
        # Create the donut chart using Plotly
        fig = go.Figure(data=[go.Pie(
            labels=yearly_totals['Ship_Type'],
            values=yearly_totals['Predicted_TotalCount'],
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
            annotations=[dict(text=f'Total Ships: {yearly_totals["Predicted_TotalCount"].sum()}', 
                            x=0.5, y=0.5, font_size=20, showarrow=False)],
            hoverlabel=dict(bgcolor="black", font_size=16),
            height=600
        )

        return fig


    def create_bar_chart(predictions_df, selected_port, selected_year):
        """
        Creates an interactive bar chart for ship type deployments at the selected destination and year.
        """
        # Filter the data for the selected destination and year
        ship_type_data = predictions_df[(predictions_df['Destination'] == selected_port) &
                                        (predictions_df['Year'] == selected_year)]
        
        # Create the bar chart using Plotly
        fig = px.bar(ship_type_data, 
                    x='Ship_Type', 
                    y='Predicted_TotalCount', 
                    title=f"Ship Deployment Counts for {selected_port} ({selected_year})",
                    labels={'Ship_Type': 'Ship Type', 'Predicted_TotalCount': 'Number of Ships'},
                    color='Ship_Type', 
                    color_continuous_scale='Viridis',
                    category_orders={"Ship_Type": ship_types.tolist()})
        
        # Update layout to make the bars wider and improve visualization
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

    yearweek = f"{year}-{week_number}"

    if st.button("Predict"):
        predictions_df = generate_predictions(yearweek, destination, model, ship_types)
        # Show donut chart
        st.subheader("Ship Type Deployment Distribution")
        donut_chart = create_donut_chart(predictions_df, destination)
        st.plotly_chart(donut_chart)

        # Show bar chart
        st.subheader("Ship Type Deployment Counts")
        bar_chart = create_bar_chart(predictions_df, destination, year)
        st.plotly_chart(bar_chart)