import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Basic username/password authentication
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if st.session_state["authenticated"] == False:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == "admin" and password == "admin":
                st.session_state["authenticated"] = True
            else:
                st.error("Incorrect username or password")
        return False
    else:
        return True

if check_password():
    st.title('EV Charging Data Dashboard')

    # Upload dataset file
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        dataset = pd.read_csv(uploaded_file)

        #*** Data Cleaning ***
        # Change Null values in 'vehicle_model' to 'Other'; 7579 total
        dataset['vehicle_model'].fillna('Other', inplace=True)

        # To make sure X is updated with the new values
        X = dataset.drop('expected_departure', axis=1).values

        # Convert columns to datetime objects; 'errors='coerce' to handle values that can't convert
        dataset['start_date_time'] = pd.to_datetime(dataset['start_date_time'], errors='coerce')
        dataset['end_date_time'] = pd.to_datetime(dataset['end_date_time'], errors='coerce')
        dataset['start_charge'] = pd.to_datetime(dataset['start_charge'], errors='coerce')
        dataset['termin_charge'] = pd.to_datetime(dataset['termin_charge'], errors='coerce')

        # Drop rows where both 'start_date_time' and 'end_date_time' are missing
        dataset = dataset.dropna(subset=['start_date_time', 'end_date_time'])

        # Reset the index after dropping rows
        dataset.reset_index(drop=True, inplace=True)

        # Drop rows where both 'start_charge' and 'termin_charge' are missing
        dataset = dataset.dropna(subset=['start_charge', 'termin_charge'])

        # Reset the index after dropping rows
        dataset.reset_index(drop=True, inplace=True)

        # Resolve erroneous values in 'miles_requested'
        neg_miles = dataset[(dataset['miles_requested'] < 1)].index

        # Replace negative values with the mean of the remaining values in the column
        dataset.loc[neg_miles, 'miles_requested'] = dataset['miles_requested'].mean()

        # Sort by 'miles_requested' to verify it worked
        sorted_dataset = dataset.sort_values(by='miles_requested')

        # Resolve erroneous values in 'kwh_requested'
        neg_kwh = dataset[(dataset['kwh_requested'] < .01)].index

        # Replace negative values with the mean of the remaining values in the column
        dataset.loc[neg_kwh, 'kwh_requested'] = dataset['kwh_requested'].mean()

        # Sort by 'kwh_requested' to verify it worked
        sorted_dataset_2 = dataset.sort_values(by='kwh_requested')

        # Resolve erroneous values in 'energy_charged'
        zero_energy = dataset[dataset['energy_charged'] < 0.001].index

        # Add a small constant to all 'energy_charged' values to avoid being 0
        small_constant = 0.001
        dataset['energy_charged'] = dataset['energy_charged'].apply(lambda x: x + small_constant if x < 0.001 else x)

        # Replace negative values with the mean of the remaining values in the column
        dataset.loc[zero_energy, 'energy_charged'] = dataset['energy_charged'].mean()

        # Calculate the mean
        max_charge_power_mean = dataset['max_charge_power'].mean()
        energy_charged_mean = dataset['energy_charged'].mean()

        # Fill null values with the mean
        dataset['max_charge_power'].fillna(max_charge_power_mean, inplace=True)
        dataset['energy_charged'].fillna(energy_charged_mean, inplace=True)

        # Sort by 'energy_charged' to verify it worked
        sorted_dataset_3 = dataset.sort_values(by='max_charge_power')

        # Drop irrelevant columns
        dataset.drop(['driverId', 'station', 'controlled_duration', 'cost_for_session', 'afterPaid'], axis=1, inplace=True)

        # Convert 'expected_departure' and 'request_entry_time' columns to datetime
        dataset['expected_departure'] = pd.to_datetime(dataset['expected_departure'])
        dataset['request_entry_time'] = pd.to_datetime(dataset['request_entry_time'])

        # Create column for time difference(in seconds) between 'expected_departure' and 'request_entry_time'
        dataset['request_duration'] = (dataset['expected_departure'] - dataset['request_entry_time'])

        # Convert to timedelta
        dataset['request_duration'] = pd.to_timedelta(dataset['request_duration'], unit='s').round('1s')

        # Convert 'start_charge' and 'termin_charge' columns to datetime
        dataset['termin_charge'] = pd.to_datetime(dataset['termin_charge'])
        dataset['start_charge'] = pd.to_datetime(dataset['start_charge'])

        # Create column for time difference(in seconds) between 'start_charge' and 'termin_charge'
        dataset['charge_duration'] = (dataset['termin_charge'] - dataset['start_charge']).dt.total_seconds()

        # Convert to timedelta
        dataset['charge_duration'] = pd.to_timedelta(dataset['charge_duration'], unit='s').round('1s')

        # Convert 'expected_departure' and 'request_entry_time' columns to datetime
        dataset['expected_departure'] = pd.to_datetime(dataset['expected_departure'])
        dataset['request_entry_time'] = pd.to_datetime(dataset['request_entry_time'])

        # Create 'expected_duration', assuming 'expected_departure' and 'request_entry_time' are datetime columns
        dataset['expected_duration'] = (dataset['expected_departure'] - dataset['request_entry_time']).dt.total_seconds()

        # Convert 'charge_duration' to total seconds
        dataset['charge_duration_seconds'] = dataset['charge_duration'].dt.total_seconds()

        # Calculate the difference between 'charge_duration' and 'expected_duration'
        dataset['charge_vs_expected_diff'] = dataset['expected_duration'] - dataset['charge_duration_seconds']

        # Convert to timedelta
        dataset['charge_vs_expected_diff'] = pd.to_timedelta(dataset['charge_vs_expected_diff'], unit='s').round('1s')

        # Set a minimum threshold for charge duration
        min_charge_duration_threshold = pd.Timedelta(minutes=5)

        # Calculate 'miles_per_hour'
        dataset['miles_per_hour'] = (dataset['miles_requested'] / (dataset['charge_duration'].dt.total_seconds() / 3600)).round(2)

        # Filter out cases with charge duration below the threshold
        dataset = dataset[dataset['charge_duration'] >= min_charge_duration_threshold]

        # Remove rows with infinite or NaN values in 'miles_per_hour'
        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        dataset = dataset.dropna(subset=['miles_per_hour'])

        # Apply one-hot encoding to the 'vehicle_model' column
        dataset = pd.get_dummies(dataset, columns=['vehicle_model'], prefix='vehicle')

        # Extract datetime features to capture time of day patterns
        dataset['day_of_week'] = dataset['request_entry_time'].dt.dayofweek
        dataset['hour_of_day'] = dataset['request_entry_time'].dt.hour

        # Winsorize 'miles_per_hour'
        from scipy.stats.mstats import winsorize

        # Percentage limits for winsorization
        lower_limit = 0.0001  # 0.01%
        upper_limit = 0.0001   # 0.01%

        # Winsorize the 'miles_per_hour' column
        winsorized_miles_per_hour = winsorize(dataset['miles_per_hour'], limits=(lower_limit, upper_limit))

        # Replace the original 'miles_per_hour' column with the winsorized values
        dataset['miles_per_hour'] = winsorized_miles_per_hour

        # Define a threshold to consider data points as outliers
        outlier_threshold = 400

        # Filter out rows with 'miles_per_hour' greater than the threshold
        dataset = dataset[dataset['miles_per_hour'] <= outlier_threshold]

        # Dataset categorization
        dataset['charge_duration_minutes'] = dataset['charge_duration_seconds'] / 60
        conditions = [
            (dataset['charge_duration_minutes'] < 150),
            (dataset['charge_duration_minutes'] >= 150) & (dataset['charge_duration_minutes'] <= 250),
            (dataset['charge_duration_minutes'] > 250)
        ]
        labels = ['short', 'medium', 'long']
        dataset['duration_category'] = np.select(conditions, labels)


        #*** EDA Visualizations based on model filters ***
        # Interactive elements for subsetting data
        selected_hour = st.slider('Filter Hour for Model Prediction', 0, 23, 9)
        selected_day = st.selectbox('Filter Day for Model Prediction', range(7))

        # Creating a subset of data
        subset_data = dataset[(dataset['start_date_time'].dt.hour == selected_hour) & 
                              (dataset['day_of_week'] == selected_day)]

        # Preparing the subset for model prediction
        features_subset = subset_data[['miles_requested', 'max_charge_power', 'kwh_requested', 'energy_charged', 'day_of_week', 'hour_of_day']]
        target_subset = subset_data['duration_category']

        # Splitting the subset
        X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(features_subset, target_subset, test_size=0.2, random_state=42)

        # Training the model on the subset
        rf_classifier_sub = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier_sub.fit(X_train_sub, y_train_sub)

        # Displaying classification report for the subset
        # predictions_sub = rf_classifier_sub.predict(X_test_sub)
        # st.text(f'Random Forest Classification Report for Hour {selected_hour} and Day {selected_day}')
        # st.text(classification_report(y_test_sub, predictions_sub))
        predictions_sub = rf_classifier_sub.predict(X_test_sub)
        report_dict_sub = classification_report(y_test_sub, predictions_sub, output_dict=True)
        report_df_sub = pd.DataFrame(report_dict_sub).transpose()
        st.header(f'Random Forest Classification Report for Hour {selected_hour} and Day {selected_day}')
        st.dataframe(report_df_sub.style.format("{:.2f}"))

        # Separator/Header before overall visualizations
        st.header("Overall Dataset Visualizations")

        #*** Overall EDA Visualizations ***
        # Number of Vehicles vs. Charging Duration
        plt.figure(figsize=(12, 6))
        sns.histplot(dataset['charge_duration'].dt.total_seconds() / 60, bins=30, kde=True)
        plt.title('Number of Vehicles vs. Charging Duration')
        plt.xlabel('Charging Duration (Minutes)')
        plt.ylabel('Number of Vehicles')
        st.pyplot(plt)

        # Number of Vehicles vs. Energy Charged (kWh)
        plt.figure(figsize=(12, 6))
        sns.histplot(dataset['energy_charged'], bins=20, kde=True)
        plt.title('Distribution of Energy Charged (kWh)')
        plt.xlabel('Energy Charged (kWh)')
        plt.ylabel('Number of Vehicles')
        st.pyplot(plt)

        # Number of Vehicles vs. Day of the Week
        plt.figure(figsize=(12, 6))
        sns.histplot(dataset['day_of_week'], bins=7, kde=False)
        plt.title('Distribution of Charging by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Number of Vehicles')
        st.pyplot(plt)

        # Number of Vehicles vs. Start Hour of the Day
        plt.figure(figsize=(12, 6))
        sns.countplot(x=dataset['start_date_time'].dt.hour)
        plt.title('Number of Vehicles vs. Start Hour of the Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Vehicles')
        st.pyplot(plt)

        # Number of Vehicles vs. End Hour of the Day
        plt.figure(figsize=(12, 6))
        sns.countplot(x=dataset['end_date_time'].dt.hour)
        plt.title('Number of Vehicles vs. End Hour of the Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Vehicles')
        st.pyplot(plt)

        # Energy Charged vs. kWh Requested
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='kwh_requested', y='energy_charged', data=dataset)
        plt.title('Energy Charged vs. kWh Requested')
        plt.xlabel('kWh Requested')
        plt.ylabel('Energy Charged')
        st.pyplot(plt)

        # Random Forest Classification
        features = dataset[['miles_requested', 'max_charge_power', 'kwh_requested', 'energy_charged', 'day_of_week', 'hour_of_day']]
        target = dataset['duration_category']

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train, y_train)

        predictions = rf_classifier.predict(X_test)
        st.text('Random Forest Classification Report')

        # Display classification report as a DataFrame
        report_dict = classification_report(y_test, predictions, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        # st.header(f'Random Forest Classification Report for Complete Dataset')
        st.dataframe(report_df)