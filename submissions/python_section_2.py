import pandas as pd
import numpy as np
from datetime import time, timedelta

def calculate_distance_matrix(df: pd.DataFrame)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here

    # Get all unique toll locations from both columns 'id_start' and 'id_end'
    locations = pd.concat([df['id_start'], df['id_end']]).unique()
    
    # Initialize an empty distance matrix with all values as infinity
    distance_matrix = pd.DataFrame(np.inf, index=locations, columns=locations)
    
    # Set diagonal to 0 (distance from a location to itself is zero)
    np.fill_diagonal(distance_matrix.values, 0)
    
    # Fill in the matrix with direct distances from the DataFrame
    for _, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']
        
        # Set both directions as the distance (symmetric matrix)
        distance_matrix.loc[id_start, id_end] = distance
        distance_matrix.loc[id_end, id_start] = distance
    
    # Apply Floyd-Warshall algorithm for all pairs shortest paths
    for k in locations:
        for i in locations:
            for j in locations:
                # If the distance through k is shorter, update it
                if distance_matrix.loc[i, j] > distance_matrix.loc[i, k] + distance_matrix.loc[k, j]:
                    distance_matrix.loc[i, j] = distance_matrix.loc[i, k] + distance_matrix.loc[k, j]
    
    # Return the input DataFrame df with the computed distances
    return df

# Corrected dataset with equal length columns
data = {
    'id_start': [1001400, 1001402, 1001404, 1001406, 1001408, 1001410, 1001412, 1001414, 1001416, 1001418,
                 1001420, 1001422, 1001424, 1001426, 1001428, 1001430, 1001432, 1001434, 1001436, 1001436, 
                 1001438, 1001438, 1001440, 1001442, 1001488, 1004356, 1004354, 1004355, 1001444, 1001446, 
                 1001448, 1001450, 1001452, 1001454, 1001456, 1001458, 1001460, 1001460, 1001461, 1001462, 
                 1001464, 1001466, 1001468, 1001470],
    'id_end': [1001402, 1001404, 1001406, 1001408, 1001410, 1001412, 1001414, 1001416, 1001418, 1001420,
               1001422, 1001424, 1001426, 1001428, 1001430, 1001432, 1001434, 1001436, 1001438, 1001437,
               1001437, 1001440, 1001442, 1001488, 1004356, 1004354, 1004355, 1001444, 1001446, 1001448, 
               1001450, 1001452, 1001454, 1001456, 1001458, 1001460, 1001461, 1001462, 1001464, 1001466, 
               1001468, 1001470, 1001472],
    'distance': [9.7, 20.2, 16, 21.7, 11.1, 15.6, 18.2, 13.2, 13.6, 12.9, 9.6, 11.4, 18.6, 15.8, 8.6, 9, 
                 7.9, 4, 9, 5, 4, 10, 3.9, 4.5, 4, 2, 2, 0.7, 6.6, 9.6, 15.7, 9.9, 11.3, 13.6, 8.9, 5.1, 
                 12.8, 17.9, 5.1, 26.7, 8.5, 10.7, 10.6, 16]
}

# Creating the DataFrame from the corrected data
df = pd.DataFrame(data)

# Calculate the distance matrix and return the DataFrame
df_result = calculate_distance_matrix(df)

# Display the result
print(df_result)





def unroll_distance_matrix(df) ->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    # Ensure the input DataFrame has the required columns
    if not {'id_start', 'id_end', 'distance'}.issubset(df.columns):
        raise ValueError("Input DataFrame must contain 'id_start', 'id_end', and 'distance' columns.")

    # Create a new DataFrame for the unrolled data
    unrolled_data = []

    # Loop through each row in the original DataFrame
    for index, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']
        
        # Append to the unrolled_data list for each combination
        unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})
        
        # Avoid self-connections by checking the distance for other combinations
        for idx, other_row in df.iterrows():
            if id_start != other_row['id_start']:
                unrolled_data.append({'id_start': id_start, 'id_end': other_row['id_start'], 'distance': other_row['distance']})

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(unrolled_data)

    # Drop duplicates to ensure unique combinations
    df = df.drop_duplicates(subset=['id_start', 'id_end'])

    return df

# Example usage
data = {
    'id_start': [1001400, 1001402, 1001404, 1001406, 1001408, 1001410, 1001412, 1001414, 1001416, 1001418,
                 1001420, 1001422, 1001424, 1001426, 1001428, 1001430, 1001432, 1001434, 1001436, 1001436, 
                 1001438, 1001438, 1001440, 1001442, 1001488, 1004356, 1004354, 1004355, 1001444, 1001446, 
                 1001448, 1001450, 1001452, 1001454, 1001456, 1001458, 1001460, 1001460, 1001461, 1001462, 
                 1001464, 1001466, 1001468, 1001470],
    'id_end': [1001402, 1001404, 1001406, 1001408, 1001410, 1001412, 1001414, 1001416, 1001418, 1001420,
               1001422, 1001424, 1001426, 1001428, 1001430, 1001432, 1001434, 1001436, 1001438, 1001437,
               1001437, 1001440, 1001442, 1001488, 1004356, 1004354, 1004355, 1001444, 1001446, 1001448, 
               1001450, 1001452, 1001454, 1001456, 1001458, 1001460, 1001461, 1001462, 1001464, 1001466, 
               1001468, 1001470, 1001472],
    'distance': [9.7, 20.2, 16, 21.7, 11.1, 15.6, 18.2, 13.2, 13.6, 12.9, 9.6, 11.4, 18.6, 15.8, 8.6, 9, 
                 7.9, 4, 9, 5, 4, 10, 3.9, 4.5, 4, 2, 2, 0.7, 6.6, 9.6, 15.7, 9.9, 11.3, 13.6, 8.9, 5.1, 
                 12.8, 17.9, 5.1, 26.7, 8.5, 10.7, 10.6, 16]
    
 } 
df = pd.DataFrame(data)

# Call the function
result_df = unroll_distance_matrix(df)
print(result_df)

def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():


    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> pd.DataFrame:
    # Ensure the input DataFrame has the required columns
    if not {'id_start', 'distance'}.issubset(df.columns):
        raise ValueError("Input DataFrame must contain 'id_start' and 'distance' columns.")
    
    # Calculate the average distance for the reference id_start
    average_distance = df[df['id_start'] == reference_id]['distance'].mean()
    
    if np.isnan(average_distance):
        raise ValueError(f"No distances found for id_start {reference_id}.")
    
    # Calculate the 10% threshold
    lower_bound = average_distance * 0.9
    upper_bound = average_distance * 1.1

    # Find id_start values within the 10% threshold of the average distance
    ids_within_threshold = df[(df['distance'] >= lower_bound) & (df['distance'] <= upper_bound)]
    
    # Get the unique id_start values and create a new DataFrame
    result_df = pd.DataFrame({'id_start': ids_within_threshold['id_start'].unique()})

    return result_df

# Example usage
data = {
    'id_start': [1001400, 1001402, 1001404, 1001406, 1001408, 1001410, 1001412, 1001414, 1001416, 1001418,
                 1001420, 1001422, 1001424, 1001426, 1001428, 1001430, 1001432, 1001434, 1001436, 1001436, 
                 1001438, 1001438, 1001440, 1001442, 1001488, 1004356, 1004354, 1004355, 1001444, 1001446, 
                 1001448, 1001450, 1001452, 1001454, 1001456, 1001458, 1001460, 1001460, 1001461, 1001462, 
                 1001464, 1001466, 1001468, 1001470],
    'id_end': [1001402, 1001404, 1001406, 1001408, 1001410, 1001412, 1001414, 1001416, 1001418, 1001420,
               1001422, 1001424, 1001426, 1001428, 1001430, 1001432, 1001434, 1001436, 1001438, 1001437,
               1001437, 1001440, 1001442, 1001488, 1004356, 1004354, 1004355, 1001444, 1001446, 1001448, 
               1001450, 1001452, 1001454, 1001456, 1001458, 1001460, 1001461, 1001462, 1001464, 1001466, 
               1001468, 1001470, 1001472],
    'distance': [9.7, 20.2, 16, 21.7, 11.1, 15.6, 18.2, 13.2, 13.6, 12.9, 9.6, 11.4, 18.6, 15.8, 8.6, 9, 
                 7.9, 4, 9, 5, 4, 10, 3.9, 4.5, 4, 2, 2, 0.7, 6.6, 9.6, 15.7, 9.9, 11.3, 13.6, 8.9, 5.1, 
                 12.8, 17.9, 5.1, 26.7, 8.5, 10.7, 10.6, 16]
    
 } 
df = pd.DataFrame(data)

# Assuming we call the function with id_start = 1
reference_id = 1
result_df = find_ids_within_ten_percentage_threshold(df, reference_id)
print(result_df)



def calculate_toll_rate(df) ->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here 
    # Ensure the input DataFrame has the required column
    if 'distance' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'distance' column.")
    
    # Define rate coefficients for each vehicle type
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    # Calculate toll rates for each vehicle type
    for vehicle, rate in rates.items():
        df[vehicle] = df['distance'] * rate
    
    return df

# Example usage
data = {
    'id_start': [1001400, 1001402, 1001404, 1001406, 1001408, 1001410, 1001412, 1001414, 1001416, 1001418,
                 1001420, 1001422, 1001424, 1001426, 1001428, 1001430, 1001432, 1001434, 1001436, 1001436, 
                 1001438, 1001438, 1001440, 1001442, 1001488, 1004356, 1004354, 1004355, 1001444, 1001446, 
                 1001448, 1001450, 1001452, 1001454, 1001456, 1001458, 1001460, 1001460, 1001461, 1001462, 
                 1001464, 1001466, 1001468, 1001470],
    'id_end': [1001402, 1001404, 1001406, 1001408, 1001410, 1001412, 1001414, 1001416, 1001418, 1001420,
               1001422, 1001424, 1001426, 1001428, 1001430, 1001432, 1001434, 1001436, 1001438, 1001437,
               1001437, 1001440, 1001442, 1001488, 1004356, 1004354, 1004355, 1001444, 1001446, 1001448, 
               1001450, 1001452, 1001454, 1001456, 1001458, 1001460, 1001461, 1001462, 1001464, 1001466, 
               1001468, 1001470, 1001472],
    'distance': [9.7, 20.2, 16, 21.7, 11.1, 15.6, 18.2, 13.2, 13.6, 12.9, 9.6, 11.4, 18.6, 15.8, 8.6, 9, 
                 7.9, 4, 9, 5, 4, 10, 3.9, 4.5, 4, 2, 2, 0.7, 6.6, 9.6, 15.7, 9.9, 11.3, 13.6, 8.9, 5.1, 
                 12.8, 17.9, 5.1, 26.7, 8.5, 10.7, 10.6, 16]
    
 } 
df = pd.DataFrame(data)

# Call the function to calculate toll rates
result_df = calculate_toll_rate(df)
print(result_df)



def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    
    # Ensure the input DataFrame has the required columns
    if not {'id_start', 'id_end', 'distance', 'moto', 'car', 'rv', 'bus', 'truck'}.issubset(df.columns):
        raise ValueError("Input DataFrame must contain 'id_start', 'id_end', 'distance', and vehicle rate columns.")
    
    # Define the days of the week and time intervals
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Time intervals and their corresponding discount factors
    time_intervals = {
        'weekday': [
            (time(0, 0), time(10, 0), 0.8),
            (time(10, 0), time(18, 0), 1.2),
            (time(18, 0), time(23, 59), 0.8)
        ],
        'weekend': [
            (time(0, 0), time(23, 59), 0.7)
        ]
    }
    
    # Initialize a list to collect new rows
    new_rows = []

    # Create a full day span for each unique id_start, id_end pair
    for (id_start, id_end), group in df.groupby(['id_start', 'id_end']):
        for day in days_of_week:
            for hour in range(24):
                # Create the start and end times
                start_time = time(hour, 0)
                end_time = time(hour, 59)
                
                # Determine the discount factor based on the day of the week
                if day in ['Saturday', 'Sunday']:
                    discount_factor = time_intervals['weekend'][0][2]  # constant discount for weekends
                else:
                    # Check which weekday interval applies
                    discount_factor = 1.0  # default to no discount
                    for start, end, factor in time_intervals['weekday']:
                        if start_time >= start and end_time <= end:
                            discount_factor = factor
                            break

                # Calculate new toll rates based on the discount factor
                for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                    toll_rate = group[vehicle].values[0] * discount_factor  # get the original toll rate
                    new_rows.append({
                        'id_start': id_start,
                        'id_end': id_end,
                        'start_day': day,
                        'start_time': start_time,
                        'end_day': day,
                        'end_time': end_time,
                        vehicle: toll_rate
                    })

    # Create a new DataFrame from the collected rows
    result_df = pd.DataFrame(new_rows)
    
    return result_df

df = pd.DataFrame(data)

# Call the function to calculate time-based toll rates
result_df = calculate_time_based_toll_rates(df)
print(result_df)


   
