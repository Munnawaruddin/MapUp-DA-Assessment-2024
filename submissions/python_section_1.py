from typing import Dict, List

import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.

    for i in range(0, len(lst), n):
        end = min(i + n, len(lst))  # Calculate the end of the current group
        
        # Reverse the current group in place
        left, right = i, end - 1
        while left < right:
            lst[left], lst[right] = lst[right], lst[left]  # Swap elements
            left += 1
            right -= 1
    
    return lst

# Test cases
print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3))  # Output: [3, 2, 1, 6, 5, 4, 8, 7]
print(reverse_by_n_elements([1, 2, 3, 4, 5], 2))           # Output: [2, 1, 4, 3, 5]
print(reverse_by_n_elements([10, 20, 30, 40, 50, 60, 70], 4))  # Output: [40, 30, 20, 10, 70, 60, 50]


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here

    length_dict = {}
    
    # Grouping strings by their length
    for string in lst:
        length = len(string)  # Get the length of the string
        if length not in length_dict:
            length_dict[length] = []  # Initialize the list for this length if not present
        length_dict[length].append(string)  # Add the string to the corresponding list
    
    # Sort the dictionary by the key (length)
    return dict(sorted(length_dict.items()))

# test cases
print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))
print(group_by_length(["one", "two", "three", "four"]))


from typing import Any, Dict

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
    def _flatten(current_dict: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
        items = {}
        for key, value in current_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                # Recursively flatten dictionaries
                items.update(_flatten(value, new_key))
            elif isinstance(value, list):
                # Handle lists by iterating with indices
                for i, item in enumerate(value):
                    list_key = f"{new_key}[{i}]"
                    if isinstance(item, dict):
                        # If the list item is a dictionary, recurse into it
                        items.update(_flatten(item, list_key))
                    else:
                        # Otherwise, directly add the list item
                        items[list_key] = item
            else:
                # If neither dict nor list, just add the value
                items[new_key] = value
        return items
    
    return _flatten(nested_dict)

# test case
nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}

flattened_dict = flatten_dict(nested_dict)
print(flattened_dict)



def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    result = []  # List to store unique permutations

    def backtrack(start: int):
        if start == len(nums):
            result.append(nums[:])  # Append a copy of the current permutation
            return
        
        seen = set()  # To keep track of duplicates
        for i in range(start, len(nums)):
            if nums[i] in seen:
                continue  # Skip duplicates
            
            seen.add(nums[i])  # Mark this number as seen
            nums[start], nums[i] = nums[i], nums[start]  # Swap to create new permutation
            backtrack(start + 1)  # Recurse
            nums[start], nums[i] = nums[i], nums[start]  # Swap back to restore order

    nums.sort()  # Sort the list to facilitate duplicate handling
    backtrack(0)  # Start the backtracking process
    return result  # Return the list of unique permutations
    pass
# test case
input_list = [1, 1, 2]
unique_permutations_result = unique_permutations(input_list)
print(unique_permutations_result)


import re
def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    # Regular expression pattern for matching the specified date formats
    date_pattern = r'\b(\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}\.\d{2}\.\d{2})\b'

    # Find all matching dates
    dates = re.findall(date_pattern, text)

    return dates
    pass

# test case
input_text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
found_dates = find_all_dates(input_text)
print(found_dates)



def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Earth radius in meters (mean radius)
    radius = 6371000  
    distance = radius * c
    return distance

def decode_polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    # Decode the polyline string
    coordinates = polyline.decode(polyline_str)
    
    # Create a DataFrame from the coordinates
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    # Initialize the distance column with zeros
    df['distance'] = 0.0
    
    # Calculate distances between successive points
    for i in range(1, len(df)):
        lat1, lon1 = df.iloc[i - 1]['latitude'], df.iloc[i - 1]['longitude']
        lat2, lon2 = df.iloc[i]['latitude'], df.iloc[i]['longitude']
        df.at[i, 'distance'] = haversine(lat1, lon1, lat2, lon2)
    
    return df

# Example usage
polyline_str = "i}~fHpy~fN}n@{z@WmAc@}F"  # Corrected polyline string
result_df = decode_polyline_to_dataframe(polyline_str)
print(result_df)

   



def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then transform each element 
    by replacing it with the sum of all elements in the same row and column, excluding itself.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code hear
    n = len(matrix)
    
    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[0] * n for _ in range(n)]  # Initialize the rotated matrix
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]
    
    # Step 2: Replace each element with the sum of its row and column excluding itself
    final_matrix = [[0] * n for _ in range(n)]  # Initialize the final transformed matrix
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i])  # Sum of the entire row
            col_sum = sum(rotated_matrix[k][j] for k in range(n))  # Sum of the entire column
            final_matrix[i][j] = row_sum + col_sum - rotated_matrix[i][j]  # Exclude the current element
    
    return final_matrix  # Adjusted return statement

# Test cases
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = rotate_and_multiply_matrix(matrix)
print(result)



def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Checks if each (id, id_2) pair has timestamps covering a full 24-hour period 
    and spans all 7 days of the week.
    
    Args:
    - df (pd.DataFrame): DataFrame containing columns id, id_2, 
      startDay, startTime, endDay, endTime.
    
    Returns:
    - pd.Series: A boolean Series indicating if each (id, id_2) pair has incorrect timestamps.
    """
    # Combine startDay and startTime into a single datetime column for easier handling
    df['start_datetime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end_datetime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

    # Create a multi-index based on id and id_2
    grouped = df.groupby(['id', 'id_2'])

    def check_time(group):
        # Check if the time spans a full 24-hour period
        start_min = group['start_datetime'].min().floor('D')
        end_max = group['end_datetime'].max().ceil('D')
        
        # Get all unique days covered in the timestamps
        unique_days = group['start_datetime'].dt.day_name().unique()
        
        # Check if it covers all days of the week and spans full 24 hours
        full_day_covered = (end_max - start_min) >= pd.Timedelta(days=7)
        all_days_present = len(unique_days) == 7
        
        return not (full_day_covered and all_days_present)

    # Apply the check to each group and return a Series with multi-index
    results = grouped.apply(check_time)
    
    return pd.Series(results)  # Return as a pd.Series as required

