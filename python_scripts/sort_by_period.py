import pandas as pd

# wanted_periods = [
#     "medieval",
#     "reformation",
#     "viking",
#     "bronze",
#     "iron"
# ]

# Load the CSV files
dime_data_df = pd.read_csv('/mnt/c/projs/Classification-and-3D-reconstruction-of-archaeological-artifacts/dime_data/DIME billeder.csv', delimiter=';')
thesaurus_df = pd.read_csv('/mnt/c/projs/Classification-and-3D-reconstruction-of-archaeological-artifacts/dime_data/thesaurus.csv')

# Merge the dataframes on thesaurus
merged_df = pd.merge(dime_data_df, thesaurus_df, left_on='thesaurus', right_on='id')

# Filter the merged dataframe for each period and concatenate the results
filtered_dfs = []
filtered_dfs.append(merged_df[(merged_df['periode'].str.contains("medieval|historic", case=False)) & (merged_df['keywords'].str.contains('mønt', case=False))])

# Concatenate the filtered dataframes
filtered_df = pd.concat(filtered_dfs)

# Select only the necessary columns
result_df = filtered_df[['filnavn', 'periode']]

# Export the filenames to a text file
result_df['filnavn'].to_csv('/mnt/c/projs/Classification-and-3D-reconstruction-of-archaeological-artifacts/dime_data/medievalAndHistoric.txt', index=False, header=False)

print("File with filenames created successfully.")

# Print the result
print(result_df)
