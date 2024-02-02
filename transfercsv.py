import pandas as pd

# Replace 'your_file.xlsx' with the actual file path
xlsx_file = 'Asian Speakers.xlsx'

df = pd.read_excel(xlsx_file)

df['Combined_Column'] = df['vid_id'].astype(str) + '_' + df['desc_lang'].astype(str)

df = df[['vid_id', 'Combined_Column', 'Start', 'End', 'x', 'y', 'title_lang','desc_lang','long_desc_lang','is_asian']]

selected_columns = ['vid_id', 'Combined_Column', 'Start', 'End', 'desc_lang']


# Select only the desired columns
df_selected = df[selected_columns]

# Save as CSV
df_selected.to_csv('Asian_speaker.csv', index=False)