import pandas as pd

# Load the CSV
print("Loading CSV...")
df = pd.read_csv("data/chess_1000_no_x/chess_1000_no_x.csv")

# Remove all '+' characters
print("Removing '+' characters...")
df['transcript'] = df['transcript'].str.replace('+', '', regex=False)

# Save back to CSV
print("Saving...")
df.to_csv("data/chess_1000_no_x/chess_1000_no_x.csv", index=False)

print("Done! All '+' characters have been removed.")
