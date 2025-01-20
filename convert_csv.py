import pandas as pd
import os
import re

def create_valid_filename(s):
    s = re.sub(r"[^\w\-_\. ]", "", s)
    s = s.replace(" ", "_")
    return s

# Read the CSV file
df = pd.read_csv("data/gear-store.csv")

# Create a directory for text files if it doesnt exist
os.makedirs("data/product_descriptions", exist_ok=True)

# Convert each row to a text file
for _, row in df.iterrows():
    # Create filename from product name
    filename = create_valid_filename(row["name"]) + ".txt"
    filepath = os.path.join("data/product_descriptions", filename)
    
    # Create content
    content = f"""Product: {row["name"]}
Category: {row["category"]}
Price: ${row["price"]}
Description: {row["description"]}
"""
    
    # Write to file
    with open(filepath, "w") as f:
        f.write(content)

print("Created text files for all products") 