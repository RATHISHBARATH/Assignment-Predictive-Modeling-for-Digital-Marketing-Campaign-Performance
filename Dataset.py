# I used this script to create random data set

import pandas as pd
import numpy as np
import random
import string
from IPython.display import HTML

# Function to generate random headlines
def generate_headline():
    words = ["Amazing", "Incredible", "Exclusive", "Limited", "Special", "Offer"]
    return ' '.join(random.choices(words, k=3))

# Function to generate random CTAs
def generate_cta():
    ct_as = ["Buy Now", "Sign Up", "Learn More", "Get Started", "Join Now"]
    return random.choice(ct_as)

# Function to generate random hero images
def generate_hero_image():
    images = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg", "image5.jpg"]
    return random.choice(images)

# Function to generate random offers
def generate_offer():
    offers = ["10% Off", "Free Trial", "BOGO", "Free Shipping", "Discounted Price"]
    return random.choice(offers)

# Generate the dataset
np.random.seed(42)  # For reproducibility
num_rows = 100000
data = {
    'Campaign ID': np.arange(1, num_rows + 1),
    'Impressions': np.random.randint(1000, 100000, size=num_rows),
    'Engagements': np.random.randint(100, 50000, size=num_rows),
    'Spend': np.round(np.random.uniform(100, 10000, size=num_rows), 2),
    'Headline': [generate_headline() for _ in range(num_rows)],
    'CTA': [generate_cta() for _ in range(num_rows)],
    'Hero Image': [generate_hero_image() for _ in range(num_rows)],
    'Offer': [generate_offer() for _ in range(num_rows)],
}

# Calculate derived metrics
data['CTR'] = np.round((data['Engagements'] / data['Impressions']) * 100, 2)
data['Conversion Rate'] = np.round(np.random.uniform(0.1, 10, size=num_rows), 2)
data['ROI'] = np.round((np.random.uniform(1, 100, size=num_rows) - data['Spend']) / data['Spend'], 2)

# Create a DataFrame
df = pd.DataFrame(data)

# Generate insights
insight_types = ["High Engagement", "Low Spend", "High ROI", "Low CTR", "High Conversion Rate"]
df['Insight Type'] = [random.choice(insight_types) for _ in range(num_rows)]
df['Insight Description'] = ["This campaign had " + insight for insight in df['Insight Type']]

# Save the dataset to a CSV file
file_name = 'insights_dataset.csv'
df.to_csv(file_name, index=False)

# Create a download link
def create_download_link(filename):
    from IPython.display import HTML
    import base64

    # Encode the file to base64
    with open(filename, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()

    # Create an HTML link
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return HTML(href)

# Display the download link
create_download_link(file_name)

# I used this script to create random data set
