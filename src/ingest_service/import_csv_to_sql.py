import csv
import re
import psycopg2
from datetime import datetime

import argparse

# Set up the argument parser
parser = argparse.ArgumentParser(description='Database connection parameters.')
parser.add_argument('--dbname', type=str, default='du_products', help='Database name')
parser.add_argument('--user', type=str, default='postgres', help='Database user')
parser.add_argument('--password', type=str, default='password', help='Database password')
parser.add_argument('--host', type=str, default='localhost', help='Database host')
parser.add_argument('--port', type=str, default='5432', help='Database port')

# Parse the arguments
args = parser.parse_args()

# Database connection parameters
db_params = {
    'dbname': args.dbname,
    'user': args.user,
    'password': args.password,
    'host': args.host,
    'port': args.port
}

# CSV file path
csv_file_path = './data/gear-store.csv'

# Connect to the database
conn = psycopg2.connect(**db_params)
cur = conn.cursor()

# Create the table if it doesn't exist
with open("init.txt", "r") as file:
    create_table_query = file.read()
    cur.execute(create_table_query)

# Open the CSV file and insert data
with open(csv_file_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header row

    for row in reader:
        # Extract data from CSV row
        category, subcategory, name, description, price = row
        price = float(price)  # Convert price to float

        # Insert data into the brands table (assuming NVIDIA is the only brand for now)
        cur.execute("INSERT INTO du_products.brands (name) VALUES (%s) ON CONFLICT (name) DO NOTHING RETURNING brand_id", ('NVIDIA',))
        brand_result = cur.fetchone()
        brand_id = brand_result[0] if brand_result else None

        # If brand_id is None, fetch the existing brand_id
        if brand_id is None:
            cur.execute("SELECT brand_id FROM du_products.brands WHERE name = 'NVIDIA'")
            brand_id = cur.fetchone()[0]

        # Insert data into the categories table
        cur.execute("INSERT INTO du_products.categories (name, slug) VALUES (%s, %s) ON CONFLICT (slug) DO NOTHING RETURNING category_id", (category, category.lower().replace(' ', '-')))
        category_result = cur.fetchone()
        category_id = category_result[0] if category_result else None

        # If category_id is None, fetch the existing category_id
        if category_id is None:
            cur.execute("SELECT category_id FROM du_products.categories WHERE name = %s", (category,))
            category_id = cur.fetchone()[0]

        # Insert data into the products table
        cur.execute(
            """
            INSERT INTO du_products.products (sku, brand_id, name, slug, description, base_price, currency, stock_status, condition)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (sku) DO NOTHING
            RETURNING product_id
            """,
            ('SKU-' + name.replace(' ', '-'), brand_id, name, name.lower().replace(' ', '-'), description, price, 'USD', 'IN_STOCK', 'NEW')
        )
        product_result = cur.fetchone()
        product_id = product_result[0] if product_result else None

        # If product_id is None, fetch the existing product_id
        if product_id is None:
            cur.execute("SELECT product_id FROM du_products.products WHERE name = %s", (name,))
            product_id = cur.fetchone()[0]

        # Insert data into the product_categories table
        cur.execute("INSERT INTO du_products.product_categories (product_id, category_id) VALUES (%s, %s) ON CONFLICT (product_id, category_id) DO NOTHING", (product_id, category_id))

        # Insert data into the product_variants table (assuming a default variant for each product)
        cur.execute(
            """
            INSERT INTO du_products.product_variants (product_id, sku_variant, retail_price, stock_quantity)
            VALUES (%s, %s, %s, %s)
            """,
            (product_id, 'VARIANT-' + name.replace(' ', '-'), price, 100)
        )

        # Insert data into the specifications table
        cur.execute("INSERT INTO du_products.specifications (name, category_id) VALUES (%s, %s) ON CONFLICT (name) DO NOTHING RETURNING spec_id", ('Specification', category_id))
        spec_result = cur.fetchone()
        spec_id = spec_result[0] if spec_result else None

        # If spec_id is None, fetch the existing spec_id
        if spec_id is None:
            cur.execute("SELECT spec_id FROM du_products.specifications WHERE name = 'Specification'")
            spec_id = cur.fetchone()[0]

        # Insert data into the product_specifications table
        cur.execute("INSERT INTO du_products.product_specifications (product_id, spec_id, value) VALUES (%s, %s, %s)", (product_id, spec_id, 'Value'))

        # Insert data into the product_images table
        cur.execute(
            """
            INSERT INTO du_products.product_images (product_id, image_url, is_primary, sort_order)
            VALUES (%s, %s, %s, %s)
            """,
            (product_id, 'https://via.placeholder.com/150', True, 1)
        )
        
        # Commit after each row
        conn.commit()

# Commit the changes and close the connection
conn.commit()
cur.close()
conn.close()

print("CSV Data imported successfully!")
