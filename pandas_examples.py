"""
pandas_examples.py

A concise, well-commented set of examples demonstrating common pandas DataFrame
manipulations. Run this file to see printed outputs for each section.

Requirements: pandas, numpy
Install with (PowerShell):
    pip install pandas numpy

This script is intentionally verbose with comments so you can learn what each
operation does. Each section prints a header so you can follow the output.
"""

import pandas as pd
import numpy as np
from io import StringIO


def show(title, obj):
    
    """Utility for printing section headers and objects clearly."""
    print('\n' + '=' * 80)
    print(f'{title}')
    print('-' * 80)
    # For DataFrame/Series, print a short summary then the full repr
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        print(obj)
    else:
        print(repr(obj))


def main():
    # 1) Create DataFrame from a Python dict (common pattern)
    # Columns become keys; values are lists (must have same length)
    data = {
        'product': ['A', 'B', 'C', 'A', 'B', 'C'],
        'region': ['North', 'South', 'East', 'West', 'North', 'South'],
        'price': [10.0, 20.5, 15.0, 11.0, np.nan, 14.0],  # note one NaN
        'quantity': [5, 10, 7, 3, 8, 2],
        'date': [
            '2021-01-01',
            '2021-01-02',
            '2021-02-01',
            '2021-02-03',
            '2021-03-05',
            '2021-03-07',
        ],
    }

    df = pd.DataFrame(data)
    show('1) Original DataFrame', df)
    
    # 2) Basic inspection methods
    show('2a) DataFrame.head() - first rows', df.head(3))
    show('2b) DataFrame.info() - schema and non-null counts', df.info())
    show('2c) DataFrame.describe() - numeric summary', df.describe())

    # 3) Convert types: parse the date column into datetime objects
    # This lets us do time-based operations later
    df['date'] = pd.to_datetime(df['date'])
    show('3) After parsing `date` column to datetime', df)

    # 4) Selection: column access, loc, iloc
    # - Single column returns a Series
    show('4a) Select single column `price`', df['price'])
    # - loc selects by label (row index and column labels)
    show('4b) Select rows 0..2 and columns `product`,`price` using .loc',
         df.loc[0:2, ['product', 'price']])
    # - iloc selects by integer positions
    show('4c) Select first two rows using .iloc', df.iloc[0:2])

    # 5) Boolean indexing (filtering rows)
    # Rows where price is greater than 12 (note NaN comparisons are False)
    show('5) Filter rows where price > 12', df[df['price'] > 12])

    # 6) Handling missing values
    # Detect missing values
    show('6a) Missing values (isna)', df.isna())
    # Fill missing price with the column mean (simple imputation)
    mean_price = df['price'].mean(skipna=True)
    show('6b) price mean', mean_price)
    df_filled = df.copy()
    df_filled['price'] = df_filled['price'].fillna(mean_price)
    show('6c) After fillna(price mean)', df_filled)
    # Drop rows with any missing values
    show('6d) Drop rows with any NaN', df.dropna())

    # 7) Creating new columns
    # Example: compute revenue = price * quantity
    df_filled['revenue'] = df_filled['price'] * df_filled['quantity']
    show('7) Added `revenue` column', df_filled)

    # 8) Applying functions
    # Use a vectorized operation to create a discount flag
    df_filled['is_expensive'] = df_filled['price'] > 15
    show('8a) Vectorized boolean column `is_expensive`', df_filled)
    # Use .apply for elementwise transformation (slower than vectorized ops)
    def price_label(p):
        if p < 12:
            return 'low'
        elif p < 16:
            return 'medium'
        else:
            return 'high'

    df_filled['price_label'] = df_filled['price'].apply(price_label)
    show('8b) price_label via apply()', df_filled)

    # 9) GroupBy aggregation: get total revenue per product
    grouped = df_filled.groupby('product').agg(
        total_revenue=('revenue', 'sum'),
        avg_price=('price', 'mean'),
        total_quantity=('quantity', 'sum'),
    )
    show('9) GroupBy product aggregations', grouped)

    # 10) Pivot tables: product x region revenue
    pivot = pd.pivot_table(
        df_filled,
        index='product',
        columns='region',
        values='revenue',
        aggfunc='sum',
        fill_value=0,
    )
    show('10) Pivot table (product x region revenue)', pivot)

    # 11) Merging DataFrames (SQL-like joins)
    # Create a small DataFrame with product info
    product_info = pd.DataFrame({
        'product': ['A', 'B', 'C', 'D'],
        'category': ['alpha', 'beta', 'alpha', 'gamma'],
        'cost': [6.0, 12.0, 8.0, 5.0],
    })
    show('11a) product_info', product_info)
    # Left join to attach category and cost to our main table
    merged = pd.merge(df_filled, product_info, on='product', how='left')
    show('11b) After merge (left join) with product_info', merged)

    # 12) Reshaping: melt (wide -> long) and sort_values
    # Suppose we have a wide table of monthly sales; here we'll make a tiny example
    wide = pd.DataFrame({
        'product': ['A', 'B'],
        'Jan': [100, 120],
        'Feb': [110, 115],
    })
    show('12a) Wide table', wide)
    long = wide.melt(id_vars='product', var_name='month', value_name='sales')
    show('12b) Melted to long format', long)
    # Sort values by sales descending
    show('12c) Sorted by sales descending', long.sort_values('sales', ascending=False))

    # 13) Read/Write CSV in-memory using StringIO (no disk required)
    csv_buffer = StringIO()
    df_filled.to_csv(csv_buffer, index=False)
    csv_text = csv_buffer.getvalue()
    show('13a) CSV text (first 200 chars)', csv_text[:200])
    # Read it back into a DataFrame
    csv_buffer.seek(0)
    df_from_csv = pd.read_csv(csv_buffer, parse_dates=['date'])
    show('13b) Read back from CSV into DataFrame', df_from_csv)

    # 14) Time series basics: set a datetime index and resample
    df_time = df_filled.set_index('date')
    show('14a) Set `date` as index', df_time)
    # Resample monthly and sum revenue
    monthly = df_time.resample('M').agg({'revenue': 'sum', 'quantity': 'sum'})
    show('14b) Monthly resampled sums', monthly)

    # 15) Export a small sample to JSON (dictionary orient)
    sample_json = df_filled.head().to_json(orient='records', date_format='iso')
    show('15) Sample JSON (records orient)', sample_json)

    print('\nAll examples completed successfully.')

    # export dataframe to csv
    df_filled.to_csv('filled_data.csv', index=False)

    # read in csv
    df_readback = pd.read_csv('filled_data.csv', parse_dates=['date'])
if __name__ == '__main__':
    main()
