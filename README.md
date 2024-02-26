# Time Series Forecasting Model

## Overview
This project contains a Python script for forecasting product demand in a retail store using time series analysis. The script uses the SARIMA model from the `statsmodels` library to predict future sales based on historical data.

## Dataset
The dataset `retail_sales.csv` should have the following columns:
- Date: The date of the sales record.
- Article_ID: The ID of the article sold.
- Country_Code: The country code where the sale occurred.
- Sold_Units: The number of units sold.

## Requirements
- Python 3.x
- pandas
- numpy
- matplotlib
- statsmodels

## Setup
1. Ensure that Python and all required libraries are installed.
2. Place the `retail_sales.csv` file in the same directory as the script.

## Usage
Run the script using the following command:
```bash
python time_series_forecasting.py
