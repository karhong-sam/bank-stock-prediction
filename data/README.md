Here’s the revised **README** incorporating your GitHub profile and adhering to best practices:

---

# README for Dataset Collection and Preprocessing

## Overview
This dataset combines data from various sources, including Yahoo Finance, Bank Negara Malaysia (BNM), and the World Bank. It has been meticulously preprocessed and standardized to facilitate research on financial forecasting and stock market analysis.

---

## Metadata

### General Information
- **Title**: Integrated Financial Dataset for MBB Stock Prediction
- **Description**: A comprehensive dataset combining stock data, macroeconomic indicators, and monetary policy rates to support financial forecasting research.
- **Creators**:  
  - Sam Kar Hong (Primary Researcher)  
  - GitHub Profile: [@karhong-sam](https://github.com/karhong-sam)
- **Sources**:
  - [Bank Negara Malaysia (BNM)](https://www.bnm.gov.my/monetary-stability/opr-decisions)
  - [World Bank DataBank](https://databank.worldbank.org/reports.aspx?source=2&series=NY.GDP.MKTP.KD&country=#)
  - [Yahoo Finance](https://finance.yahoo.com/)
- **Date of Creation**: January 2025
- **Time Period Covered**: July 1, 2004, to August 1, 2024
- **Geographical Focus**: Malaysia
- **Keywords**: Stock Price Prediction, Financial Forecasting, Macroeconomic Indicators, Deep Learning

---

## Data Sources
1. **Overnight Policy Rate (OPR)**
   - **Source**: [Bank Negara Malaysia - OPR Decisions](https://www.bnm.gov.my/monetary-stability/opr-decisions)
   - **Collection Method**: Web scraping using Selenium.
   - **Description**: Monthly OPR data extracted from BNM's official website.

2. **Macroeconomic Indicators**
   - **Source**: [World Bank Data](https://databank.worldbank.org/reports.aspx?source=2&series=NY.GDP.MKTP.KD&country=#)
   - **Indicators**:
     - **GDP** (constant 2015 MYR)
     - **GDP Growth** (YoY %)
     - **Inflation Rate** (%)

3. **Stock Data**
   - **Source**: Yahoo Finance
   - **Stock**: Malayan Banking Berhad (MBB) (`1155.KL`)
   - **Period**: Daily data from July 1, 2004, to August 1, 2024.

---

## Data Collection Steps

### 1. OPR Data Collection
- **Method**: Web scraping using Selenium.
- **Steps**:
  1. Accessed BNM's website.
  2. Extracted OPR data from the HTML table rendered by JavaScript.
  3. Processed the data to retain only the `Date` and `New OPR Level (%)` columns.
  4. Standardized the date to the first day of the month for consistency.

- **Code Snippet**:
```python
opr_df_cleaned['date'] = opr_df_cleaned['date'].apply(lambda x: x.replace(day=1) if pd.notnull(x) else None)
```

### 2. Macroeconomic Data
- **Source**: Downloaded CSV from the World Bank DataBank.
- **Steps**:
  1. Downloaded raw data for GDP, GDP Growth, and Inflation Rate.
  2. Converted GDP from USD to MYR using an approximate exchange rate of 4.5.
  3. Renamed columns for clarity and standardized naming conventions.

- **Code Snippet**:
```python
gdp_data = gdp_data.rename(columns={'GDP (constant 2015 US$)':'GDP (constant 2015 MYR)'})
gdp_data['GDP (constant 2015 MYR)'] = gdp_data['GDP (constant 2015 MYR)'] * 4.5
```

### 3. Stock Data (MBB)
- **Method**: Extracted daily stock data using the `yfinance` library.
- **Steps**:
  1. Retrieved historical data for MBB (`1155.KL`) from July 1, 2004, to August 1, 2024.
  2. Included OHLC (Open, High, Low, Close) and Adjusted Close prices alongside trading volume.

---

## Preprocessing Steps

### 1. Date Standardization
- Unified the date index across all datasets to daily frequency spanning the period from July 1, 2004, to August 1, 2024.

### 2. Handling Missing Data
- Applied forward or backward filling methods (`bfill` or `ffill`) to fill in missing values for:
  - OPR: Filled missing months using backward fill (`bfill`).
  - Macroeconomic indicators: Filled missing values to ensure continuous data.
  - Stock data: Aligned and forward-filled to maintain consistency with macroeconomic data.

### 3. Final Integration
- Merged all datasets into a single DataFrame based on the standardized date index.
- Ensured consistent column naming and documented the source of each variable.

---

## Licensing
This dataset and associated code are licensed under the **Apache License 2.0**.  

**Copyright [2024] [@karhong-sam]**  
GitHub Profile: [@karhong-sam](https://github.com/karhong-sam)

You may use, copy, modify, and distribute this dataset, provided proper attribution is given.  
For full license details, refer to the [LICENSE](LICENSE) file or visit [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

---

## Attribution
This dataset includes data derived from the following sources:
1. [Yahoo Finance](https://finance.yahoo.com/): Stock data for Malayan Banking Berhad (MBB) (`1155.KL`).
2. [Bank Negara Malaysia](https://www.bnm.gov.my/monetary-stability/opr-decisions): Overnight Policy Rate (OPR) data.
3. [World Bank DataBank](https://databank.worldbank.org/): GDP, GDP Growth, and Inflation data.

Use of this dataset must comply with the licensing terms of the original data sources.

---

## Contact
If you have any questions or suggestions, feel free to reach out:  
GitHub Profile: [@karhong-sam](https://github.com/karhong-sam)

---