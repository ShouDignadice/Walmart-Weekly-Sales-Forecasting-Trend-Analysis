from pathlib import Path

import pandas as pd

pd.options.display.float_format = "{:.2f}".format

DATA_DIR = Path('data')
OUTPUT_DIR = Path('outputs')

def load_data(data_dir: Path) -> pd.DataFrame:

    df = pd.read_csv(data_dir / 'Walmart_Sales.csv')

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.month_name()
    df["Week_Of_Year"] = df["Date"].dt.isocalendar().week.astype(int)

    return df

def get_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:

    summary = {
        "Rows": len(df),
        "Columns": df.shape[1],
        "Stores": df["Store"].nunique(),
        "Start_Date": df["Date"].min().date(),
        "End_Date": df["Date"].max().date(),
    }

    return pd.DataFrame([summary])

def get_weekly_sales_trend(df: pd.DataFrame) -> pd.DataFrame:

    weekly_sales_trend = (
        df.groupby("Date", as_index=False)
        ["Weekly_Sales"].sum().sort_values("Date")
    )

    return weekly_sales_trend

def get_monthly_average_sales(df: pd.DataFrame) -> pd.DataFrame:

    monthly_sales = (
        df.groupby(["Month", "Month_Name"], as_index=False)
        ["Weekly_Sales"].mean().sort_values("Month")
    )

    return monthly_sales

def main() -> None:

    sales_df = load_data(DATA_DIR)
    dataset_summary = get_dataset_summary(sales_df)
    weekly_sales_trend = get_weekly_sales_trend(sales_df)
    monthly_average_sales = get_monthly_average_sales(sales_df)

    ## Testing loading data step 1 & 2
    #print(sales_df.head())
    #print(sales_df.info())

    ## Testing dataset summary step 3
    #print(dataset_summary)

    ## Testing weekly sales trend step 4
    #print(weekly_sales_trend.head(10))

    ## Testing monthly average sales step 5
    #print(monthly_average_sales)

if __name__ == "__main__":
    main()