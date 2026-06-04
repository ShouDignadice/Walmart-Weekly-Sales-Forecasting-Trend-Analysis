from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

pd.options.display.float_format = "{:,.2f}".format

DATA_DIR = Path('data')
OUTPUT_DIR = Path('outputs')
PLOTS_DIR = OUTPUT_DIR / 'plots'

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

def get_holiday_sales_comparison(df: pd.DataFrame) -> pd.DataFrame:

    holiday_sales = (
        df.groupby("Holiday_Flag", as_index=False)
        .agg(
            Average_Weekly_Sales=("Weekly_Sales", "mean"),
            Total_Weekly_Sales=("Weekly_Sales", "sum"),
            Number_Of_Records=("Weekly_Sales", "count"),
        )
        .replace({"Holiday_Flag": {0: "Non-Holiday", 1: "Holiday"}})
    )

    return holiday_sales

def get_top_sales_weeks(weekly_sales: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:

    top_sales_weeks = (
        weekly_sales.sort_values("Weekly_Sales", ascending=False).head(top_n)
    )

    return top_sales_weeks

def get_store_average_sales(df: pd.DataFrame) -> pd.DataFrame:

    store_average_sales = (
        df.groupby("Store", as_index=False)
        ["Weekly_Sales"].mean().sort_values("Weekly_Sales", ascending=False)
    )

    return store_average_sales

def plot_weekly_sales_trend(weekly_sales_trend: pd.DataFrame) -> None:

    plt.figure(figsize=(12, 8))

    plt.plot(
        weekly_sales_trend["Date"],
        weekly_sales_trend["Weekly_Sales"]
    )

    plt.title("Total Weekly Sales Trend Across All Stores")
    plt.xlabel("Date")
    plt.ylabel("Total Weekly Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(PLOTS_DIR / 'weekly_sales_trend.png')
    plt.close()

def plot_monthly_average_sales(monthly_sales: pd.DataFrame) -> None:

    plt.figure(figsize=(12, 8))

    plt.bar(
        monthly_sales["Month_Name"],
        monthly_sales["Weekly_Sales"]
    )

    plt.title("Average Weekly Sales by Month")
    plt.xlabel("Month")
    plt.ylabel("Average Weekly Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(PLOTS_DIR / 'monthly_average_sales.png')
    plt.close()

def plot_holiday_sales_comparison(holiday_sales: pd.DataFrame) -> None:

    plt.figure(figsize=(12, 8))

    plt.bar(
        holiday_sales["Holiday_Flag"],
        holiday_sales["Average_Weekly_Sales"]
    )

    plt.title("Average Weekly Sales: Holiday vs Non-Holiday")
    plt.xlabel("Week Type")
    plt.ylabel("Average Weekly Sales")
    plt.tight_layout()

    plt.savefig(PLOTS_DIR / "holiday_vs_nonholiday_sales.png")
    plt.close()

def plot_top_sales_weeks(top_sales_weeks: pd.DataFrame) -> None:

    chart_data = top_sales_weeks.copy()
    chart_data["Date"] = chart_data["Date"].astype(str)

    plt.figure(figsize=(12, 8))

    plt.bar(
        chart_data["Date"],
        chart_data["Weekly_Sales"]
    )

    plt.title("Top 10 Highest Sales Weeks")
    plt.xlabel("Date")
    plt.ylabel("Total Weekly Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(PLOTS_DIR / "top_10_sales_weeks.png")
    plt.close()

def plot_store_average_sales(store_average_sales: pd.DataFrame) -> None:
    top_stores = store_average_sales.head(10)

    plt.figure(figsize=(10, 6))

    plt.bar(
        top_stores["Store"].astype(str),
        top_stores["Weekly_Sales"]
    )

    plt.title("Top 10 Stores by Average Weekly Sales")
    plt.xlabel("Store")
    plt.ylabel("Average Weekly Sales")
    plt.tight_layout()

    plt.savefig(PLOTS_DIR / "top_10_store_average_sales.png")
    plt.close()

def main() -> None:



    sales_df = load_data(DATA_DIR)
    dataset_summary = get_dataset_summary(sales_df)
    weekly_sales_trend = get_weekly_sales_trend(sales_df)
    monthly_average_sales = get_monthly_average_sales(sales_df)
    holiday_sales = get_holiday_sales_comparison(sales_df)
    top_sales_weeks = get_top_sales_weeks(weekly_sales_trend, top_n=10)
    store_average_sales = get_store_average_sales(sales_df)

    ## Testing loading data step 1 & 2
    #print(sales_df.head())
    #print(sales_df.info())

    ## Testing dataset summary step 3
    #print(dataset_summary)

    ## Testing weekly sales trend step 4
    #print(weekly_sales_trend.head(10))
    plot_weekly_sales_trend(weekly_sales_trend)

    ## Testing monthly average sales step 5
    #print(monthly_average_sales)
    plot_monthly_average_sales(monthly_average_sales)

    ## Testing holiday sales comparison step 6
    #print(holiday_sales)
    plot_holiday_sales_comparison(holiday_sales)

    ## Testing get top sales weeks step 7
    #print(top_sales_weeks)
    plot_top_sales_weeks(top_sales_weeks)

    ## Testing store average sales step 8
    #hprint(store_average_sales.head(10).to_string(index=False))
    plot_store_average_sales(store_average_sales)


if __name__ == "__main__":
    main()