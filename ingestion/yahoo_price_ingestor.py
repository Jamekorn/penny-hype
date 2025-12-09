import argparse
import os
from datetime import datetime
import pandas as pd
import yfinance as yf

def fetch_price_history(symbol: str, start: str, end: str):
    #fetch daily OHLVC (summary) data for sy,bol between start and end. (YYYY-MM-DD)

    data = yf.download(symbol, start = start, end = end, progress = False)
    data.reset_index(inplace = True)                    #move Data from index to column
    data.rename(columns = str.lower, inplace = True)    #rename to all lower case
    data['symbol'] = symbol.upper()                     #stock tickers are upper case. So change to upper
    

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required = 'True', help = "Ticker symbol, e.g. PCSA")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--out",
        default=os.path.join("data", "raw", "prices.csv"),
        help="Output CSV path",
    )

    args = parser.parse_args()

    df = fetch_price_history(args.symbol, args.start, args.end)

    # make sure folder exists
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # append or create new
    if os.path.exists(args.out):
        df.to_csv(args.out, mode="a", header=False, index=False)
    else:
        df.to_csv(args.out, index=False)

    print(f"Saved {len(df)} rows for {args.symbol} to {args.out}")


if __name__ == "__main__":
    main()

    
