import os
import argparse
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

SEC_EMAIL = os.getenv("SEC_USER_AGENT_EMAIL")

if not SEC_EMAIL:
    raise RuntimeError("Please set SEC_USER_AGENT_EMAIL in your .env file")


HEADERS = {"User-Agent": SEC_EMAIL}

def get_ticker_cik_map() -> dict:
    # download the mapping which connects betwee the tiker and the CIK (Central Index Key Code) only once per run
    # Returns a dict with Ticker in front "IXHL":293932

    url = "https://www.sec.gov/files/company_tickers.json" # Link to ticker - cik relation
    resp = requests.get(url, headers=HEADERS, timeout=20)  # request to download the file (Header important)
    resp.raise_for_status()                                # Notify when requests fail
    data = resp.json()                                     # transform the file to JSON. has info on the ticker, cik, and official name
    mapping = {}                                           # we don't need the whole information so we only map cik and ticker
    
    for _, row in data.items():
        ticker = row["ticker"].upper()
        cik = int(row["cik_str"])
        mapping[ticker] = cik
    return mapping                                         # return cik - ticker ampping



def get_company_submissions(cik: int) -> dict: 
    
    # for a CIK, get back a JSON of informaiton about general information. 
    #THis includes websidebut also -> SEC Filing number and Types of Filings (Not the entire file, just the file number)

    cik_str = f"{cik:010d}"                                         # pad to ten digits
    url = f"https://data.sec.gov/submissions/CIK{cik_str}.json"     # url of submissions from the CIK
    resp = requests.get(url, headers = HEADERS, timeout = 20)       # request SEC files
    resp.raise_for_status()
    return resp.json()                                              #returns the company's past filings in the form of JSON


def extract_filings_table(submissions_json: dict, limit: int = 100) -> pd.DataFrame:
    #Extract the filins table from the submissions into JSON
    #Remove the extra information

    recent = submissions_json['filings']['recent']                 # shows recent filings, and its information: acess number, file number, date etc. 
    df = pd.DataFrame(recent)                                      # transform into data frame
    df = df.head(limit).copy()                                     # limit to a certain number of rows

    # rename the columns to nicer names

    rename_map = {
        "filingDate": "filing_date",
        "reportDate": "report_date",
        "form": "form_type",
        "acessionNumber": "accession_number",
        "primaryFocDescription": "primary_doc_desc",
    }
    df.rename(columns=rename_map, inplace = True)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True, help="e.g. AAPL")
    parser.add_argument(
        "--out",
        default=os.path.join("data", "raw", "sec_filings.csv"),
        help="Output CSV path",
    )
    parser.add_argument("--limit", type=int, default=100, help="Max filings to keep")
    args = parser.parse_args()

    ticker = args.ticker.upper()

    # 1. map ticker -> CIK
    cik_map = get_ticker_cik_map()
    if ticker not in cik_map:
        raise ValueError(f"Ticker {ticker} not found in SEC map")

    cik = cik_map[ticker]
    print(f"[SEC] {ticker} -> CIK {cik}")

    # 2. get submissions JSON
    submissions = get_company_submissions(cik)

    # 3. DataFrame of recent filings
    df = extract_filings_table(submissions, limit=args.limit)
    df["ticker"] = ticker
    df["cik"] = cik

    # 4. save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if os.path.exists(args.out):
        df.to_csv(args.out, mode="a", header=False, index=False)
    else:
        df.to_csv(args.out, index=False)

    print(f"Saved {len(df)} rows for {ticker} to {args.out}")


if __name__ == "__main__":
    main()


