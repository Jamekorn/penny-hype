import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
from datetime import datetime
from typing import List, Dict
from loguru import logger

class SECEdgarCollector:
    BASE_URL = "https://www.sec.gov/Archives/edgar/data"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "University Project (email@university.edu)"
        })
    
    def get_cik_from_ticker(self, ticker: str) -> str:
        """Get CIK number from ticker symbol"""
        try:
            # Use SEC company tickers JSON
            url = "https://www.sec.gov/files/company_tickers.json"
            response = self.session.get(url)
            data = response.json()
            
            for company in data.values():
                if company["ticker"] == ticker.upper():
                    return str(company["cik_str"]).zfill(10)
        except Exception as e:
            logger.error(f"Error getting CIK for {ticker}: {e}")
        
        return None
    
    def get_filings(self, ticker: str, filing_type: str = "10-K", 
                   limit: int = 5) -> pd.DataFrame:
        """Get recent SEC filings"""
        cik = self.get_cik_from_ticker(ticker)
        if not cik:
            return pd.DataFrame()
        
        filings = []
        try:
            # Get company facts
            url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
            response = self.session.get(url)
            if response.status_code == 200:
                data = response.json()
                
                # Extract recent filings
                for fact in data.get("facts", {}).get("us-gaap", {}).values():
                    for unit in fact.get("units", {}).values():
                        for item in unit[:limit]:
                            if "accn" in item and "filed" in item:
                                filings.append({
                                    "ticker": ticker,
                                    "cik": cik,
                                    "accession_number": item.get("accn"),
                                    "filing_date": item.get("filed"),
                                    "form": filing_type,
                                    "frame": item.get("frame", ""),
                                    "value": item.get("val"),
                                    "description": fact.get("label", "")
                                })
        
        except Exception as e:
            logger.error(f"Error getting filings for {ticker}: {e}")
        
        return pd.DataFrame(filings)
    
    def extract_risk_factors(self, filing_text: str) -> Dict:
        """Extract risk factors from filing text"""
        risks = {}
        
        # Look for risk factor section
        risk_patterns = [
            r"Item\s*1A\.?\s*Risk\s*Factors(.*?)Item\s*1B",
            r"RISK\s*FACTORS(.*?)ITEM",
            r"risk\s*factors(.*?)(?:ITEM|PART)"
        ]
        
        for pattern in risk_patterns:
            match = re.search(pattern, filing_text, re.IGNORECASE | re.DOTALL)
            if match:
                risk_text = match.group(1)
                
                # Count risk-related keywords
                risk_keywords = {
                    "doubt": len(re.findall(r"doubt", risk_text, re.IGNORECASE)),
                    "risk": len(re.findall(r"\brisk\b", risk_text, re.IGNORECASE)),
                    "uncertain": len(re.findall(r"uncertain", risk_text, re.IGNORECASE)),
                    "may": len(re.findall(r"\bmay\b", risk_text, re.IGNORECASE)),
                    "could": len(re.findall(r"\bcould\b", risk_text, re.IGNORECASE)),
                    "adverse": len(re.findall(r"adverse", risk_text, re.IGNORECASE)),
                    "loss": len(re.findall(r"\bloss\b", risk_text, re.IGNORECASE)),
                    "fail": len(re.findall(r"\bfail\b", risk_text, re.IGNORECASE))
                }
                
                risks["risk_text"] = risk_text[:5000]  # First 5000 chars
                risks["keyword_counts"] = risk_keywords
                risks["total_risk_words"] = sum(risk_keywords.values())
                break
        
        return risks
    