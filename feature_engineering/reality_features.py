import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Dict, List
from loguru import logger

class RealityFeatureEngineer:
    def __init__(self):
        self.dilution_keywords = [
            "warrant", "convertible", "offering", "atm", "pipe",
            "dilution", "authorized shares", "issuance", "equity",
            "reverse split", "stock split", "placement"
        ]
        
        self.positive_catalyst_keywords = [
            "acquisition", "merger", "partnership", "collaboration",
            "contract", "approval", "fda", "launch", "expansion",
            "profit", "growth", "revenue increase", "earnings beat"
        ]
        
        self.risk_keywords = [
            "substantial doubt", "going concern", "risk", "uncertain",
            "may not", "could not", "adverse", "loss", "failure",
            "challenge", "difficulty", "volatility"
        ]
    
    def extract_dilution_risk(self, filing_text: str) -> Dict:
        """Extract dilution risk indicators from filing text"""
        text_lower = filing_text.lower()
        
        features = {}
        
        # Check for dilution keywords
        for keyword in self.dilution_keywords:
            count = len(re.findall(rf"\b{keyword}\b", text_lower))
            features[f"dilution_{keyword}_count"] = count
        
        # Check for specific patterns
        patterns = {
            "atm_program": r"at-the-market|atm\s+program",
            "pipe_financing": r"pipe|private investment",
            "warrant_coverage": r"warrant coverage",
            "reverse_split": r"reverse\s+split|1-for-\d+",
            "share_increase": r"increase.*authorized shares|authorized.*increase"
        }
        
        for feature_name, pattern in patterns.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            features[f"has_{feature_name}"] = 1 if matches else 0
        
        # Overall dilution risk score
        total_dilution_mentions = sum([v for k, v in features.items() if "count" in k])
        features["total_dilution_mentions"] = total_dilution_mentions
        features["dilution_risk_score"] = min(total_dilution_mentions / 10, 1.0)
        
        return features
    
    def calculate_runway(self, financial_data: pd.DataFrame) -> Dict:
        """Calculate cash runway based on financial data"""
        features = {}
        
        if financial_data.empty:
            return features
        
        try:
            # Extract key metrics (assuming specific column names)
            if 'cash' in financial_data.columns:
                latest_cash = financial_data['cash'].iloc[-1]
                
                if 'operating_cash_flow' in financial_data.columns:
                    # Negative OCF means cash burn
                    latest_ocf = financial_data['operating_cash_flow'].iloc[-1]
                    
                    if latest_ocf < 0:  # Company is burning cash
                        monthly_burn = abs(latest_ocf) / 3  # Quarterly to monthly
                        if monthly_burn > 0:
                            runway_months = latest_cash / monthly_burn
                            features["cash_runway_months"] = runway_months
                            features["is_burning_cash"] = 1
                            
                            # Risk flags
                            if runway_months < 3:
                                features["runway_risk"] = "critical"
                            elif runway_months < 6:
                                features["runway_risk"] = "high"
                            elif runway_months < 12:
                                features["runway_risk"] = "medium"
                            else:
                                features["runway_risk"] = "low"
            
            # Going concern mention
            if 'going_concern' in financial_data.columns:
                features["going_concern_mention"] = int(financial_data['going_concern'].iloc[-1])
        
        except Exception as e:
            logger.error(f"Error calculating runway: {e}")
        
        return features
    
    def extract_risk_tone_delta(self, current_filing: str, 
                               previous_filing: str) -> Dict:
        """Calculate change in risk tone between filings"""
        features = {}
        
        if not current_filing or not previous_filing:
            return features
        
        # Count risk keywords in current filing
        current_risk_count = 0
        previous_risk_count = 0
        
        for keyword in self.risk_keywords:
            current_matches = len(re.findall(rf"\b{re.escape(keyword)}\b", 
                                           current_filing.lower()))
            previous_matches = len(re.findall(rf"\b{re.escape(keyword)}\b", 
                                            previous_filing.lower()))
            
            current_risk_count += current_matches
            previous_risk_count += previous_matches
        
        # Calculate deltas
        features["risk_count_current"] = current_risk_count
        features["risk_count_previous"] = previous_risk_count
        features["risk_count_delta"] = current_risk_count - previous_risk_count
        features["risk_count_pct_change"] = (
            (current_risk_count - previous_risk_count) / 
            max(previous_risk_count, 1)
        )
        
        # Risk section length
        current_length = len(current_filing)
        previous_length = len(previous_filing)
        features["risk_density_current"] = current_risk_count / max(current_length, 1)
        features["risk_density_previous"] = previous_risk_count / max(previous_length, 1)
        features["risk_density_delta"] = (
            features["risk_density_current"] - features["risk_density_previous"]
        )
        
        return features
    
    def extract_positive_catalysts(self, filing_text: str) -> Dict:
        """Extract positive catalysts from filing text"""
        text_lower = filing_text.lower()
        
        features = {}
        
        # Check for positive catalyst keywords
        for keyword in self.positive_catalyst_keywords:
            count = len(re.findall(rf"\b{keyword}\b", text_lower))
            features[f"catalyst_{keyword}_count"] = count
        
        # Specific patterns for common catalysts
        catalyst_patterns = {
            "new_contract": r"contract.*\$[\d,]+|won.*contract",
            "fda_approval": r"fda.*approv|approv.*fda",
            "merger_announced": r"merger.*agreement|acquire.*company",
            "partnership_announced": r"partner.*with|strategic.*partner",
            "expansion_announced": r"expand.*to|new.*facility|open.*location"
        }
        
        for catalyst_name, pattern in catalyst_patterns.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            features[f"has_{catalyst_name}"] = 1 if matches else 0
        
        # Overall catalyst score
        total_catalyst_mentions = sum([v for k, v in features.items() if "count" in k])
        features["total_catalyst_mentions"] = total_catalyst_mentions
        features["catalyst_score"] = min(total_catalyst_mentions / 5, 1.0)
        
        return features
    
    def extract_all_reality_features(self, 
                                   sec_filings: pd.DataFrame,
                                   financial_data: pd.DataFrame,
                                   previous_filing_text: str = None) -> Dict:
        """Extract all reality tower features"""
        features = {}
        
        if sec_filings.empty:
            return features
        
        # Combine all filing text
        filing_texts = []
        if 'text' in sec_filings.columns:
            filing_texts = sec_filings['text'].dropna().tolist()
        
        combined_text = " ".join(filing_texts) if filing_texts else ""
        
        # Dilution Risk
        dilution_features = self.extract_dilution_risk(combined_text)
        features.update(dilution_features)
        
        # Runway Calculation
        runway_features = self.calculate_runway(financial_data)
        features.update(runway_features)
        
        # Risk Tone Delta (if previous filing available)
        if previous_filing_text and combined_text:
            risk_delta_features = self.extract_risk_tone_delta(
                combined_text, previous_filing_text
            )
            features.update(risk_delta_features)
        
        # Positive Catalysts
        catalyst_features = self.extract_positive_catalysts(combined_text)
        features.update(catalyst_features)
        
        # Calculate overall reality score
        reality_score = 0.5  # Neutral baseline
        
        # Adjust based on features
        if "dilution_risk_score" in features:
            reality_score -= features["dilution_risk_score"] * 0.3
        
        if "catalyst_score" in features:
            reality_score += features["catalyst_score"] * 0.3
        
        if "runway_risk" in features:
            if features["runway_risk"] == "critical":
                reality_score -= 0.3
            elif features["runway_risk"] == "high":
                reality_score -= 0.2
            elif features["runway_risk"] == "medium":
                reality_score -= 0.1
        
        if "risk_density_delta" in features:
            if features["risk_density_delta"] > 0.1:
                reality_score -= 0.1
            elif features["risk_density_delta"] < -0.1:
                reality_score += 0.1
        
        features["reality_score"] = max(0, min(1, reality_score))
        
        return features