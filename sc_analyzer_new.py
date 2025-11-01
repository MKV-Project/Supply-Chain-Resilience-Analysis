"""
Optimized Supply Chain Analysis Engine
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

@dataclass
class CompanyData:
    """Container for company information"""
    name: str
    sector: str
    data: pd.DataFrame
    ticker: str
    metrics: Dict[str, float]

class SCAnalyzer:
    """Supply Chain Analyzer with Enhanced Features"""
    
    def __init__(self):
        self._cache = {}
        self._sector_map = {
            'Semiconductors': ['TSM', 'NVDA', 'INTC', 'AMD', 'AVGO', 'ASML', 'QCOM'],
            'Automotive': ['TSLA', 'F', 'GM', 'TM', 'TATAMOTORS.NS', 'MARUTI.NS'],
            'Consumer Electronics': ['AAPL', 'SONY', 'HPQ', 'DELL', 'MSFT'],
            'Telecom_Industrial': ['CSCO', 'ERIC', 'NOK', 'ABB']
        }
        self._impact_thresholds = {
            'Critical': 60,
            'Severe': 40,
            'Moderate': 20,
            'Low': 0
        }
    
    def analyze(self, tickers: List[str], start_date: datetime, end_date: datetime, 
            risk_threshold: float = 0.3) -> Dict[str, Any]:
        """Run complete analysis pipeline"""
        try:
            tickers = [t.strip().upper() for t in tickers if t.strip()]
            if not tickers:
                raise ValueError("No valid tickers provided")

            # Fetch and process data
            companies = self._process_companies(tickers, start_date, end_date)
            if not companies:
                raise ValueError("No valid stock data collected")

            # Run analyses
            results = {
                'metadata': {
                    'tickers': tickers,
                    'period': f"{start_date} to {end_date}",
                    'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'total_companies': len(companies)
                },
                'performance': self._get_performance_data(companies),
                'risk': self._analyze_risk(companies, risk_threshold),
                'supply_chain_impact': self._analyze_supply_chain(companies),
                'sector_vulnerability': self._analyze_sectors(companies),
                'time_series_data': self.get_time_series_data(companies),  # Add this line
                'companies': [c.ticker for c in companies]
            }

            return results

        except Exception as e:
            raise Exception(f"Analysis failed: {str(e)}")

    def _process_companies(self, tickers: List[str], start_date: datetime, 
                         end_date: datetime) -> List[CompanyData]:
        """Process company data with parallel execution"""
        companies = []
        
        for ticker in tickers:
            try:
                if data := self._fetch_stock_data(ticker, start_date, end_date):
                    companies.append(data)
            except Exception as e:
                warnings.warn(f"Error processing {ticker}: {str(e)}")
                continue
                
        return companies

    @lru_cache(maxsize=100)
    def _fetch_stock_data(self, ticker: str, start_date: datetime, 
                         end_date: datetime) -> Optional[CompanyData]:
        """Fetch and process stock data with caching"""
        try:
            # Download data
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if len(data) < 30:
                return None
            
            # Calculate metrics
            data['Return'] = data['Close'].pct_change()
            data['Volatility'] = data['Return'].rolling(30, min_periods=10).std() * np.sqrt(252)
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            if data.isnull().any().any():
                return None
            
            # Calculate key metrics
            returns = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
            volatility = float(data['Volatility'].mean() * 100)
            drawdown = ((data['Close'].min() / data['Close'].max()) - 1) * 100
            
            return CompanyData(
                name=stock.info.get('longName', ticker),
                sector=self._determine_sector(ticker, stock),
                data=data,
                ticker=ticker,
                metrics={
                    'return': round(returns, 2),
                    'volatility': round(volatility, 2),
                    'drawdown': round(drawdown, 2)
                }
            )
            
        except Exception:
            return None

    def _determine_sector(self, ticker: str, stock: yf.Ticker) -> str:
        """Determine company sector"""
        # Check predefined sectors
        for sector, tickers in self._sector_map.items():
            if ticker in tickers:
                return sector
        
        # Use Yahoo Finance data
        try:
            sector = stock.info.get('sector', '').lower()
            industry = stock.info.get('industry', '').lower()
            
            if any(word in sector + industry 
                   for word in ['semiconductor', 'chip']):
                return 'Semiconductors'
            elif any(word in sector + industry 
                    for word in ['auto', 'vehicle']):
                return 'Automotive'
            elif any(word in sector + industry 
                    for word in ['electronic', 'computer']):
                return 'Consumer Electronics'
            elif any(word in sector + industry 
                    for word in ['telecom', 'industrial']):
                return 'Telecom_Industrial'
        except:
            pass
        
        return 'Other'

    def _get_performance_data(self, companies: List[CompanyData]) -> Dict[str, Dict]:
        """Extract performance data"""
        return {
            company.ticker: {
                'name': company.name,
                'sector': company.sector,
                **company.metrics
            }
            for company in companies
        }

    def _analyze_risk(self, companies: List[CompanyData], 
                     threshold: float) -> Dict[str, Dict]:
        """Analyze company risks"""
        if len(companies) < 3:
            return {}
        
        # Prepare feature matrix
        features = []
        valid_companies = []
        
        for company in companies:
            vol = company.metrics['volatility']
            dd = abs(company.metrics['drawdown'])
            
            if not np.isnan([vol, dd]).any():
                features.append([vol, dd])
                valid_companies.append(company)
        
        if len(features) < 3:
            return {}
        
        # Run isolation forest
        detector = IsolationForest(contamination=min(threshold, 0.5), 
                                 random_state=42)
        scores = detector.fit_predict(np.array(features))
        
        return {
            company.ticker: {
                'name': company.name,
                'sector': company.sector,
                'score': 'High' if score == -1 else 'Low'
            }
            for company, score in zip(valid_companies, scores)
        }

    def _analyze_supply_chain(self, companies: List[CompanyData]) -> List[Dict]:
        """Analyze supply chain impacts"""
        return [
            {
                'Company': company.name,
                'Ticker': company.ticker,
                'Sector': company.sector,
                'Semiconductor_Dependency': self._get_dependency(company.sector),
                'Financial_Impact_pct': abs(company.metrics['drawdown']),
                'Impact_Severity': self._get_severity(abs(company.metrics['drawdown'])),
                'Estimated_Recovery_Months': self._estimate_recovery(
                    company.metrics['return'],
                    company.metrics['volatility']
                ),
                'Supply_Chain_Resilience': self._calculate_resilience(
                    company.metrics['return'],
                    company.metrics['drawdown']
                ),
                'Strategic_Recommendation': self._get_recommendation(
                    company.sector,
                    abs(company.metrics['drawdown'])
                )
            }
            for company in companies
        ]

    def _analyze_sectors(self, companies: List[CompanyData]) -> List[Dict]:
        """Analyze sector vulnerabilities"""
        sector_data = {}
        
        for company in companies:
            if company.sector not in sector_data:
                sector_data[company.sector] = {
                    'impacts': [],
                    'returns': [],
                    'companies': 0,
                    'critical': 0,
                    'severe': 0
                }
            
            data = sector_data[company.sector]
            impact = abs(company.metrics['drawdown'])
            
            data['impacts'].append(impact)
            data['returns'].append(company.metrics['return'])
            data['companies'] += 1
            
            if impact >= self._impact_thresholds['Critical']:
                data['critical'] += 1
            elif impact >= self._impact_thresholds['Severe']:
                data['severe'] += 1
        
        return [
            {
                'Sector': sector,
                'Companies_Analyzed': data['companies'],
                'Avg_Financial_Impact_pct': round(np.mean(data['impacts']), 1),
                'Avg_Return_Pct': round(np.mean(data['returns']), 1),
                'Critical_Impact_Companies': data['critical'],
                'Severe_Impact_Companies': data['severe'],
                'Supply_Chain_Risk_Level': self._get_risk_level(np.mean(data['impacts']))
            }
            for sector, data in sector_data.items()
        ]

    def _get_dependency(self, sector: str) -> str:
        """Get sector dependency level"""
        dependencies = {
            'Semiconductors': 'Supplier',
            'Automotive': 'Critical (50-150 chips/vehicle)',
            'Consumer Electronics': 'High (Core component)',
            'Telecom_Industrial': 'Medium (Infrastructure)',
            'Other': 'Low'
        }
        return dependencies.get(sector, 'Low')

    def _get_severity(self, impact: float) -> str:
        """Get impact severity level"""
        for severity, threshold in self._impact_thresholds.items():
            if impact >= threshold:
                return severity
        return 'Low'

    def _estimate_recovery(self, returns: float, volatility: float) -> str:
        """Estimate recovery time"""
        if returns > 50: return '3-6 months'
        if returns > 20: return '6-12 months'
        if returns > 0: return '12-18 months'
        return '18+ months'

    def _calculate_resilience(self, returns: float, drawdown: float) -> float:
        """Calculate resilience score"""
        base = 50
        return_boost = max(0, returns * 0.5)
        drawdown_penalty = min(40, abs(drawdown) * 0.8)
        return round(max(0, min(100, base + return_boost - drawdown_penalty)), 1)

    def _get_recommendation(self, sector: str, impact: float) -> str:
        """Get strategic recommendation"""
        recommendations = {
            'Automotive': {
                'high': 'Immediate supplier diversification and inventory buildup',
                'medium': 'Diversify suppliers and increase safety stock',
                'low': 'Strengthen existing supplier relationships'
            },
            'Consumer Electronics': {
                'high': 'Increase inventory buffers and dual-source components',
                'medium': 'Optimize component sourcing and increase flexibility',
                'low': 'Maintain current sourcing strategy with monitoring'
            },
            'Semiconductors': {
                'high': 'Expand production capacity and geographic diversification',
                'low': 'Invest in R&D and process optimization'
            }
        }
        
        sector_recs = recommendations.get(sector, {
            'high': 'Review and diversify supply chain dependencies',
            'low': 'Monitor supply chain risks regularly'
        })
        
        if impact >= 45:
            return sector_recs.get('high', sector_recs['low'])
        elif impact >= 25:
            return sector_recs.get('medium', sector_recs['low'])
        return sector_recs['low']

    def _get_risk_level(self, impact: float) -> str:
        """Get sector risk level"""
        if impact >= 40: return 'Extreme'
        if impact >= 25: return 'High'
        if impact >= 15: return 'Medium'
        return 'Low'
    
    def get_time_series_data(self, companies: List[CompanyData], max_companies: int = 6) -> pd.DataFrame:
        """Get actual normalized price data for time series visualization"""
        try:
            time_series_data = []
            
            # Select diverse companies
            selected_companies = self._select_diverse_companies(companies, max_companies)
            
            for company in selected_companies:
                if hasattr(company, 'data') and not company.data.empty:
                    # Use actual historical data
                    data = company.data.reset_index()
                    
                    # Normalize prices to base 100
                    base_price = data['Close'].iloc[0]
                    data['Normalized_Price'] = (data['Close'] / base_price) * 100
                    
                    # Sample data points for cleaner visualization (max 100 points)
                    if len(data) > 100:
                        data = data.iloc[::len(data)//100]
                    
                    for _, row in data.iterrows():
                        time_series_data.append({
                            'Date': row['Date'],
                            'Normalized_Price': round(row['Normalized_Price'], 2),
                            'Company': company.name,
                            'Ticker': company.ticker,
                            'Sector': company.sector
                        })
            
            return pd.DataFrame(time_series_data)
            
        except Exception as e:
            print(f"Error generating time series data: {str(e)}")
            return pd.DataFrame()

    def _select_diverse_companies(self, companies: List[CompanyData], max_companies: int) -> List[CompanyData]:
        """Select diverse companies for visualization"""
        selected = []
        sectors_covered = set()
        
        # First pass: one per sector
        for company in companies:
            if company.sector not in sectors_covered:
                selected.append(company)
                sectors_covered.add(company.sector)
        
        # Second pass: fill remaining slots
        if len(selected) < max_companies:
            remaining = [c for c in companies if c not in selected]
            selected.extend(remaining[:max_companies - len(selected)])
        
        return selected[:max_companies]