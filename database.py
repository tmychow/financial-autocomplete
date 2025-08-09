"""
Database module for Finance RL Autocomplete Training
Combines database setup, Tiingo data loading, and sample data for Colab compatibility
"""

import sqlite3
import asyncio
import aiosqlite
from contextlib import asynccontextmanager
from typing import List, Dict, Optional
import os
import httpx
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_PATH = "/tmp/financial_data.db"  # Use temp directory for Colab

# ============== Core Database Functions ==============

async def init_database():
    """Initialize the SQLite database with required tables"""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute('''
            CREATE TABLE IF NOT EXISTS tickers (
                ticker TEXT PRIMARY KEY,
                company_name TEXT,
                sector TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        await db.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                metric_name TEXT PRIMARY KEY,
                description TEXT,
                unit TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        await db.execute('''
            CREATE TABLE IF NOT EXISTS financial_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                period TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ticker) REFERENCES tickers(ticker),
                FOREIGN KEY (metric_name) REFERENCES metrics(metric_name),
                UNIQUE(ticker, metric_name, period)
            )
        ''')
        
        # Create indexes for performance
        await db.execute('CREATE INDEX IF NOT EXISTS idx_financial_data_ticker ON financial_data(ticker)')
        await db.execute('CREATE INDEX IF NOT EXISTS idx_financial_data_metric ON financial_data(metric_name)')
        await db.execute('CREATE INDEX IF NOT EXISTS idx_financial_data_period ON financial_data(period)')
        
        await db.commit()

@asynccontextmanager
async def get_db():
    """Get database connection context manager"""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        yield db

# ============== Insert Functions ==============

async def insert_ticker(ticker: str, company_name: str, sector: str = None):
    async with get_db() as db:
        await db.execute(
            "INSERT OR REPLACE INTO tickers (ticker, company_name, sector) VALUES (?, ?, ?)",
            (ticker, company_name, sector)
        )
        await db.commit()

async def insert_metric(metric_name: str, description: str, unit: str):
    async with get_db() as db:
        await db.execute(
            "INSERT OR REPLACE INTO metrics (metric_name, description, unit) VALUES (?, ?, ?)",
            (metric_name, description, unit)
        )
        await db.commit()

async def insert_financial_data(ticker: str, metric_name: str, period: str, value: float, unit: str = None):
    async with get_db() as db:
        await db.execute(
            "INSERT OR REPLACE INTO financial_data (ticker, metric_name, period, value, unit) VALUES (?, ?, ?, ?, ?)",
            (ticker, metric_name, period, value, unit)
        )
        await db.commit()

# ============== Query Functions ==============

async def get_all_tickers() -> List[str]:
    async with get_db() as db:
        async with db.execute("SELECT ticker FROM tickers ORDER BY ticker") as cursor:
            rows = await cursor.fetchall()
            return [row["ticker"] for row in rows]

async def get_tickers_with_names() -> List[Dict[str, str]]:
    async with get_db() as db:
        async with db.execute("SELECT ticker, company_name FROM tickers ORDER BY ticker") as cursor:
            rows = await cursor.fetchall()
            return [{"ticker": row["ticker"], "company_name": row["company_name"]} for row in rows]

async def get_all_metrics() -> List[Dict[str, str]]:
    async with get_db() as db:
        async with db.execute("SELECT metric_name, description, unit FROM metrics ORDER BY metric_name") as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

async def get_financial_value(ticker: str, metric_name: str, period: str) -> Optional[Dict[str, any]]:
    async with get_db() as db:
        async with db.execute(
            "SELECT value, unit FROM financial_data WHERE ticker = ? AND metric_name = ? AND period = ?",
            (ticker, metric_name, period)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return {"value": row["value"], "unit": row["unit"]}
            return None

async def get_available_periods(ticker: str, metric_name: str) -> List[str]:
    async with get_db() as db:
        async with db.execute(
            "SELECT DISTINCT period FROM financial_data WHERE ticker = ? AND metric_name = ? ORDER BY period DESC",
            (ticker, metric_name)
        ) as cursor:
            rows = await cursor.fetchall()
            return [row["period"] for row in rows]

def parse_period(period: str) -> tuple:
    """Parse a period string like '2024Q1' or '2024FY' into (year, quarter)"""
    if 'FY' in period:
        year = int(period.replace('FY', ''))
        return (year, 5)  # FY comes after Q4 for sorting
    elif 'Q' in period:
        parts = period.split('Q')
        year = int(parts[0])
        quarter = int(parts[1])
        return (year, quarter)
    return (0, 0)

async def get_latest_period(ticker: str, metric_name: str) -> Optional[str]:
    """Get the most recent period for a given ticker and metric"""
    periods = await get_available_periods(ticker, metric_name)
    if not periods:
        return None
    
    # Sort periods properly (2024Q2 > 2024Q1, 2024FY > 2024Q4)
    sorted_periods = sorted(periods, key=parse_period, reverse=True)
    return sorted_periods[0] if sorted_periods else None

async def get_tickers_with_data() -> List[str]:
    """Get only tickers that actually have financial data in the database"""
    async with get_db() as db:
        async with db.execute("""
            SELECT DISTINCT ticker 
            FROM financial_data 
            ORDER BY ticker
        """) as cursor:
            rows = await cursor.fetchall()
            return [row["ticker"] for row in rows]


# ============== Tiingo Data Loader (Optional) ==============

class TiingoDataLoader:
    """Optional Tiingo data loader for when API key is available"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("TIINGO_API_KEY")
        self.base_url = "https://api.tiingo.com/tiingo"
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Token {self.api_key}' if self.api_key else ''
        }
    
    async def fetch_daily_data(self, ticker: str) -> List[Dict]:
        """Fetch daily data from Tiingo for market cap, PE ratio, etc."""
        url = f"{self.base_url}/fundamentals/{ticker}/daily"
        params = {'token': self.api_key}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, headers={'Content-Type': 'application/json'})
            if response.status_code == 200:
                data = response.json()
                print(f"Successfully fetched daily data for {ticker}")
                return data
            else:
                print(f"Error fetching daily data for {ticker}: {response.status_code}")
                return []
    
    async def process_daily_data(self, ticker: str, daily_data: List[Dict], statement_data: List[Dict] = None):
        """Process daily data, aligning with statement periods"""
        if not daily_data or not isinstance(daily_data, list):
            print(f"No daily data to process for {ticker}")
            return
        
        print(f"Processing {len(daily_data)} daily entries for {ticker}")
        print(f"  First entry date: {daily_data[0].get('date', '')[:10] if daily_data else 'N/A'}")
        print(f"  Last entry date: {daily_data[-1].get('date', '')[:10] if daily_data else 'N/A'}")
        
        # Create a date index for fast lookup
        daily_by_date = {}
        for entry in daily_data:
            date_str = entry.get('date', '')[:10]
            if date_str:
                daily_by_date[date_str] = entry
        
        # Get periods from statement data if available
        statement_periods = {}
        if statement_data:
            for stmt in statement_data:
                year = stmt.get('year')
                quarter = stmt.get('quarter')
                date_str = stmt.get('date', '')[:10]
                
                if year and quarter is not None and date_str:
                    if quarter == 0:
                        period = f"{year}FY"
                    else:
                        period = f"{year}Q{quarter}"
                    statement_periods[period] = date_str
        
        # Process daily metrics for each statement period
        daily_metrics = ['marketCap', 'peRatio', 'pbRatio', 'trailingPEG1Y']
        
        for period, statement_date in statement_periods.items():
            # Look for daily data starting from statement date and going back up to 7 days
            metrics_found = {}
            
            # Try to find data on or near the statement date
            from datetime import datetime, timedelta
            try:
                base_date = datetime.strptime(statement_date, '%Y-%m-%d')
                
                # Look back up to 7 days for each metric
                for days_back in range(8):  # 0 to 7 days
                    check_date = (base_date - timedelta(days=days_back)).strftime('%Y-%m-%d')
                    
                    if check_date in daily_by_date:
                        entry = daily_by_date[check_date]
                        
                        # Collect any metrics we haven't found yet
                        for metric in daily_metrics:
                            if metric not in metrics_found and entry.get(metric) is not None:
                                metrics_found[metric] = entry.get(metric)
                        
                        # If we found all metrics, stop looking
                        if len(metrics_found) == len(daily_metrics):
                            break
            except ValueError:
                print(f"Invalid date format: {statement_date}")
                continue
            
            # Debug: check what we found vs what we expected
            if len(metrics_found) < len(daily_metrics):
                missing = [m for m in daily_metrics if m not in metrics_found]
                print(f"  Period {period}: Could not find daily data for {missing} within 7 days of {statement_date}")
            
            # Insert the metrics we found
            for metric_name, value in metrics_found.items():
                try:
                    unit = self._determine_unit(metric_name)
                    
                    # Convert large USD values to billions (unless already in billions)
                    if unit == 'USD' and abs(value) > 1e6:
                        value = value / 1e9
                        unit = 'USD_billions'
                    elif unit == 'USD_billions' and abs(value) > 1e6:
                        # Already should be in billions, but if raw value is too large, convert
                        value = value / 1e9
                    
                    await insert_financial_data(
                        ticker=ticker,
                        metric_name=metric_name,
                        period=period,
                        value=float(value),
                        unit=unit
                    )
                    print(f"Inserted daily {metric_name}={value:.2f} {unit} for {period}")
                except Exception as e:
                    print(f"Error inserting daily {metric_name}: {e}")
    
    async def load_ticker_data(self, ticker: str, company_name: str, sector: str = None):
        """Load data for a single ticker from Tiingo API"""
        if not self.api_key:
            print(f"No Tiingo API key found, skipping {ticker}")
            return False
        
        print(f"Loading data for {ticker} from Tiingo...")
        await insert_ticker(ticker, company_name, sector)
        
        # Fetch fundamental statements data
        url = f"{self.base_url}/fundamentals/{ticker}/statements"
        params = {'token': self.api_key}
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, headers={'Content-Type': 'application/json'})
                if response.status_code == 200:
                    statement_data = response.json()
                    await self._process_financial_data(ticker, statement_data)
                    
                    # Also fetch daily data for market cap, PE ratio, etc.
                    daily_data = await self.fetch_daily_data(ticker)
                    if daily_data:
                        await self.process_daily_data(ticker, daily_data, statement_data)
                    
                    return True
                else:
                    print(f"Error fetching data for {ticker}: {response.status_code}")
                    return False
        except Exception as e:
            print(f"Error loading {ticker}: {e}")
            return False
    
    async def _process_financial_data(self, ticker: str, data: List[Dict]):
        """Process financial data from Tiingo API response"""
        if not data or not isinstance(data, list):
            return
        
        print(f"Processing {len(data)} statement objects for {ticker}")
        
        for i, statement_obj in enumerate(data):
            if not isinstance(statement_obj, dict):
                continue
            
            # Extract period information
            date_str = statement_obj.get('date', '')[:10]
            year = statement_obj.get('year')
            quarter = statement_obj.get('quarter')
            
            if quarter == 0:
                period = f"{year}FY"
            else:
                period = f"{year}Q{quarter}"
            
            print(f"Processing {period} (date: {date_str})")
            
            # Get the nested statement data
            statements = statement_obj.get('statementData', {})
            if not statements:
                print(f"No statementData found for {period}")
                continue
            
            # Extract all sub-statements (they are lists)
            income_statements = statements.get('incomeStatement', [])
            balance_sheets = statements.get('balanceSheet', [])
            cash_flows = statements.get('cashFlow', [])
            overview_data = statements.get('overview', [])
            
            inserted_count = 0
            
            # Convert dataCode/value pairs to a dictionary
            def convert_to_dict(statement_list):
                result = {}
                for item in statement_list:
                    if isinstance(item, dict) and 'dataCode' in item and 'value' in item:
                        result[item['dataCode']] = item['value']
                return result
            
            # Process income statement metrics with exact Tiingo dataCodes
            if income_statements:
                income_data = convert_to_dict(income_statements)
                
                income_metrics = ['revenue', 'epsDil', 'netinc', 'eps', 'ebitda', 'rnd', 'sga', 
                                  'opinc', 'grossProfit', 'opex', 'ebt', 'costRev', 'ebit', 
                                  'netIncComStock', 'intexp', 'consolidatedIncome', 'shareswa', 
                                  'nonControllingInterests']
                
                for metric in income_metrics:
                    if metric in income_data and income_data[metric] is not None:
                        value = income_data[metric]
                        unit = self._determine_unit(metric)
                        
                        # Convert large USD values to billions
                        if unit == 'USD' and abs(value) > 1e6:
                            value = value / 1e9
                            unit = 'USD_billions'
                        
                        # Convert percentages from decimal to percentage (0.46 -> 46)
                        if unit == 'percentage':
                            value = value * 100
                        
                        await insert_financial_data(ticker, metric, period, float(value), unit)
                        inserted_count += 1
            
            # Process balance sheet metrics with exact Tiingo dataCodes
            if balance_sheets:
                balance_data = convert_to_dict(balance_sheets)
                
                balance_metrics = ['assets', 'ppeq', 'investmentsCurrent', 'liabilitiesCurrent', 
                                   'acctRec', 'cashAndEq', 'assetsCurrent', 'inventory', 'acctPay', 
                                   'totalLiabilities', 'sharesBasic', 'equity', 'taxLiabilities', 
                                   'deferredRev', 'debt']
                
                for metric in balance_metrics:
                    if metric in balance_data and balance_data[metric] is not None:
                        value = balance_data[metric]
                        unit = self._determine_unit(metric)
                        
                        if unit == 'USD' and abs(value) > 1e6:
                            value = value / 1e9
                            unit = 'USD_billions'
                        
                        if unit == 'percentage':
                            value = value * 100
                        
                        await insert_financial_data(ticker, metric, period, float(value), unit)
                        inserted_count += 1
            
            # Process cash flow metrics with exact Tiingo dataCodes
            if cash_flows:
                cashflow_data = convert_to_dict(cash_flows)
                
                cashflow_metrics = ['freeCashFlow', 'payDiv', 'depamor', 'issrepayDebt', 
                                    'investmentsAcqDisposals', 'ncff', 'issrepayEquity', 
                                    'ncfo', 'ncfi', 'capex', 'sbcomp']
                
                for metric in cashflow_metrics:
                    if metric in cashflow_data and cashflow_data[metric] is not None:
                        value = cashflow_data[metric]
                        unit = self._determine_unit(metric)
                        
                        if unit == 'USD' and abs(value) > 1e6:
                            value = value / 1e9
                            unit = 'USD_billions'
                        
                        if unit == 'percentage':
                            value = value * 100
                        
                        await insert_financial_data(ticker, metric, period, float(value), unit)
                        inserted_count += 1
            
            # Process overview metrics (calculated ratios and margins) with exact Tiingo dataCodes
            if overview_data:
                overview_dict = convert_to_dict(overview_data)
                
                overview_metrics = ['revenueQoQ', 'grossMargin', 'debtEquity', 'profitMargin',
                                    'rps', 'roa', 'currentRatio', 'longTermDebtEquity', 
                                    'roe', 'epsQoQ', 'bvps', 'bookVal']
                
                for metric in overview_metrics:
                    if metric in overview_dict and overview_dict[metric] is not None:
                        value = overview_dict[metric]
                        unit = self._determine_unit(metric)
                        
                        if unit == 'USD' and abs(value) > 1e6:
                            value = value / 1e9
                            unit = 'USD_billions'
                        
                        if unit == 'percentage':
                            value = value * 100
                        
                        await insert_financial_data(ticker, metric, period, float(value), unit)
                        inserted_count += 1
            
            print(f"  Inserted {inserted_count} metrics for {period}")
    
    def _determine_unit(self, metric_name: str) -> str:
        """Determine the unit for a given metric"""
        # Percentage metrics
        percentage_metrics = ['grossMargin', 'roe', 'roa', 'revenueQoQ', 'epsQoQ', 'profitMargin']
        
        # Ratio metrics
        ratio_metrics = ['peRatio', 'debtEquity', 'currentRatio', 'trailingPEG1Y', 'pbRatio', 
                         'longTermDebtEquity', 'rps']
        
        # Per share metrics
        per_share_metrics = ['eps', 'epsDil', 'bvps']
        
        # Count metrics  
        count_metrics = ['sharesBasic', 'shareswa']
        
        # Large USD metrics that should be in billions
        billion_dollar_metrics = ['marketCap', 'bookVal']
        
        if metric_name in percentage_metrics:
            return 'percentage'
        elif metric_name in ratio_metrics:
            return 'ratio'
        elif metric_name in per_share_metrics:
            return 'USD'
        elif metric_name in count_metrics:
            return 'count'
        elif metric_name in billion_dollar_metrics:
            return 'USD_billions'
        else:
            # Default to USD for financial metrics
            return 'USD'

# ============== Database Setup ==============

async def setup_database(tiingo_api_key: str = None, force_reload: bool = False):
    """
    Setup database with Tiingo data
    
    Args:
        tiingo_api_key: Tiingo API key (will check env if not provided)
        force_reload: Force reload data even if it exists
    
    Raises:
        ValueError: If no Tiingo API key is provided
    """
    await init_database()
    print(f"Database initialized at {DATABASE_PATH}")
    
    # Check if data already exists
    existing_tickers = await get_all_tickers()
    existing_metrics = await get_all_metrics()
    
    if existing_tickers and existing_metrics and not force_reload:
        print(f"Database already populated with {len(existing_tickers)} tickers and {len(existing_metrics)} metrics")
        return
    
    api_key = tiingo_api_key or os.getenv("TIINGO_API_KEY")
    if not api_key:
        raise ValueError(
            "TIINGO_API_KEY is required. Please set it in your environment or .env file.\n"
            "Get a free API key at: https://api.tiingo.com/account/api/token"
        )
    
    # Pre-populate metrics table with all known metrics (using exact Tiingo dataCodes)
    metrics = [
        # Income Statement metrics
        ("revenue", "Total Revenue", "USD_billions"),
        ("epsDil", "Diluted Earnings Per Share", "USD"),
        ("netinc", "Net Income", "USD_billions"),
        ("eps", "Earnings Per Share", "USD"),
        ("ebitda", "EBITDA", "USD_billions"),
        ("rnd", "Research & Development", "USD_billions"),
        ("sga", "Selling, General & Administrative", "USD_billions"),
        ("opinc", "Operating Income", "USD_billions"),
        ("grossProfit", "Gross Profit", "USD_billions"),
        ("opex", "Operating Expenses", "USD_billions"),
        ("ebt", "Earnings Before Tax", "USD_billions"),
        ("costRev", "Cost of Revenue", "USD_billions"),
        ("ebit", "Earnings Before Interest & Taxes", "USD_billions"),
        ("netIncComStock", "Net Income Common Stock", "USD_billions"),
        ("intexp", "Interest Expense", "USD_billions"),
        ("consolidatedIncome", "Consolidated Income", "USD_billions"),
        ("shareswa", "Weighted Average Shares", "count"),
        ("nonControllingInterests", "Non-Controlling Interests", "USD_billions"),
        
        # Balance Sheet metrics
        ("assets", "Total Assets", "USD_billions"),
        ("ppeq", "Property, Plant & Equipment", "USD_billions"),
        ("investmentsCurrent", "Current Investments", "USD_billions"),
        ("liabilitiesCurrent", "Current Liabilities", "USD_billions"),
        ("acctRec", "Accounts Receivable", "USD_billions"),
        ("cashAndEq", "Cash and Cash Equivalents", "USD_billions"),
        ("assetsCurrent", "Current Assets", "USD_billions"),
        ("inventory", "Inventory", "USD_billions"),
        ("acctPay", "Accounts Payable", "USD_billions"),
        ("totalLiabilities", "Total Liabilities", "USD_billions"),
        ("sharesBasic", "Basic Shares Outstanding", "count"),
        ("equity", "Shareholders Equity", "USD_billions"),
        ("taxLiabilities", "Tax Liabilities", "USD_billions"),
        ("deferredRev", "Deferred Revenue", "USD_billions"),
        ("debt", "Total Debt", "USD_billions"),
        
        # Cash Flow metrics
        ("freeCashFlow", "Free Cash Flow", "USD_billions"),
        ("payDiv", "Payment of Dividends", "USD_billions"),
        ("depamor", "Depreciation & Amortization", "USD_billions"),
        ("issrepayDebt", "Issuance/Repayment of Debt", "USD_billions"),
        ("investmentsAcqDisposals", "Investment Acquisitions & Disposals", "USD_billions"),
        ("ncff", "Net Cash Flow from Financing", "USD_billions"),
        ("issrepayEquity", "Issuance/Repayment of Equity", "USD_billions"),
        ("ncfo", "Net Cash Flow from Operations", "USD_billions"),
        ("ncfi", "Net Cash Flow from Investing", "USD_billions"),
        ("capex", "Capital Expenditures", "USD_billions"),
        ("sbcomp", "Stock-Based Compensation", "USD_billions"),
        
        # Overview/Calculated metrics
        ("revenueQoQ", "Revenue Growth Quarter over Quarter", "%"),
        ("grossMargin", "Gross Margin", "%"),
        ("debtEquity", "Debt to Equity Ratio", "ratio"),
        ("profitMargin", "Profit Margin", "%"),
        ("rps", "Revenue Per Share", "ratio"),
        ("roa", "Return on Assets", "%"),
        ("currentRatio", "Current Ratio", "ratio"),
        ("longTermDebtEquity", "Long-Term Debt to Equity", "ratio"),
        ("roe", "Return on Equity", "%"),
        ("epsQoQ", "EPS Growth Quarter over Quarter", "%"),
        ("bvps", "Book Value Per Share", "USD"),
        ("bookVal", "Book Value", "USD_billions"),
        
        # Daily metrics
        ("marketCap", "Market Capitalization", "USD_billions"),
        ("peRatio", "P/E Ratio", "ratio"),
        ("pbRatio", "Price to Book Ratio", "ratio"),
        ("trailingPEG1Y", "PEG Ratio", "ratio"),
    ]
    
    for metric_name, description, unit in metrics:
        await insert_metric(metric_name, description, unit)
    
    loader = TiingoDataLoader(api_key)
    
    # Load all tickers for training (matching parent project)
    tickers_to_load = [
        ("AAPL", "Apple Inc.", "Technology"),
        ("MSFT", "Microsoft Corporation", "Technology"),
        ("GOOGL", "Alphabet Inc.", "Technology"),
        ("AMZN", "Amazon.com Inc.", "Technology"),
        ("META", "Meta Platforms Inc.", "Technology"),
        ("TSLA", "Tesla Inc.", "Consumer Cyclical"),
        ("NVDA", "NVIDIA Corporation", "Technology"),
        ("INTC", "Intel Corporation", "Technology"),
        ("VST", "Vistra Corp.", "Utilities"),
        ("GEV", "GE Vernova Inc.", "Industrials"),
        ("MU", "Micron Technology Inc.", "Technology"),
        ("CEG", "Constellation Energy Corporation", "Utilities"),
        ("VRT", "Vertiv Holdings Co.", "Technology"),
        ("NRG", "NRG Energy Inc.", "Utilities"),
        ("TLN", "Talen Energy Corporation", "Utilities"),
        ("NBIS", "Nebius Group N.V.", "Technology"),
        ("COHR", "Coherent Corp.", "Technology"),
        ("ONTO", "Onto Innovation Inc.", "Technology"),
        ("BE", "Bloom Energy Corporation", "Industrials"),
        ("CRDO", "Credo Technology Group Holding Ltd", "Technology"),
        ("FN", "Fabrinet", "Technology"),
        ("TSM", "Taiwan Semiconductor Manufacturing", "Technology"),
        ("AVGO", "Broadcom Inc.", "Technology"),
        ("MRVL", "Marvell Technology Inc.", "Technology"),
        ("AMAT", "Applied Materials Inc.", "Technology"),
        ("ORCL", "Oracle Corporation", "Technology"),
        ("TSSI", "TSS Inc.", "Technology"),
        ("LRCX", "Lam Research Corporation", "Technology"),
        ("ASML", "ASML Holding N.V.", "Technology"),
        ("ANET", "Arista Networks Inc.", "Technology"),
    ]
    
    success_count = 0
    for ticker, company_name, sector in tickers_to_load:
        if await loader.load_ticker_data(ticker, company_name, sector):
            success_count += 1
    
    if success_count == 0:
        raise ValueError(
            "Failed to load any data from Tiingo API. Please check:\n"
            "1. Your API key is valid\n"
            "2. You have the Fundamental Data add-on enabled\n"
            "3. Your internet connection is working"
        )
    
    print(f"Successfully loaded data for {success_count} tickers from Tiingo")
    
    # Verify data was loaded
    tickers = await get_all_tickers()
    metrics = await get_all_metrics()
    
    if not tickers or not metrics:
        raise ValueError("Database verification failed: no tickers or metrics found")
    
    print(f"Database ready with {len(tickers)} tickers and {len(metrics)} metrics")

if __name__ == "__main__":
    # Test database setup
    import sys
    try:
        asyncio.run(setup_database())
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)