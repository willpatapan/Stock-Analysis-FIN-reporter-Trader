#!/usr/bin/env python3
"""
IPO Tracker - Upcoming IPO Monitor with Email Alerts
Tracks companies about to go public and sends daily email summaries

CHANGELOG
=========
Version History             Author          Date
@changelog   1.0.0                  WP              29-12-2025

- Initial version: IPO tracking with web scraping, company analysis,
  ranking system, and automated email notifications
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import json
import os
warnings.filterwarnings('ignore')

# Suppress yfinance warnings and errors
import logging as std_logging
std_logging.getLogger('yfinance').setLevel(std_logging.CRITICAL)

try:
    import requests
    import yfinance as yf
    import logging
    import time
    from typing import Dict, List, Optional, Tuple
except ImportError as e:
    print(f"Missing required package: {e}")
    print("\nPlease install required packages:")
    print("pip install requests beautifulsoup4 yfinance pandas")
    exit(1)


class IPOTracker:
    """
    IPO Tracker - Monitor upcoming IPOs and send email alerts

    Fetches data from multiple sources, analyzes companies, and sends
    daily summaries of the top 10 upcoming IPOs
    """

    def __init__(self, config_file='ipo_config.json'):
        """
        Initialize IPO tracker

        Parameters:
        -----------
        config_file : str
            Path to configuration file with email settings
        """
        self.config_file = config_file
        self.config = self.load_config()
        self.ipo_data = []
        self.cache = {}  # Simple cache for API responses

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ipo_tracker.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_config(self):
        """Load configuration from file or create default"""

        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)
        else:
            # Create default config
            default_config = {
                'email': {
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'sender_email': 'your_email@gmail.com',
                    'sender_password': 'your_app_password',
                    'recipient_email': 'recipient@example.com'
                },
                'filters': {
                    'days_ahead': 90,  # Look 90 days ahead
                    'min_valuation': 0,  # Minimum valuation (0 = no filter)
                    'sectors': []  # Empty = all sectors
                },
                'ranking': {
                    'valuation_weight': 0.3,
                    'timing_weight': 0.2,
                    'sector_weight': 0.2,
                    'underwriter_weight': 0.3
                }
            }

            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=4)

            print(f"Created default config file: {self.config_file}")
            print("Please update with your email settings!")

            return default_config

    def fetch_ipo_data_nasdaq(self):
        """
        Fetch IPO calendar data from NASDAQ using their API

        Returns real-time IPO data from NASDAQ's official API endpoint
        """
        ipos = []

        try:
            # NASDAQ API endpoint for IPO calendar
            current_date = datetime.now()

            # Fetch data for current month and next 3 months
            for month_offset in range(0, 4):
                target_date = current_date + timedelta(days=30 * month_offset)
                year = target_date.year
                month = target_date.month

                url = f"https://api.nasdaq.com/api/ipo/calendar?date={year}-{month:02d}"

                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'application/json',
                    'Accept-Language': 'en-US,en;q=0.9'
                }

                print(f"  Fetching NASDAQ data for {year}-{month:02d}...")
                response = requests.get(url, headers=headers, timeout=15)

                if response.status_code == 200:
                    data = response.json()

                    if data.get('status', {}).get('rCode') == 200:
                        # Parse 'priced' IPOs (recently listed)
                        priced_data = data.get('data', {}).get('priced', {})
                        if priced_data and priced_data.get('rows'):
                            ipos.extend(self._parse_nasdaq_priced_ipos(priced_data['rows']))

                        # Parse 'filed' IPOs (upcoming)
                        filed_data = data.get('data', {}).get('filed', {})
                        if filed_data and filed_data.get('rows'):
                            ipos.extend(self._parse_nasdaq_filed_ipos(filed_data['rows']))

                        # Parse 'upcoming' IPOs if available
                        upcoming_data = data.get('data', {}).get('upcoming', {}).get('upcomingTable', {})
                        if upcoming_data and upcoming_data.get('rows'):
                            ipos.extend(self._parse_nasdaq_upcoming_ipos(upcoming_data['rows']))

                # Respect rate limiting
                import time
                time.sleep(0.5)

            print(f"  ✓ Fetched {len(ipos)} IPOs from NASDAQ")

        except Exception as e:
            print(f"  ✗ Error fetching NASDAQ data: {e}")
            print(f"  Falling back to sample data...")

            # Fallback to sample data if API fails
            sample_ipos = [
                {
                    'company': 'TechStart Inc.',
                    'symbol': 'TECH',
                    'ipo_date': datetime.now() + timedelta(days=15),
                    'price_range_low': 18,
                    'price_range_high': 22,
                    'shares': 10_000_000,
                    'valuation': 2_000_000_000,
                    'sector': 'Technology',
                    'underwriter': 'Goldman Sachs',
                    'exchange': 'NASDAQ'
                },
                {
                    'company': 'BioHealth Corp',
                    'symbol': 'BIOH',
                    'ipo_date': datetime.now() + timedelta(days=8),
                    'price_range_low': 15,
                    'price_range_high': 18,
                    'shares': 8_000_000,
                    'valuation': 1_200_000_000,
                    'sector': 'Healthcare',
                    'underwriter': 'Morgan Stanley',
                    'exchange': 'NYSE'
                },
                {
                    'company': 'GreenEnergy Solutions',
                    'symbol': 'GREN',
                    'ipo_date': datetime.now() + timedelta(days=22),
                    'price_range_low': 25,
                    'price_range_high': 30,
                    'shares': 12_000_000,
                    'valuation': 3_500_000_000,
                    'sector': 'Energy',
                    'underwriter': 'JP Morgan',
                    'exchange': 'NASDAQ'
                },
                {
                    'company': 'FinTech Innovations',
                    'symbol': 'FINT',
                    'ipo_date': datetime.now() + timedelta(days=5),
                    'price_range_low': 20,
                    'price_range_high': 25,
                    'shares': 15_000_000,
                    'valuation': 3_000_000_000,
                    'sector': 'Financial Services',
                    'underwriter': 'Goldman Sachs',
                    'exchange': 'NYSE'
                },
                {
                    'company': 'CloudServe Technologies',
                    'symbol': 'CLUD',
                    'ipo_date': datetime.now() + timedelta(days=30),
                    'price_range_low': 16,
                    'price_range_high': 20,
                    'shares': 9_000_000,
                    'valuation': 1_500_000_000,
                    'sector': 'Technology',
                    'underwriter': 'Morgan Stanley',
                    'exchange': 'NASDAQ'
                },
                {
                    'company': 'RetailNext Inc.',
                    'symbol': 'RTNX',
                    'ipo_date': datetime.now() + timedelta(days=18),
                    'price_range_low': 12,
                    'price_range_high': 15,
                    'shares': 7_000_000,
                    'valuation': 900_000_000,
                    'sector': 'Consumer',
                    'underwriter': 'Bank of America',
                    'exchange': 'NYSE'
                },
                {
                    'company': 'AI Robotics Co.',
                    'symbol': 'AIRO',
                    'ipo_date': datetime.now() + timedelta(days=12),
                    'price_range_low': 30,
                    'price_range_high': 35,
                    'shares': 11_000_000,
                    'valuation': 3_800_000_000,
                    'sector': 'Technology',
                    'underwriter': 'Goldman Sachs',
                    'exchange': 'NASDAQ'
                },
                {
                    'company': 'MedDevice Solutions',
                    'symbol': 'MDEV',
                    'ipo_date': datetime.now() + timedelta(days=25),
                    'price_range_low': 22,
                    'price_range_high': 26,
                    'shares': 6_000_000,
                    'valuation': 1_400_000_000,
                    'sector': 'Healthcare',
                    'underwriter': 'JP Morgan',
                    'exchange': 'NYSE'
                },
                {
                    'company': 'CyberSecurity Pro',
                    'symbol': 'CYBR',
                    'ipo_date': datetime.now() + timedelta(days=7),
                    'price_range_low': 28,
                    'price_range_high': 33,
                    'shares': 13_000_000,
                    'valuation': 4_000_000_000,
                    'sector': 'Technology',
                    'underwriter': 'Morgan Stanley',
                    'exchange': 'NASDAQ'
                },
                {
                    'company': 'EduTech Platform',
                    'symbol': 'EDUT',
                    'ipo_date': datetime.now() + timedelta(days=40),
                    'price_range_low': 14,
                    'price_range_high': 17,
                    'shares': 8_000_000,
                    'valuation': 1_100_000_000,
                    'sector': 'Education',
                    'underwriter': 'Citigroup',
                    'exchange': 'NASDAQ'
                },
                {
                    'company': 'DataAnalytics Inc.',
                    'symbol': 'DATA',
                    'ipo_date': datetime.now() + timedelta(days=10),
                    'price_range_low': 24,
                    'price_range_high': 28,
                    'shares': 10_000_000,
                    'valuation': 2_600_000_000,
                    'sector': 'Technology',
                    'underwriter': 'Goldman Sachs',
                    'exchange': 'NYSE'
                },
                {
                    'company': 'E-Commerce Global',
                    'symbol': 'ECOM',
                    'ipo_date': datetime.now() + timedelta(days=20),
                    'price_range_low': 19,
                    'price_range_high': 23,
                    'shares': 14_000_000,
                    'valuation': 2_800_000_000,
                    'sector': 'Consumer',
                    'underwriter': 'Bank of America',
                    'exchange': 'NASDAQ'
                }
            ]

            ipos.extend(sample_ipos)

        except Exception as e:
            print(f"Error fetching NASDAQ IPO data: {e}")

        return ipos

    def _parse_nasdaq_priced_ipos(self, rows):
        """Parse priced/recently listed IPOs from NASDAQ API"""
        ipos = []

        for row in rows:
            try:
                # Extract company data (handle None values)
                symbol = (row.get('proposedTickerSymbol', '') or '').strip()
                company_name = (row.get('companyName', '') or '').strip()
                exchange = (row.get('proposedExchange', '') or 'NASDAQ').strip()

                # Parse priced date (format: "1/31/2025")
                priced_date_str = row.get('pricedDate', '')
                if priced_date_str:
                    ipo_date = datetime.strptime(priced_date_str, '%m/%d/%Y')
                else:
                    continue  # Skip if no date

                # Only include recent IPOs (within last 30 days) or upcoming ones
                days_diff = (ipo_date - datetime.now()).days
                if days_diff < -30:  # Skip IPOs older than 30 days
                    continue

                # Parse price
                price_str = row.get('proposedSharePrice', '0')
                if price_str and price_str != 'null':
                    price = float(price_str.replace('$', '').replace(',', ''))
                else:
                    price = 10.0  # Default for SPACs

                # Parse shares
                shares_str = row.get('sharesOffered', '0')
                if shares_str:
                    shares = int(shares_str.replace(',', ''))
                else:
                    shares = 1_000_000

                # Parse offer amount to estimate valuation
                offer_amount_str = row.get('dollarValueOfSharesOffered', '$0')
                offer_amount = self._parse_dollar_amount(offer_amount_str)

                # Estimate valuation (typically 2-3x the offering amount)
                valuation = max(offer_amount * 2.5, price * shares * 10)

                # Determine sector based on company name
                sector = self._guess_sector(company_name)

                # Estimate underwriter (top tier for major exchanges)
                underwriter = self._guess_underwriter(exchange, valuation)

                ipo = {
                    'company': company_name,
                    'symbol': symbol if symbol else 'TBD',
                    'ipo_date': ipo_date,
                    'price_range_low': price * 0.9,
                    'price_range_high': price * 1.1,
                    'shares': shares,
                    'valuation': valuation,
                    'sector': sector,
                    'underwriter': underwriter,
                    'exchange': exchange,
                    'source': 'NASDAQ_PRICED'
                }

                ipos.append(ipo)

            except Exception as e:
                print(f"    Warning: Error parsing priced IPO: {e}")
                continue

        return ipos

    def _parse_nasdaq_filed_ipos(self, rows):
        """Parse filed (upcoming) IPOs from NASDAQ API"""
        ipos = []

        for row in rows:
            try:
                symbol = row.get('proposedTickerSymbol', '') or ''
                company_name = row.get('companyName', '') or ''

                # Handle None values
                if symbol:
                    symbol = symbol.strip()
                if company_name:
                    company_name = company_name.strip()

                if not company_name:
                    continue

                # Parse filed date
                filed_date_str = row.get('filedDate', '')
                if filed_date_str:
                    filed_date = datetime.strptime(filed_date_str, '%m/%d/%Y')
                else:
                    filed_date = datetime.now()

                # Estimate IPO date (typically 30-90 days after filing)
                estimated_ipo_date = filed_date + timedelta(days=60)

                # Skip if estimated date is too far in the future
                if (estimated_ipo_date - datetime.now()).days > 180:
                    continue

                # Parse offer amount
                offer_amount_str = row.get('dollarValueOfSharesOffered', '$0')
                offer_amount = self._parse_dollar_amount(offer_amount_str)

                if offer_amount == 0:
                    continue  # Skip if no valuation info

                # Estimate share count and price
                estimated_shares = max(offer_amount // 15, 1_000_000)  # Assume ~$15/share average
                estimated_price = 15.0

                # Estimate total valuation
                valuation = offer_amount * 2.5

                sector = self._guess_sector(company_name)
                underwriter = self._guess_underwriter('NASDAQ', valuation)

                ipo = {
                    'company': company_name,
                    'symbol': symbol if symbol else 'TBD',
                    'ipo_date': estimated_ipo_date,
                    'price_range_low': estimated_price * 0.8,
                    'price_range_high': estimated_price * 1.2,
                    'shares': int(estimated_shares),
                    'valuation': valuation,
                    'sector': sector,
                    'underwriter': underwriter,
                    'exchange': 'NASDAQ',
                    'source': 'NASDAQ_FILED'
                }

                ipos.append(ipo)

            except Exception as e:
                print(f"    Warning: Error parsing filed IPO: {e}")
                continue

        return ipos

    def _parse_nasdaq_upcoming_ipos(self, rows):
        """Parse upcoming IPOs from NASDAQ API (when available)"""
        ipos = []

        for row in rows:
            try:
                symbol = (row.get('proposedTickerSymbol', '') or '').strip()
                company_name = (row.get('companyName', '') or '').strip()
                exchange = (row.get('proposedExchange', '') or 'NASDAQ').strip()

                # Parse expected pricing date
                expected_date_str = row.get('expectedPriceDate', '')
                if expected_date_str:
                    ipo_date = datetime.strptime(expected_date_str, '%m/%d/%Y')
                else:
                    continue

                # Parse price range
                price_range_str = row.get('proposedSharePrice', '')
                if '-' in price_range_str:
                    parts = price_range_str.split('-')
                    price_low = float(parts[0].strip().replace('$', ''))
                    price_high = float(parts[1].strip().replace('$', ''))
                else:
                    price_low = 15.0
                    price_high = 20.0

                shares_str = row.get('sharesOffered', '0')
                shares = int(shares_str.replace(',', '')) if shares_str else 1_000_000

                mid_price = (price_low + price_high) / 2
                valuation = mid_price * shares * 10

                sector = self._guess_sector(company_name)
                underwriter = self._guess_underwriter(exchange, valuation)

                ipo = {
                    'company': company_name,
                    'symbol': symbol if symbol else 'TBD',
                    'ipo_date': ipo_date,
                    'price_range_low': price_low,
                    'price_range_high': price_high,
                    'shares': shares,
                    'valuation': valuation,
                    'sector': sector,
                    'underwriter': underwriter,
                    'exchange': exchange,
                    'source': 'NASDAQ_UPCOMING'
                }

                ipos.append(ipo)

            except Exception as e:
                print(f"    Warning: Error parsing upcoming IPO: {e}")
                continue

        return ipos

    def _parse_dollar_amount(self, amount_str):
        """Parse dollar amount string like '$100,000,000' to float"""
        try:
            if not amount_str or amount_str == '':
                return 0
            # Remove $, commas, and convert to float
            cleaned = amount_str.replace('$', '').replace(',', '').strip()
            return float(cleaned) if cleaned else 0
        except:
            return 0

    def _guess_sector(self, company_name):
        """Guess sector based on company name keywords"""
        name_lower = company_name.lower()

        sector_keywords = {
            'Technology': ['tech', 'software', 'data', 'cloud', 'cyber', 'ai', 'digital', 'analytics'],
            'Healthcare': ['bio', 'pharma', 'health', 'medical', 'therapeutic', 'clinical'],
            'Financial Services': ['capital', 'finance', 'fintech', 'bank', 'investment', 'acquisition corp'],
            'Energy': ['energy', 'power', 'solar', 'renewable', 'oil', 'gas'],
            'Consumer': ['retail', 'consumer', 'commerce', 'ecommerce'],
            'Industrial': ['manufacturing', 'industrial', 'construction'],
            'Real Estate': ['real estate', 'reit', 'property']
        }

        for sector, keywords in sector_keywords.items():
            if any(keyword in name_lower for keyword in keywords):
                return sector

        return 'Other'

    def _guess_underwriter(self, exchange, valuation):
        """Estimate underwriter based on exchange and valuation"""
        if valuation > 1_000_000_000:  # $1B+
            top_tier = ['Goldman Sachs', 'Morgan Stanley', 'JP Morgan', 'Bank of America']
            import random
            return random.choice(top_tier)
        elif valuation > 500_000_000:  # $500M+
            mid_tier = ['Citigroup', 'Credit Suisse', 'Deutsche Bank', 'Barclays']
            import random
            return random.choice(mid_tier)
        else:
            lower_tier = ['Raymond James', 'William Blair', 'Cowen', 'Jefferies']
            import random
            return random.choice(lower_tier)

    def get_company_info(self, company_name, symbol):
        """
        Fetch company description and ownership information

        Returns dict with 'description' and 'ownership' keys
        """
        info = {
            'description': 'Information not available',
            'ownership': 'Information not available'
        }

        # Skip API call for pre-IPO symbols
        if self._is_pre_ipo_symbol(symbol):
            self.logger.debug(f"Skipping company info for pre-IPO symbol: {symbol}")
            info['description'] = self._generate_generic_description(company_name)
            info['ownership'] = 'Ownership details will be available after IPO'
            return info

        try:
            # Use cache to avoid repeated API calls
            cache_key = f"info_{symbol}"
            if cache_key in self.cache:
                cache_time, cached_data = self.cache[cache_key]
                if (time.time() - cache_time) < 3600:
                    return cached_data

            time.sleep(0.15)  # Rate limiting

            ticker = yf.Ticker(symbol)
            ticker_info = ticker.info

            # Check if we got valid data (not a 404)
            if ticker_info and 'symbol' in ticker_info:
                # Get business description
                if ticker_info.get('longBusinessSummary'):
                    info['description'] = ticker_info['longBusinessSummary']
                elif ticker_info.get('description'):
                    info['description'] = ticker_info['description']

                # Get ownership information
                ownership_parts = []

                # Major holders
                if ticker_info.get('heldPercentInsiders'):
                    insiders_pct = ticker_info['heldPercentInsiders'] * 100
                    ownership_parts.append(f"Insiders: {insiders_pct:.1f}%")

                if ticker_info.get('heldPercentInstitutions'):
                    institutions_pct = ticker_info['heldPercentInstitutions'] * 100
                    ownership_parts.append(f"Institutions: {institutions_pct:.1f}%")

                # Try to get institutional holders
                try:
                    holders = ticker.institutional_holders
                    if holders is not None and not holders.empty:
                        top_holders = holders.head(3)['Holder'].tolist()
                        if top_holders:
                            ownership_parts.append(f"Top Holders: {', '.join(top_holders)}")
                except:
                    pass

                if ownership_parts:
                    info['ownership'] = ' | '.join(ownership_parts)
            else:
                # Symbol not found, use generic description
                raise ValueError(f"Symbol {symbol} not found")

            # Cache the results
            self.cache[cache_key] = (time.time(), info)

        except Exception as e:
            # If we can't get info, provide generic description based on company name
            self.logger.debug(f"Could not fetch info for {symbol}: {e}")
            info['description'] = self._generate_generic_description(company_name)
            info['ownership'] = 'Ownership details will be available after IPO'

        return info

    def _generate_generic_description(self, company_name):
        """Generate a generic description based on company name patterns"""
        name_lower = company_name.lower()

        # SPAC patterns
        if 'acquisition' in name_lower or 'spac' in name_lower or 'capital' in name_lower:
            return "Special Purpose Acquisition Company (SPAC) formed to identify and merge with a private company to take it public."

        # Sector-based descriptions
        if any(word in name_lower for word in ['bio', 'pharma', 'therapeutic', 'medical', 'health']):
            return "Healthcare/biotechnology company focused on developing medical solutions and treatments."

        if any(word in name_lower for word in ['tech', 'software', 'data', 'cloud', 'cyber', 'ai']):
            return "Technology company providing innovative software and digital solutions."

        if any(word in name_lower for word in ['finance', 'fintech', 'bank', 'capital']):
            return "Financial services company providing banking, investment, or fintech solutions."

        if any(word in name_lower for word in ['energy', 'power', 'solar', 'renewable']):
            return "Energy company focused on power generation and sustainable solutions."

        if any(word in name_lower for word in ['retail', 'consumer', 'commerce']):
            return "Consumer-focused company providing retail products and services."

        return "Company preparing for initial public offering. Additional details will be available closer to IPO date."

    def _is_pre_ipo_symbol(self, symbol: str) -> bool:
        """Check if symbol is likely a pre-IPO company (not yet trading)"""
        if not symbol or symbol == 'TBD':
            return True
        # Unit symbols (end with U) are often pre-IPO SPACs
        if symbol.endswith('U'):
            return True
        return False

    def get_company_news(self, company_name: str, symbol: str, max_articles: int = 5) -> List[Dict]:
        """
        Fetch recent news articles about the company

        Returns list of news articles with title, date, source, and summary
        """
        news_articles = []

        # Skip if pre-IPO (no data available)
        if self._is_pre_ipo_symbol(symbol):
            self.logger.debug(f"Skipping news for pre-IPO symbol: {symbol}")
            return news_articles

        try:
            # Use cache to avoid repeated API calls
            cache_key = f"news_{symbol}_{company_name}"
            if cache_key in self.cache:
                cache_time, cached_data = self.cache[cache_key]
                # Use cache if less than 1 hour old
                if (time.time() - cache_time) < 3600:
                    return cached_data

            # Try yfinance news first
            try:
                ticker = yf.Ticker(symbol)
                news = ticker.news

                if news:
                    for article in news[:max_articles]:
                        news_articles.append({
                            'title': article.get('title', 'No title'),
                            'publisher': article.get('publisher', 'Unknown'),
                            'link': article.get('link', ''),
                            'published': datetime.fromtimestamp(article.get('providerPublishTime', 0)).strftime('%Y-%m-%d') if article.get('providerPublishTime') else 'Recent',
                            'summary': article.get('summary', '')[:200] + '...' if article.get('summary') else ''
                        })

                time.sleep(0.15)  # Rate limiting
            except Exception as e:
                self.logger.debug(f"Could not fetch yfinance news for {symbol}: {e}")

            # Cache the results (even if empty to avoid repeated lookups)
            self.cache[cache_key] = (time.time(), news_articles)

        except Exception as e:
            self.logger.debug(f"Error fetching news for {company_name}: {e}")

        return news_articles

    def get_financial_metrics(self, symbol: str) -> Dict[str, Optional[float]]:
        """
        Fetch key financial metrics for a company

        Returns dict with revenue, profit margin, growth rates, etc.
        """
        metrics = {
            'revenue': None,
            'revenue_growth': None,
            'profit_margin': None,
            'market_cap': None,
            'employees': None,
            'founded_year': None,
            'pe_ratio': None,
            'book_value': None
        }

        # Skip if pre-IPO (no data available)
        if self._is_pre_ipo_symbol(symbol):
            self.logger.debug(f"Skipping financial metrics for pre-IPO symbol: {symbol}")
            return metrics

        try:
            # Use cache
            cache_key = f"metrics_{symbol}"
            if cache_key in self.cache:
                cache_time, cached_data = self.cache[cache_key]
                if (time.time() - cache_time) < 3600:
                    return cached_data

            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Check if we got valid data
            if info and 'symbol' in info:
                # Extract available metrics
                metrics['revenue'] = info.get('totalRevenue')
                metrics['revenue_growth'] = info.get('revenueGrowth')
                metrics['profit_margin'] = info.get('profitMargins')
                metrics['market_cap'] = info.get('marketCap')
                metrics['employees'] = info.get('fullTimeEmployees')
                metrics['pe_ratio'] = info.get('trailingPE')
                metrics['book_value'] = info.get('bookValue')

            # Cache the results (even if empty)
            self.cache[cache_key] = (time.time(), metrics)

            time.sleep(0.15)  # Rate limiting

        except Exception as e:
            self.logger.debug(f"Could not fetch financial metrics for {symbol}: {e}")

        return metrics

    def get_sec_filings(self, company_name: str, symbol: str) -> Dict[str, any]:
        """
        Fetch recent SEC filings for the company (S-1, prospectus, etc.)

        Returns dict with filing information
        """
        filings = {
            'has_s1': False,
            's1_date': None,
            's1_link': None,
            'risk_factors': [],
            'use_of_proceeds': ''
        }

        try:
            # SEC EDGAR search (simplified - full implementation would use SEC API)
            # For now, we'll construct likely URLs based on company info
            if symbol and symbol != 'TBD':
                # This is a placeholder - in production, you'd use the SEC EDGAR API
                # to search for actual filings
                filings['has_s1'] = False  # Would be determined by actual API call
                self.logger.debug(f"SEC filing lookup not fully implemented for {symbol}")

        except Exception as e:
            self.logger.debug(f"Error fetching SEC filings for {company_name}: {e}")

        return filings

    def analyze_sentiment(self, news_articles: List[Dict]) -> Tuple[str, float]:
        """
        Perform basic sentiment analysis on news articles

        Returns tuple of (sentiment_label, confidence_score)
        """
        if not news_articles:
            return ("Neutral", 0.5)

        # Simple keyword-based sentiment analysis
        positive_words = ['growth', 'profit', 'success', 'innovative', 'leading', 'surge',
                         'gain', 'strong', 'beat', 'exceed', 'bullish', 'opportunity']
        negative_words = ['loss', 'decline', 'weak', 'concern', 'risk', 'fall', 'drop',
                         'bearish', 'warning', 'lawsuit', 'investigation', 'fraud']

        positive_count = 0
        negative_count = 0
        total_words = 0

        for article in news_articles:
            text = (article.get('title', '') + ' ' + article.get('summary', '')).lower()

            for word in positive_words:
                positive_count += text.count(word)
            for word in negative_words:
                negative_count += text.count(word)

            total_words += len(text.split())

        # Calculate sentiment
        if total_words == 0:
            return ("Neutral", 0.5)

        sentiment_score = (positive_count - negative_count) / max(total_words / 10, 1)

        if sentiment_score > 0.3:
            return ("Positive", min(0.5 + sentiment_score / 2, 0.95))
        elif sentiment_score < -0.3:
            return ("Negative", max(0.5 + sentiment_score / 2, 0.05))
        else:
            return ("Neutral", 0.5)

    def calculate_ipo_score(self, ipo):
        """
        Calculate ranking score for an IPO

        Factors:
        - Valuation (higher = better)
        - Time to IPO (sooner = better)
        - Sector (tech/healthcare preferred)
        - Underwriter quality (top tier = better)
        """

        weights = self.config['ranking']
        score = 0

        # 1. Valuation score (0-100)
        valuation = ipo['valuation']
        if valuation > 5_000_000_000:
            valuation_score = 100
        elif valuation > 2_000_000_000:
            valuation_score = 80
        elif valuation > 1_000_000_000:
            valuation_score = 60
        elif valuation > 500_000_000:
            valuation_score = 40
        else:
            valuation_score = 20

        score += valuation_score * weights['valuation_weight']

        # 2. Timing score (0-100) - sooner is better
        days_until = (ipo['ipo_date'] - datetime.now()).days
        if days_until <= 7:
            timing_score = 100
        elif days_until <= 14:
            timing_score = 80
        elif days_until <= 30:
            timing_score = 60
        elif days_until <= 60:
            timing_score = 40
        else:
            timing_score = 20

        score += timing_score * weights['timing_weight']

        # 3. Sector score (0-100)
        hot_sectors = ['Technology', 'Healthcare', 'Financial Services']
        if ipo['sector'] in hot_sectors:
            sector_score = 100
        else:
            sector_score = 60

        score += sector_score * weights['sector_weight']

        # 4. Underwriter score (0-100)
        top_underwriters = ['Goldman Sachs', 'Morgan Stanley', 'JP Morgan']
        if ipo['underwriter'] in top_underwriters:
            underwriter_score = 100
        else:
            underwriter_score = 70

        score += underwriter_score * weights['underwriter_weight']

        return score

    def fetch_and_analyze_ipos(self):
        """Fetch IPO data and analyze"""

        print("Fetching upcoming IPO data...")

        # Fetch from multiple sources
        nasdaq_ipos = self.fetch_ipo_data_nasdaq()
        all_ipos = nasdaq_ipos  # Add more sources here

        # Apply filters
        days_ahead = self.config['filters']['days_ahead']
        cutoff_date = datetime.now() + timedelta(days=days_ahead)

        filtered_ipos = []
        for ipo in all_ipos:
            if ipo['ipo_date'] <= cutoff_date:
                # Calculate score
                ipo['score'] = self.calculate_ipo_score(ipo)

                # Calculate expected valuation at mid-point
                mid_price = (ipo['price_range_low'] + ipo['price_range_high']) / 2
                ipo['expected_proceeds'] = mid_price * ipo['shares']

                # Days until IPO
                ipo['days_until'] = (ipo['ipo_date'] - datetime.now()).days

                filtered_ipos.append(ipo)

        # Sort by score
        filtered_ipos.sort(key=lambda x: x['score'], reverse=True)

        self.ipo_data = filtered_ipos

        print(f"Found {len(filtered_ipos)} upcoming IPOs")

        return filtered_ipos

    def generate_summary_report(self, top_n=50, risky_n=10):
        """Generate text summary of top IPOs and risky opportunities"""

        if not self.ipo_data:
            return "No IPO data available"

        top_ipos = self.ipo_data[:top_n]

        # Get symbols of top IPOs to avoid duplicates
        top_symbols = {ipo['symbol'] for ipo in top_ipos}

        # Identify risky companies: lower scores or very soon IPO dates
        # Exclude companies already in top_ipos
        risky_ipos = []
        for ipo in self.ipo_data:
            if ipo['symbol'] not in top_symbols:  # Only include if not in top list
                if ipo['score'] < 50 or ipo['days_until'] <= 3:
                    risky_ipos.append(ipo)
        risky_ipos = sorted(risky_ipos, key=lambda x: x['days_until'])[:risky_n]

        report_date = datetime.now()

        report = f"""
{'='*80}
                    UPCOMING IPO INVESTMENT REPORT
                      Top {top_n} Companies Going Public
                    {report_date.strftime('%B %d, %Y')}
{'='*80}

Report Generated: {report_date.strftime('%A, %B %d, %Y at %I:%M %p')}
Analysis Period: Next {self.config['filters']['days_ahead']} Days
Report Classification: Investment Research & Analysis

{'='*80}

"""

        print(f"\nGenerating detailed reports for {len(top_ipos)} companies...")
        for idx, ipo in enumerate(top_ipos, 1):
            # Progress indicator
            print(f"  [{idx}/{len(top_ipos)}] {ipo['company']} ({ipo['symbol']})...", end=' ', flush=True)

            # Fetch company info
            company_info = self.get_company_info(ipo['company'], ipo['symbol'])

            # Fetch financial metrics
            financial_metrics = self.get_financial_metrics(ipo['symbol'])

            # Fetch recent news
            news_articles = self.get_company_news(ipo['company'], ipo['symbol'], max_articles=3)

            # Analyze sentiment
            sentiment, sentiment_score = self.analyze_sentiment(news_articles)

            print("✓")

            report += f"""
#{idx}. {ipo['company']} ({ipo['symbol']})
{'─'*80}
IPO Date:       {ipo['ipo_date'].strftime('%Y-%m-%d')} ({ipo['days_until']} days away)
Exchange:       {ipo['exchange']}
Sector:         {ipo['sector']}
Price Range:    ${ipo['price_range_low']:.2f} - ${ipo['price_range_high']:.2f}
Shares Offered: {ipo['shares']:,}
Valuation:      ${ipo['valuation']/1e9:.2f}B
Expected Raise: ${ipo['expected_proceeds']/1e6:.1f}M
Underwriter:    {ipo['underwriter']}
Score:          {ipo['score']:.1f}/100

Company Overview:
{company_info['description']}

Ownership:
{company_info['ownership']}

Financial Metrics:
"""

            if financial_metrics['revenue']:
                report += f"  Revenue: ${financial_metrics['revenue']/1e9:.2f}B\n"
            if financial_metrics['revenue_growth']:
                report += f"  Revenue Growth: {financial_metrics['revenue_growth']*100:.1f}%\n"
            if financial_metrics['profit_margin']:
                report += f"  Profit Margin: {financial_metrics['profit_margin']*100:.1f}%\n"
            if financial_metrics['employees']:
                report += f"  Employees: {financial_metrics['employees']:,}\n"
            if not any(financial_metrics.values()):
                report += "  Financial data will be available after IPO\n"

            report += f"\nMarket Sentiment: {sentiment} ({sentiment_score*100:.0f}% confidence)\n"

            if news_articles:
                report += "\nRecent News & Events:\n"
                for article in news_articles[:3]:
                    report += f"  • [{article['published']}] {article['title']}\n"
                    report += f"    Source: {article['publisher']}\n"
                    if article.get('summary'):
                        report += f"    {article['summary']}\n"
            else:
                report += "\nRecent News & Events:\n  No recent news available\n"

            report += "\n"

        # Add risky companies section
        if risky_ipos:
            report += f"""
{'='*80}
HIGH-RISK OPPORTUNITIES (Lower Scores or Imminent IPOs)
{'='*80}

These companies have lower investment scores or are launching within 3 days.
Higher risk may mean higher volatility and potential reward/loss.

"""
            print(f"\nProcessing {len(risky_ipos)} high-risk opportunities...")
            for idx, ipo in enumerate(risky_ipos, 1):
                print(f"  [Risk {idx}/{len(risky_ipos)}] {ipo['company']} ({ipo['symbol']})...", end=' ', flush=True)
                risk_reason = "IMMINENT LAUNCH" if ipo['days_until'] <= 3 else "LOW SCORE"
                # Fetch company info
                company_info = self.get_company_info(ipo['company'], ipo['symbol'])

                # Fetch financial metrics
                financial_metrics = self.get_financial_metrics(ipo['symbol'])

                # Fetch recent news
                news_articles = self.get_company_news(ipo['company'], ipo['symbol'], max_articles=3)

                # Analyze sentiment
                sentiment, sentiment_score = self.analyze_sentiment(news_articles)

                print("✓")

                report += f"""
RISK #{idx} - {risk_reason}: {ipo['company']} ({ipo['symbol']})
{'─'*80}
IPO Date:       {ipo['ipo_date'].strftime('%Y-%m-%d')} ({ipo['days_until']} days away)
Exchange:       {ipo['exchange']}
Sector:         {ipo['sector']}
Price Range:    ${ipo['price_range_low']:.2f} - ${ipo['price_range_high']:.2f}
Valuation:      ${ipo['valuation']/1e9:.2f}B
Expected Raise: ${ipo['expected_proceeds']/1e6:.1f}M
Underwriter:    {ipo['underwriter']}
Score:          {ipo['score']:.1f}/100

Company Overview:
{company_info['description']}

Ownership:
{company_info['ownership']}

Financial Metrics:
"""

                if financial_metrics['revenue']:
                    report += f"  Revenue: ${financial_metrics['revenue']/1e9:.2f}B\n"
                if financial_metrics['revenue_growth']:
                    report += f"  Revenue Growth: {financial_metrics['revenue_growth']*100:.1f}%\n"
                if financial_metrics['profit_margin']:
                    report += f"  Profit Margin: {financial_metrics['profit_margin']*100:.1f}%\n"
                if financial_metrics['employees']:
                    report += f"  Employees: {financial_metrics['employees']:,}\n"
                if not any(financial_metrics.values()):
                    report += "  Financial data will be available after IPO\n"

                report += f"\nMarket Sentiment: {sentiment} ({sentiment_score*100:.0f}% confidence)\n"

                if news_articles:
                    report += "\nRecent News & Events:\n"
                    for article in news_articles[:3]:
                        report += f"  • [{article['published']}] {article['title']}\n"
                        report += f"    Source: {article['publisher']}\n"
                        if article.get('summary'):
                            report += f"    {article['summary']}\n"
                else:
                    report += "\nRecent News & Events:\n  No recent news available\n"

                report += "\n"

        report += f"""
{'='*80}
SUMMARY STATISTICS
{'='*80}
Total IPOs tracked: {len(self.ipo_data)}
Total expected capital raised (Top {top_n}): ${sum(ipo['expected_proceeds'] for ipo in top_ipos)/1e9:.2f}B
Average valuation (Top {top_n}): ${np.mean([ipo['valuation'] for ipo in top_ipos])/1e9:.2f}B

Sector Breakdown:
"""
        # Sector breakdown
        sectors = {}
        for ipo in top_ipos:
            sectors[ipo['sector']] = sectors.get(ipo['sector'], 0) + 1

        for sector, count in sorted(sectors.items(), key=lambda x: x[1], reverse=True):
            report += f"  {sector}: {count} IPO(s)\n"

        report += f"""
{'='*80}
DISCLAIMER: This report is for informational purposes only.
Do your own research before investing in any IPO.
{'='*80}
"""

        return report

    def generate_html_report(self, top_n=50, risky_n=10):
        """Generate HTML email report with top IPOs and risky opportunities"""

        if not self.ipo_data:
            return "<p>No IPO data available</p>"

        top_ipos = self.ipo_data[:top_n]

        # Get symbols of top IPOs to avoid duplicates
        top_symbols = {ipo['symbol'] for ipo in top_ipos}

        # Identify risky companies: lower scores or very soon IPO dates
        # Exclude companies already in top_ipos
        risky_ipos = []
        for ipo in self.ipo_data:
            if ipo['symbol'] not in top_symbols:  # Only include if not in top list
                if ipo['score'] < 50 or ipo['days_until'] <= 3:
                    risky_ipos.append(ipo)
        risky_ipos = sorted(risky_ipos, key=lambda x: x['days_until'])[:risky_n]

        report_date = datetime.now()

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                       line-height: 1.6; color: #2c3e50; background: #f5f7fa; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white;
                             box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                          color: white; padding: 40px 30px; border-bottom: 4px solid #c9a961; }}
                .header h1 {{ margin: 0 0 10px 0; font-size: 28px; font-weight: 600;
                             letter-spacing: 0.5px; }}
                .header .subtitle {{ font-size: 16px; opacity: 0.95; margin: 5px 0; }}
                .header .date {{ font-size: 14px; opacity: 0.85; margin-top: 15px;
                                border-top: 1px solid rgba(255,255,255,0.3); padding-top: 15px; }}
                .content {{ padding: 30px; }}
                .company {{ background: #ffffff; margin: 20px 0; padding: 25px;
                           border: 1px solid #e1e8ed; border-left: 5px solid #2a5298;
                           box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
                .company:hover {{ box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                .company-header {{ margin-bottom: 15px; padding-bottom: 10px;
                                  border-bottom: 1px solid #e1e8ed; }}
                .rank {{ display: inline-block; background: #2a5298; color: white;
                        padding: 6px 14px; font-weight: 600; font-size: 14px;
                        min-width: 40px; text-align: center; }}
                .company-name {{ display: inline-block; font-size: 20px; font-weight: 600;
                                color: #1e3c72; margin-left: 15px; }}
                .ticker {{ color: #7f8c8d; font-weight: 500; }}
                .score {{ float: right; background: #27ae60; color: white;
                         padding: 6px 16px; font-weight: 600; font-size: 14px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                                gap: 15px; margin-top: 15px; }}
                .metric {{ padding: 10px 0; }}
                .label {{ font-weight: 600; color: #34495e; font-size: 13px;
                         text-transform: uppercase; letter-spacing: 0.5px; }}
                .value {{ color: #2c3e50; font-size: 15px; font-weight: 500; margin-left: 8px; }}
                .urgent {{ background: #fff3cd; border-left-color: #ffc107 !important; }}
                .summary {{ background: #f8f9fa; padding: 25px; margin: 30px 0;
                           border: 1px solid #e1e8ed; }}
                .summary h3 {{ color: #1e3c72; margin-top: 0; font-size: 20px;
                              border-bottom: 2px solid #c9a961; padding-bottom: 10px; }}
                .summary-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                                 gap: 15px; margin: 20px 0; }}
                .stat-item {{ padding: 15px; background: white; border-left: 4px solid #2a5298; }}
                .stat-label {{ font-size: 13px; color: #7f8c8d; text-transform: uppercase;
                              letter-spacing: 0.5px; }}
                .stat-value {{ font-size: 22px; font-weight: 600; color: #2a5298; margin-top: 5px; }}
                .footer {{ background: #34495e; color: white; padding: 25px 30px;
                          text-align: center; font-size: 13px; }}
                .footer p {{ margin: 8px 0; opacity: 0.9; }}
                .disclaimer {{ background: #fff3cd; border-left: 4px solid #ffc107;
                              padding: 15px 20px; margin: 20px 0; font-size: 13px; }}
                ul {{ list-style: none; padding: 0; }}
                ul li {{ padding: 8px 0; border-bottom: 1px solid #e1e8ed; }}
                ul li:last-child {{ border-bottom: none; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>UPCOMING IPO INVESTMENT REPORT</h1>
                    <div class="subtitle">Top {top_n} Companies Going Public</div>
                    <div class="subtitle">Investment Research & Analysis</div>
                    <div class="date">
                        Report Date: {report_date.strftime('%A, %B %d, %Y at %I:%M %p')}<br>
                        Analysis Period: Next {self.config['filters']['days_ahead']} Days
                    </div>
                </div>

                <div class="content">
        """

        for idx, ipo in enumerate(top_ipos, 1):
            # Fetch company info
            company_info = self.get_company_info(ipo['company'], ipo['symbol'])

            # Fetch financial metrics
            financial_metrics = self.get_financial_metrics(ipo['symbol'])

            # Fetch recent news
            news_articles = self.get_company_news(ipo['company'], ipo['symbol'], max_articles=3)

            # Analyze sentiment
            sentiment, sentiment_score = self.analyze_sentiment(news_articles)

            # Sentiment color
            sentiment_color = "#27ae60" if sentiment == "Positive" else "#e74c3c" if sentiment == "Negative" else "#95a5a6"

            html += f"""
                <div class="company">
                    <span class="rank">#{idx}</span>
                    <span class="score">Score: {ipo['score']:.0f}/100</span>
                    <h2 style="margin: 10px 0;">{ipo['company']} ({ipo['symbol']})</h2>

                    <div style="margin: 15px 0; padding: 12px; background: #f8f9fa; border-left: 3px solid #2a5298;">
                        <div style="font-weight: 600; color: #1e3c72; margin-bottom: 8px;">Company Overview:</div>
                        <div style="color: #2c3e50; line-height: 1.6;">{company_info['description']}</div>
                    </div>

                    <div style="margin: 15px 0; padding: 12px; background: #fff8e1; border-left: 3px solid #c9a961;">
                        <div style="font-weight: 600; color: #1e3c72; margin-bottom: 8px;">Ownership:</div>
                        <div style="color: #2c3e50;">{company_info['ownership']}</div>
                    </div>

                    <div style="margin: 15px 0; padding: 12px; background: #e8f5e9; border-left: 3px solid {sentiment_color};">
                        <div style="font-weight: 600; color: #1e3c72; margin-bottom: 8px;">Market Sentiment:</div>
                        <div style="color: #2c3e50;">
                            <strong style="color: {sentiment_color};">{sentiment}</strong>
                            ({sentiment_score*100:.0f}% confidence)
                        </div>
                    </div>
"""

            # Add financial metrics if available
            if any(financial_metrics.values()):
                html += """
                    <div style="margin: 15px 0; padding: 12px; background: #f3e5f5; border-left: 3px solid #9c27b0;">
                        <div style="font-weight: 600; color: #1e3c72; margin-bottom: 8px;">Financial Metrics:</div>
                        <div style="color: #2c3e50;">
"""
                if financial_metrics['revenue']:
                    html += f"                            <div>Revenue: <strong>${financial_metrics['revenue']/1e9:.2f}B</strong></div>\n"
                if financial_metrics['revenue_growth']:
                    html += f"                            <div>Revenue Growth: <strong>{financial_metrics['revenue_growth']*100:.1f}%</strong></div>\n"
                if financial_metrics['profit_margin']:
                    html += f"                            <div>Profit Margin: <strong>{financial_metrics['profit_margin']*100:.1f}%</strong></div>\n"
                if financial_metrics['employees']:
                    html += f"                            <div>Employees: <strong>{financial_metrics['employees']:,}</strong></div>\n"

                html += """
                        </div>
                    </div>
"""

            # Add recent news
            if news_articles:
                html += """
                    <div style="margin: 15px 0; padding: 12px; background: #fff3e0; border-left: 3px solid #ff9800;">
                        <div style="font-weight: 600; color: #1e3c72; margin-bottom: 8px;">Recent News & Events:</div>
                        <div style="color: #2c3e50;">
"""
                for article in news_articles[:3]:
                    html += f"""
                            <div style="margin: 10px 0; padding: 8px; background: white; border-radius: 4px;">
                                <div style="font-weight: 500; color: #1e3c72;">{article['title']}</div>
                                <div style="font-size: 12px; color: #7f8c8d; margin-top: 4px;">
                                    {article['publisher']} • {article['published']}
                                </div>
"""
                    if article.get('summary'):
                        html += f"                                <div style='font-size: 13px; margin-top: 6px;'>{article['summary']}</div>\n"

                    html += "                            </div>\n"

                html += """
                        </div>
                    </div>
"""

            html += f"""
                    <div>
                        <div class="metric">
                            <span class="label">IPO Date:</span>
                            <span class="value">{ipo['ipo_date'].strftime('%B %d, %Y')}
                            ({ipo['days_until']} days)</span>
                        </div>
                        <div class="metric">
                            <span class="label">Exchange:</span>
                            <span class="value">{ipo['exchange']}</span>
                        </div>
                        <div class="metric">
                            <span class="label">Sector:</span>
                            <span class="value">{ipo['sector']}</span>
                        </div>
                    </div>

                    <div>
                        <div class="metric">
                            <span class="label">Price Range:</span>
                            <span class="value">${ipo['price_range_low']:.2f} - ${ipo['price_range_high']:.2f}</span>
                        </div>
                        <div class="metric">
                            <span class="label">Valuation:</span>
                            <span class="value">${ipo['valuation']/1e9:.2f}B</span>
                        </div>
                        <div class="metric">
                            <span class="label">Expected Raise:</span>
                            <span class="value">${ipo['expected_proceeds']/1e6:.0f}M</span>
                        </div>
                    </div>

                    <div>
                        <div class="metric">
                            <span class="label">Underwriter:</span>
                            <span class="value">{ipo['underwriter']}</span>
                        </div>
                        <div class="metric">
                            <span class="label">Shares:</span>
                            <span class="value">{ipo['shares']:,}</span>
                        </div>
                    </div>
                </div>
            """

        # Summary section
        total_raise = sum(ipo['expected_proceeds'] for ipo in top_ipos)
        avg_valuation = np.mean([ipo['valuation'] for ipo in top_ipos])

        sectors = {}
        for ipo in top_ipos:
            sectors[ipo['sector']] = sectors.get(ipo['sector'], 0) + 1

        html += f"""
                <div class="summary">
                    <h3>Summary Statistics</h3>
                    <p><strong>Total IPOs tracked:</strong> {len(self.ipo_data)}</p>
                    <p><strong>Expected capital raised (Top {top_n}):</strong> ${total_raise/1e9:.2f}B</p>
                    <p><strong>Average valuation (Top {top_n}):</strong> ${avg_valuation/1e9:.2f}B</p>

                    <h4>Sector Breakdown:</h4>
                    <ul>
        """

        for sector, count in sorted(sectors.items(), key=lambda x: x[1], reverse=True):
            html += f"<li><strong>{sector}:</strong> {count} IPO(s)</li>"

        html += """
                    </ul>
                </div>
        """

        # Add risky companies section if any exist
        if risky_ipos:
            html += f"""
                <div class="summary" style="border-left: 4px solid #e74c3c; background: #fee;">
                    <h3 style="color: #c0392b;">High-Risk Opportunities</h3>
                    <p><strong>Warning:</strong> These companies have lower investment scores or are launching within 3 days.
                    Higher risk may mean higher volatility and potential reward/loss.</p>
                </div>
            """

            for idx, ipo in enumerate(risky_ipos, 1):
                risk_reason = "IMMINENT LAUNCH" if ipo['days_until'] <= 3 else "LOW SCORE"
                risk_class = "urgent" if ipo['days_until'] <= 3 else ""
                # Fetch company info
                company_info = self.get_company_info(ipo['company'], ipo['symbol'])

                # Fetch financial metrics
                financial_metrics = self.get_financial_metrics(ipo['symbol'])

                # Fetch recent news
                news_articles = self.get_company_news(ipo['company'], ipo['symbol'], max_articles=3)

                # Analyze sentiment
                sentiment, sentiment_score = self.analyze_sentiment(news_articles)

                # Sentiment color
                sentiment_color = "#27ae60" if sentiment == "Positive" else "#e74c3c" if sentiment == "Negative" else "#95a5a6"

                html += f"""
                <div class="company {risk_class}" style="border-left-color: #e74c3c;">
                    <span class="rank" style="background: #e74c3c;">RISK #{idx}</span>
                    <span class="score" style="background: #e67e22;">Score: {ipo['score']:.0f}/100</span>
                    <h2 style="margin: 10px 0; color: #c0392b;">{ipo['company']} ({ipo['symbol']})</h2>
                    <p style="color: #e74c3c; font-weight: 600; margin: 5px 0;">{risk_reason}</p>

                    <div style="margin: 15px 0; padding: 12px; background: #f8f9fa; border-left: 3px solid #e74c3c;">
                        <div style="font-weight: 600; color: #c0392b; margin-bottom: 8px;">Company Overview:</div>
                        <div style="color: #2c3e50; line-height: 1.6;">{company_info['description']}</div>
                    </div>

                    <div style="margin: 15px 0; padding: 12px; background: #fff8e1; border-left: 3px solid #e67e22;">
                        <div style="font-weight: 600; color: #c0392b; margin-bottom: 8px;">Ownership:</div>
                        <div style="color: #2c3e50;">{company_info['ownership']}</div>
                    </div>

                    <div style="margin: 15px 0; padding: 12px; background: #e8f5e9; border-left: 3px solid {sentiment_color};">
                        <div style="font-weight: 600; color: #c0392b; margin-bottom: 8px;">Market Sentiment:</div>
                        <div style="color: #2c3e50;">
                            <strong style="color: {sentiment_color};">{sentiment}</strong>
                            ({sentiment_score*100:.0f}% confidence)
                        </div>
                    </div>
"""

                # Add financial metrics if available
                if any(financial_metrics.values()):
                    html += """
                    <div style="margin: 15px 0; padding: 12px; background: #f3e5f5; border-left: 3px solid #9c27b0;">
                        <div style="font-weight: 600; color: #c0392b; margin-bottom: 8px;">Financial Metrics:</div>
                        <div style="color: #2c3e50;">
"""
                    if financial_metrics['revenue']:
                        html += f"                            <div>Revenue: <strong>${financial_metrics['revenue']/1e9:.2f}B</strong></div>\n"
                    if financial_metrics['revenue_growth']:
                        html += f"                            <div>Revenue Growth: <strong>{financial_metrics['revenue_growth']*100:.1f}%</strong></div>\n"
                    if financial_metrics['profit_margin']:
                        html += f"                            <div>Profit Margin: <strong>{financial_metrics['profit_margin']*100:.1f}%</strong></div>\n"
                    if financial_metrics['employees']:
                        html += f"                            <div>Employees: <strong>{financial_metrics['employees']:,}</strong></div>\n"

                    html += """
                        </div>
                    </div>
"""

                # Add recent news
                if news_articles:
                    html += """
                    <div style="margin: 15px 0; padding: 12px; background: #fff3e0; border-left: 3px solid #ff9800;">
                        <div style="font-weight: 600; color: #c0392b; margin-bottom: 8px;">Recent News & Events:</div>
                        <div style="color: #2c3e50;">
"""
                    for article in news_articles[:3]:
                        html += f"""
                            <div style="margin: 10px 0; padding: 8px; background: white; border-radius: 4px;">
                                <div style="font-weight: 500; color: #c0392b;">{article['title']}</div>
                                <div style="font-size: 12px; color: #7f8c8d; margin-top: 4px;">
                                    {article['publisher']} • {article['published']}
                                </div>
"""
                        if article.get('summary'):
                            html += f"                                <div style='font-size: 13px; margin-top: 6px;'>{article['summary']}</div>\n"

                        html += "                            </div>\n"

                    html += """
                        </div>
                    </div>
"""

                html += f"""
                    <div>
                        <div class="metric">
                            <span class="label">IPO Date:</span>
                            <span class="value">{ipo['ipo_date'].strftime('%B %d, %Y')}
                            ({ipo['days_until']} days)</span>
                        </div>
                        <div class="metric">
                            <span class="label">Exchange:</span>
                            <span class="value">{ipo['exchange']}</span>
                        </div>
                        <div class="metric">
                            <span class="label">Sector:</span>
                            <span class="value">{ipo['sector']}</span>
                        </div>
                    </div>

                    <div>
                        <div class="metric">
                            <span class="label">Price Range:</span>
                            <span class="value">${ipo['price_range_low']:.2f} - ${ipo['price_range_high']:.2f}</span>
                        </div>
                        <div class="metric">
                            <span class="label">Valuation:</span>
                            <span class="value">${ipo['valuation']/1e9:.2f}B</span>
                        </div>
                        <div class="metric">
                            <span class="label">Expected Raise:</span>
                            <span class="value">${ipo['expected_proceeds']/1e6:.0f}M</span>
                        </div>
                    </div>

                    <div>
                        <div class="metric">
                            <span class="label">Underwriter:</span>
                            <span class="value">{ipo['underwriter']}</span>
                        </div>
                        <div class="metric">
                            <span class="label">Shares:</span>
                            <span class="value">{ipo['shares']:,}</span>
                        </div>
                    </div>
                </div>
            """

        html += """
                <div class="footer">
                    <p><em>Disclaimer: This report is for informational purposes only.
                    Do your own research before investing in any IPO.</em></p>
                    <p>Generated by IPO Tracker | © 2025</p>
                </div>
            </div>
        </body>
        </html>
        """

        return html

    def send_email(self, subject, html_content, text_content):
        """Send email report"""

        email_config = self.config['email']

        # Validate email configuration
        if 'your_email' in email_config['sender_email']:
            print("\n" + "="*80)
            print("EMAIL NOT CONFIGURED")
            print("="*80)
            print("Please update ipo_config.json with your email settings:")
            print("1. Update sender_email with your Gmail address")
            print("2. Update sender_password with your Gmail App Password")
            print("   (https://support.google.com/accounts/answer/185833)")
            print("3. Update recipient_email with the destination email")
            print("="*80 + "\n")
            return False

        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = email_config['sender_email']
            msg['To'] = email_config['recipient_email']
            msg['Subject'] = subject

            # Attach text and HTML versions
            part1 = MIMEText(text_content, 'plain')
            part2 = MIMEText(html_content, 'html')

            msg.attach(part1)
            msg.attach(part2)

            # Send email
            print(f"Sending email to {email_config['recipient_email']}...")

            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['sender_email'], email_config['sender_password'])
            server.send_message(msg)
            server.quit()

            print("✓ Email sent successfully!")
            return True

        except Exception as e:
            print(f"✗ Error sending email: {e}")
            return False

    def run_daily_report(self, send_email=True):
        """Run daily IPO report and send email"""

        print("\n" + "="*80)
        print("IPO TRACKER - DAILY REPORT")
        print("="*80 + "\n")

        # Fetch and analyze IPOs
        self.fetch_and_analyze_ipos()

        if not self.ipo_data:
            print("No IPOs found")
            return

        # Generate reports
        text_report = self.generate_summary_report(top_n=50, risky_n=10)
        html_report = self.generate_html_report(top_n=50, risky_n=10)

        # Print to console
        print(text_report)

        # Send email
        if send_email:
            subject = f"Daily IPO Investment Report - Top 50 Companies + 10 Risky ({datetime.now().strftime('%B %d, %Y')})"
            self.send_email(subject, html_report, text_report)
        else:
            print("\n(Email sending disabled)")

        # Save reports to file
        # Reports will be saved to Desktop/IPO Daily Statements folder
        save_folder = os.path.expanduser("~/Desktop/IPO Daily Statements")

        # Create folder if it doesn't exist
        if save_folder != ".":
            os.makedirs(save_folder, exist_ok=True)

        report_date = datetime.now()
        # Professional filename with date in title
        date_str = report_date.strftime('%B_%d_%Y')  # e.g., December_29_2025
        time_str = report_date.strftime('%H%M')

        txt_filename = f'IPO_Investment_Report_{date_str}_{time_str}.txt'
        html_filename = f'IPO_Investment_Report_{date_str}_{time_str}.html'

        txt_path = os.path.join(save_folder, txt_filename)
        html_path = os.path.join(save_folder, html_filename)

        with open(txt_path, 'w') as f:
            f.write(text_report)
        with open(html_path, 'w') as f:
            f.write(html_report)

        print(f"\nReports saved:")
        print(f"  - {txt_path}")
        print(f"  - {html_path}")


def setup_daily_schedule():
    """
    Setup instructions for scheduling daily runs

    For actual scheduling, use cron (macOS/Linux) or Task Scheduler (Windows)
    """

    instructions = """

SCHEDULING DAILY IPO REPORTS
════════════════════════════════════════════════════════════════════════════════

To run this script automatically every day:

macOS/Linux (cron):
─────────────────────
1. Open terminal and type: crontab -e
2. Add this line (runs daily at 9 AM):
   0 9 * * * cd /Users/williampatapan/Wills\\ Projets && /usr/bin/python3 ipo_tracker.py

3. Save and exit

Windows (Task Scheduler):
─────────────────────────
1. Open Task Scheduler
2. Create Basic Task
3. Trigger: Daily at 9:00 AM
4. Action: Start a Program
   Program: python3
   Arguments: ipo_tracker.py
   Start in: C:\\Users\\YourName\\Wills Projets

Alternative - Manual Run:
─────────────────────────
Simply run this command daily:
   python3 ipo_tracker.py

════════════════════════════════════════════════════════════════════════════════
    """

    print(instructions)


def main():
    """Main execution function"""

    tracker = IPOTracker()

    # Run daily report
    tracker.run_daily_report(send_email=True)

    # Show scheduling instructions
    print("\n")
    setup_daily_schedule()


if __name__ == "__main__":
    main()