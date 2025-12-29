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

try:
    import requests
    from bs4 import BeautifulSoup
    import yfinance as yf
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
        Fetch IPO calendar data from NASDAQ

        Note: This is a simplified example. In production, you'd want to
        use an API or more robust scraping method.
        """
        ipos = []

        try:
            # Example IPO data structure
            # In production, you would scrape from NASDAQ IPO calendar or use an API
            url = "https://www.nasdaq.com/market-activity/ipos"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            # For demonstration, we'll create sample data
            # In production, uncomment below to actually fetch
            # response = requests.get(url, headers=headers, timeout=10)
            # soup = BeautifulSoup(response.content, 'html.parser')

            # Sample data for demonstration
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

        # Identify risky companies: lower scores or very soon IPO dates
        risky_ipos = []
        for ipo in self.ipo_data:
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

        for idx, ipo in enumerate(top_ipos, 1):
            report += f"""
#{idx}. {ipo['company']} ({ipo['symbol']})
{'─'*80}
IPO Date:       {ipo['ipo_date'].strftime('%Y-%m-%d')} ({ipo['days_until']} days away)
Exchange:       {ipo['exchange']}
Sector:         {ipo['sector']}
Price Range:    ${ipo['price_range_low']} - ${ipo['price_range_high']}
Shares Offered: {ipo['shares']:,}
Valuation:      ${ipo['valuation']/1e9:.2f}B
Expected Raise: ${ipo['expected_proceeds']/1e6:.1f}M
Underwriter:    {ipo['underwriter']}
Score:          {ipo['score']:.1f}/100

"""

        # Add risky companies section
        if risky_ipos:
            report += f"""
{'='*80}
HIGH-RISK OPPORTUNITIES (Lower Scores or Imminent IPOs)
{'='*80}

These companies have lower investment scores or are launching within 3 days.
Higher risk may mean higher volatility and potential reward/loss.

"""
            for idx, ipo in enumerate(risky_ipos, 1):
                risk_reason = "IMMINENT LAUNCH" if ipo['days_until'] <= 3 else "LOW SCORE"
                report += f"""
RISK #{idx} - {risk_reason}: {ipo['company']} ({ipo['symbol']})
{'─'*80}
IPO Date:       {ipo['ipo_date'].strftime('%Y-%m-%d')} ({ipo['days_until']} days away)
Exchange:       {ipo['exchange']}
Sector:         {ipo['sector']}
Price Range:    ${ipo['price_range_low']} - ${ipo['price_range_high']}
Valuation:      ${ipo['valuation']/1e9:.2f}B
Expected Raise: ${ipo['expected_proceeds']/1e6:.1f}M
Underwriter:    {ipo['underwriter']}
Score:          {ipo['score']:.1f}/100

"""

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

        # Identify risky companies: lower scores or very soon IPO dates
        risky_ipos = []
        for ipo in self.ipo_data:
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
            html += f"""
                <div class="company">
                    <span class="rank">#{idx}</span>
                    <span class="score">Score: {ipo['score']:.0f}/100</span>
                    <h2 style="margin: 10px 0;">{ipo['company']} ({ipo['symbol']})</h2>

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
                            <span class="value">${ipo['price_range_low']} - ${ipo['price_range_high']}</span>
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
                html += f"""
                <div class="company {risk_class}" style="border-left-color: #e74c3c;">
                    <span class="rank" style="background: #e74c3c;">RISK #{idx}</span>
                    <span class="score" style="background: #e67e22;">Score: {ipo['score']:.0f}/100</span>
                    <h2 style="margin: 10px 0; color: #c0392b;">{ipo['company']} ({ipo['symbol']})</h2>
                    <p style="color: #e74c3c; font-weight: 600; margin: 5px 0;">{risk_reason}</p>

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
                            <span class="value">${ipo['price_range_low']} - ${ipo['price_range_high']}</span>
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