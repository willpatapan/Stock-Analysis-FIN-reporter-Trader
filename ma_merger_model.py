#!/usr/bin/env python3
"""
M&A Merger Model - Accretion/Dilution Analysis
Analyzes the financial impact of mergers and acquisitions on earnings per share

CHANGELOG
=========
Version History             Author          Date
@changelog   1.0.0                  WP              29-12-2025

- Initial version: Comprehensive M&A model with accretion/dilution analysis,
  purchase price allocation, synergies, and pro forma financials
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class MergerModel:
    """
    Comprehensive M&A Merger Model for Accretion/Dilution Analysis

    Analyzes the financial impact of an acquisition on the acquirer's EPS
    """

    def __init__(self, acquirer_data, target_data, deal_structure):
        """
        Initialize merger model with company data and deal terms

        Parameters:
        -----------
        acquirer_data : dict
            Financial data for the acquiring company
        target_data : dict
            Financial data for the target company
        deal_structure : dict
            Deal terms and structure
        """
        self.acquirer = acquirer_data
        self.target = target_data
        self.deal = deal_structure

        # Results storage
        self.purchase_price_allocation = {}
        self.pro_forma = {}
        self.accretion_dilution = {}

    def calculate_purchase_price(self):
        """Calculate total purchase price and metrics"""

        # Base purchase price
        if self.deal['consideration_type'] == 'cash':
            total_consideration = self.deal['offer_price_per_share'] * self.target['shares_outstanding']
        elif self.deal['consideration_type'] == 'stock':
            exchange_ratio = self.deal['exchange_ratio']
            total_consideration = exchange_ratio * self.target['shares_outstanding'] * self.acquirer['stock_price']
        else:  # mixed
            cash_portion = self.deal['cash_per_share'] * self.target['shares_outstanding']
            stock_portion = self.deal['stock_exchange_ratio'] * self.target['shares_outstanding'] * self.acquirer['stock_price']
            total_consideration = cash_portion + stock_portion

        # Calculate premium
        target_market_cap = self.target['stock_price'] * self.target['shares_outstanding']
        premium_percent = ((total_consideration / target_market_cap) - 1) * 100

        # Calculate transaction multiples
        ev = total_consideration + self.target['net_debt']
        ev_revenue = ev / self.target['revenue']
        ev_ebitda = ev / self.target['ebitda']
        pe_ratio = total_consideration / self.target['net_income']

        self.purchase_price_allocation = {
            'total_consideration': total_consideration,
            'target_market_cap': target_market_cap,
            'premium_percent': premium_percent,
            'enterprise_value': ev,
            'ev_revenue_multiple': ev_revenue,
            'ev_ebitda_multiple': ev_ebitda,
            'pe_multiple': pe_ratio,
            'target_equity_value': self.target['stock_price'] * self.target['shares_outstanding']
        }

        return self.purchase_price_allocation

    def calculate_sources_and_uses(self):
        """Calculate sources and uses of funds"""

        # Uses
        equity_purchase = self.purchase_price_allocation['total_consideration']
        transaction_fees = equity_purchase * self.deal['transaction_fees_pct']
        refinance_target_debt = self.target.get('debt_to_refinance', 0)

        total_uses = equity_purchase + transaction_fees + refinance_target_debt

        # Sources
        if self.deal['consideration_type'] == 'cash':
            cash_from_balance_sheet = min(self.acquirer['cash'], total_uses * 0.3)
            new_debt = self.deal.get('new_debt_issued', 0)
            stock_issued = 0
            shares_issued = 0
        elif self.deal['consideration_type'] == 'stock':
            cash_from_balance_sheet = transaction_fees
            new_debt = 0
            stock_issued = equity_purchase
            shares_issued = (self.deal['exchange_ratio'] * self.target['shares_outstanding'])
        else:  # mixed
            cash_from_balance_sheet = self.deal['cash_per_share'] * self.target['shares_outstanding']
            new_debt = self.deal.get('new_debt_issued', 0)
            stock_issued = self.deal['stock_exchange_ratio'] * self.target['shares_outstanding'] * self.acquirer['stock_price']
            shares_issued = self.deal['stock_exchange_ratio'] * self.target['shares_outstanding']

        remaining_cash_needed = total_uses - cash_from_balance_sheet - new_debt - stock_issued
        if remaining_cash_needed > 0:
            new_debt += remaining_cash_needed

        total_sources = cash_from_balance_sheet + new_debt + stock_issued

        sources_uses = {
            'uses': {
                'equity_purchase': equity_purchase,
                'transaction_fees': transaction_fees,
                'refinance_debt': refinance_target_debt,
                'total_uses': total_uses
            },
            'sources': {
                'cash_on_hand': cash_from_balance_sheet,
                'new_debt': new_debt,
                'stock_issued': stock_issued,
                'shares_issued': shares_issued,
                'total_sources': total_sources
            }
        }

        return sources_uses

    def calculate_goodwill_intangibles(self):
        """Calculate goodwill and purchase price allocation"""

        purchase_price = self.purchase_price_allocation['total_consideration']

        # Identified intangible assets (typically 20-30% of purchase price)
        identifiable_intangibles = purchase_price * self.deal.get('intangibles_pct', 0.25)

        # Fair value adjustments
        ppe_markup = self.target.get('ppe', 0) * self.deal.get('ppe_markup_pct', 0.10)
        inventory_markdown = self.target.get('inventory', 0) * self.deal.get('inventory_markdown_pct', -0.05)

        # Calculate goodwill
        target_book_value = self.target.get('book_value', self.target.get('total_equity', 0))

        goodwill = (purchase_price - target_book_value - identifiable_intangibles -
                   ppe_markup - inventory_markdown)

        allocation = {
            'purchase_price': purchase_price,
            'target_book_value': target_book_value,
            'identifiable_intangibles': identifiable_intangibles,
            'ppe_fair_value_adjustment': ppe_markup,
            'inventory_adjustment': inventory_markdown,
            'goodwill': goodwill,
            'goodwill_as_pct_of_price': (goodwill / purchase_price) * 100
        }

        return allocation

    def calculate_pro_forma_income_statement(self):
        """Calculate pro forma combined income statement"""

        # Revenue
        acquirer_revenue = self.acquirer['revenue']
        target_revenue = self.target['revenue']
        revenue_synergies = self.deal.get('revenue_synergies', 0)
        pro_forma_revenue = acquirer_revenue + target_revenue + revenue_synergies

        # EBITDA
        acquirer_ebitda = self.acquirer['ebitda']
        target_ebitda = self.target['ebitda']
        cost_synergies = self.deal.get('cost_synergies', 0)
        pro_forma_ebitda = acquirer_ebitda + target_ebitda + cost_synergies

        # Depreciation & Amortization
        acquirer_da = self.acquirer.get('depreciation_amortization', acquirer_ebitda * 0.15)
        target_da = self.target.get('depreciation_amortization', target_ebitda * 0.15)

        # Additional amortization of intangibles
        allocation = self.calculate_goodwill_intangibles()
        intangibles_amortization = allocation['identifiable_intangibles'] / self.deal.get('intangibles_life', 10)

        pro_forma_da = acquirer_da + target_da + intangibles_amortization

        # EBIT
        pro_forma_ebit = pro_forma_ebitda - pro_forma_da

        # Interest expense
        sources_uses = self.calculate_sources_and_uses()
        new_debt = sources_uses['sources']['new_debt']
        interest_rate = self.deal.get('debt_interest_rate', 0.05)
        additional_interest = new_debt * interest_rate

        acquirer_interest = self.acquirer.get('interest_expense', 0)
        target_interest = self.target.get('interest_expense', 0)
        pro_forma_interest = acquirer_interest + target_interest + additional_interest

        # Pre-tax income
        pro_forma_ebt = pro_forma_ebit - pro_forma_interest

        # Taxes
        tax_rate = self.deal.get('tax_rate', 0.25)
        pro_forma_taxes = pro_forma_ebt * tax_rate

        # Net income
        pro_forma_net_income = pro_forma_ebt - pro_forma_taxes

        # Shares outstanding
        acquirer_shares = self.acquirer['shares_outstanding']
        new_shares = sources_uses['sources']['shares_issued']
        pro_forma_shares = acquirer_shares + new_shares

        # EPS
        acquirer_eps = self.acquirer['net_income'] / acquirer_shares
        pro_forma_eps = pro_forma_net_income / pro_forma_shares

        # Accretion/Dilution
        eps_change = pro_forma_eps - acquirer_eps
        eps_change_pct = (eps_change / acquirer_eps) * 100

        self.pro_forma = {
            'revenue': {
                'acquirer': acquirer_revenue,
                'target': target_revenue,
                'synergies': revenue_synergies,
                'pro_forma': pro_forma_revenue
            },
            'ebitda': {
                'acquirer': acquirer_ebitda,
                'target': target_ebitda,
                'synergies': cost_synergies,
                'pro_forma': pro_forma_ebitda
            },
            'depreciation_amortization': {
                'acquirer': acquirer_da,
                'target': target_da,
                'intangibles_amortization': intangibles_amortization,
                'pro_forma': pro_forma_da
            },
            'ebit': {
                'acquirer': self.acquirer['ebitda'] - acquirer_da,
                'target': self.target['ebitda'] - target_da,
                'pro_forma': pro_forma_ebit
            },
            'interest_expense': {
                'acquirer': acquirer_interest,
                'target': target_interest,
                'new_debt_interest': additional_interest,
                'pro_forma': pro_forma_interest
            },
            'ebt': pro_forma_ebt,
            'taxes': pro_forma_taxes,
            'net_income': {
                'acquirer': self.acquirer['net_income'],
                'target': self.target['net_income'],
                'pro_forma': pro_forma_net_income
            },
            'shares_outstanding': {
                'acquirer': acquirer_shares,
                'new_shares_issued': new_shares,
                'pro_forma': pro_forma_shares
            },
            'eps': {
                'acquirer': acquirer_eps,
                'pro_forma': pro_forma_eps,
                'change': eps_change,
                'change_pct': eps_change_pct
            }
        }

        self.accretion_dilution = {
            'acquirer_eps': acquirer_eps,
            'pro_forma_eps': pro_forma_eps,
            'eps_change': eps_change,
            'eps_change_pct': eps_change_pct,
            'is_accretive': eps_change > 0
        }

        return self.pro_forma

    def sensitivity_analysis(self, variable='synergies', range_pct=0.5, steps=11):
        """
        Perform sensitivity analysis on key variables

        Parameters:
        -----------
        variable : str
            Variable to analyze ('synergies', 'purchase_price', 'interest_rate')
        range_pct : float
            Range as percentage of base value (+/-)
        steps : int
            Number of steps in the analysis
        """

        base_value = None
        results = []

        if variable == 'cost_synergies':
            base_value = self.deal.get('cost_synergies', 0)
            test_values = np.linspace(base_value * (1 - range_pct),
                                     base_value * (1 + range_pct), steps)

            for value in test_values:
                original = self.deal.get('cost_synergies', 0)
                self.deal['cost_synergies'] = value
                self.calculate_pro_forma_income_statement()
                results.append({
                    'value': value,
                    'eps': self.pro_forma['eps']['pro_forma'],
                    'eps_change_pct': self.pro_forma['eps']['change_pct']
                })
                self.deal['cost_synergies'] = original

        elif variable == 'purchase_price':
            base_value = self.deal.get('offer_price_per_share', 0)
            test_values = np.linspace(base_value * (1 - range_pct),
                                     base_value * (1 + range_pct), steps)

            for value in test_values:
                original = self.deal.get('offer_price_per_share', 0)
                self.deal['offer_price_per_share'] = value
                self.calculate_purchase_price()
                self.calculate_pro_forma_income_statement()
                results.append({
                    'value': value,
                    'eps': self.pro_forma['eps']['pro_forma'],
                    'eps_change_pct': self.pro_forma['eps']['change_pct']
                })
                self.deal['offer_price_per_share'] = original

        elif variable == 'interest_rate':
            base_value = self.deal.get('debt_interest_rate', 0.05)
            test_values = np.linspace(max(0.01, base_value - 0.05),
                                     base_value + 0.05, steps)

            for value in test_values:
                original = self.deal.get('debt_interest_rate', 0.05)
                self.deal['debt_interest_rate'] = value
                self.calculate_pro_forma_income_statement()
                results.append({
                    'value': value * 100,  # Convert to percentage
                    'eps': self.pro_forma['eps']['pro_forma'],
                    'eps_change_pct': self.pro_forma['eps']['change_pct']
                })
                self.deal['debt_interest_rate'] = original

        # Reset to base case
        self.calculate_purchase_price()
        self.calculate_pro_forma_income_statement()

        return pd.DataFrame(results)

    def generate_report(self):
        """Generate comprehensive merger analysis report"""

        print("\n" + "="*80)
        print("M&A MERGER MODEL - ACCRETION/DILUTION ANALYSIS")
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

        # Deal Overview
        print("\nDEAL OVERVIEW:")
        print(f"Consideration Type: {self.deal['consideration_type'].upper()}")
        if self.deal['consideration_type'] == 'cash':
            print(f"Offer Price per Share: ${self.deal['offer_price_per_share']:.2f}")
        elif self.deal['consideration_type'] == 'stock':
            print(f"Exchange Ratio: {self.deal['exchange_ratio']:.4f}x")

        # Purchase Price
        pp = self.purchase_price_allocation
        print(f"\nPURCHASE PRICE ANALYSIS:")
        print(f"Target Market Cap: ${pp['target_market_cap']/1e6:,.1f}M")
        print(f"Total Consideration: ${pp['total_consideration']/1e6:,.1f}M")
        print(f"Premium to Market: {pp['premium_percent']:.1f}%")
        print(f"Enterprise Value: ${pp['enterprise_value']/1e6:,.1f}M")
        print(f"EV/Revenue Multiple: {pp['ev_revenue_multiple']:.2f}x")
        print(f"EV/EBITDA Multiple: {pp['ev_ebitda_multiple']:.2f}x")
        print(f"P/E Multiple: {pp['pe_multiple']:.2f}x")

        # Sources and Uses
        su = self.calculate_sources_and_uses()
        print(f"\nSOURCES AND USES OF FUNDS:")
        print(f"\nUses:")
        print(f"  Equity Purchase: ${su['uses']['equity_purchase']/1e6:,.1f}M")
        print(f"  Transaction Fees: ${su['uses']['transaction_fees']/1e6:,.1f}M")
        print(f"  Refinance Debt: ${su['uses']['refinance_debt']/1e6:,.1f}M")
        print(f"  Total Uses: ${su['uses']['total_uses']/1e6:,.1f}M")
        print(f"\nSources:")
        print(f"  Cash on Hand: ${su['sources']['cash_on_hand']/1e6:,.1f}M")
        print(f"  New Debt: ${su['sources']['new_debt']/1e6:,.1f}M")
        print(f"  Stock Issued: ${su['sources']['stock_issued']/1e6:,.1f}M")
        if su['sources']['shares_issued'] > 0:
            print(f"  Shares Issued: {su['sources']['shares_issued']/1e6:.2f}M")
        print(f"  Total Sources: ${su['sources']['total_sources']/1e6:,.1f}M")

        # Goodwill
        gw = self.calculate_goodwill_intangibles()
        print(f"\nPURCHASE PRICE ALLOCATION:")
        print(f"Target Book Value: ${gw['target_book_value']/1e6:,.1f}M")
        print(f"Identifiable Intangibles: ${gw['identifiable_intangibles']/1e6:,.1f}M")
        print(f"PPE Fair Value Adjustment: ${gw['ppe_fair_value_adjustment']/1e6:,.1f}M")
        print(f"Goodwill: ${gw['goodwill']/1e6:,.1f}M ({gw['goodwill_as_pct_of_price']:.1f}% of price)")

        # Pro Forma Income Statement
        pf = self.pro_forma
        print(f"\nPRO FORMA INCOME STATEMENT (in millions):")
        print(f"{'Item':<30} {'Acquirer':>12} {'Target':>12} {'Synergies':>12} {'Pro Forma':>12}")
        print("-" * 80)
        print(f"{'Revenue':<30} ${pf['revenue']['acquirer']/1e6:>11,.1f} ${pf['revenue']['target']/1e6:>11,.1f} ${pf['revenue']['synergies']/1e6:>11,.1f} ${pf['revenue']['pro_forma']/1e6:>11,.1f}")
        print(f"{'EBITDA':<30} ${pf['ebitda']['acquirer']/1e6:>11,.1f} ${pf['ebitda']['target']/1e6:>11,.1f} ${pf['ebitda']['synergies']/1e6:>11,.1f} ${pf['ebitda']['pro_forma']/1e6:>11,.1f}")
        print(f"{'D&A':<30} ${pf['depreciation_amortization']['acquirer']/1e6:>11,.1f} ${pf['depreciation_amortization']['target']/1e6:>11,.1f} ${pf['depreciation_amortization']['intangibles_amortization']/1e6:>11,.1f} ${pf['depreciation_amortization']['pro_forma']/1e6:>11,.1f}")
        print(f"{'EBIT':<30} ${pf['ebit']['acquirer']/1e6:>11,.1f} ${pf['ebit']['target']/1e6:>11,.1f} ${'':>11} ${pf['ebit']['pro_forma']/1e6:>11,.1f}")
        print(f"{'Interest Expense':<30} ${pf['interest_expense']['acquirer']/1e6:>11,.1f} ${pf['interest_expense']['target']/1e6:>11,.1f} ${pf['interest_expense']['new_debt_interest']/1e6:>11,.1f} ${pf['interest_expense']['pro_forma']/1e6:>11,.1f}")
        print(f"{'Pre-Tax Income':<30} ${'':>11} ${'':>11} ${'':>11} ${pf['ebt']/1e6:>11,.1f}")
        print(f"{'Taxes':<30} ${'':>11} ${'':>11} ${'':>11} ${pf['taxes']/1e6:>11,.1f}")
        print(f"{'Net Income':<30} ${pf['net_income']['acquirer']/1e6:>11,.1f} ${pf['net_income']['target']/1e6:>11,.1f} ${'':>11} ${pf['net_income']['pro_forma']/1e6:>11,.1f}")

        # EPS Analysis
        print(f"\n{'='*80}")
        print("ACCRETION/DILUTION ANALYSIS")
        print(f"{'='*80}")
        print(f"\nShares Outstanding:")
        print(f"  Acquirer Shares: {pf['shares_outstanding']['acquirer']/1e6:.2f}M")
        print(f"  New Shares Issued: {pf['shares_outstanding']['new_shares_issued']/1e6:.2f}M")
        print(f"  Pro Forma Shares: {pf['shares_outstanding']['pro_forma']/1e6:.2f}M")

        print(f"\nEarnings Per Share:")
        print(f"  Acquirer Standalone EPS: ${pf['eps']['acquirer']:.2f}")
        print(f"  Pro Forma EPS: ${pf['eps']['pro_forma']:.2f}")
        print(f"  EPS Change: ${pf['eps']['change']:+.2f}")
        print(f"  EPS Change %: {pf['eps']['change_pct']:+.2f}%")

        if self.accretion_dilution['is_accretive']:
            print(f"\n  RESULT: ACCRETIVE - The deal is {pf['eps']['change_pct']:.2f}% accretive to EPS")
        else:
            print(f"\n  RESULT: DILUTIVE - The deal is {abs(pf['eps']['change_pct']):.2f}% dilutive to EPS")

        print("\n" + "="*80)
        print("DISCLAIMER: This analysis is for educational purposes only.")
        print("Consult with financial advisors for actual M&A decisions.")
        print("="*80 + "\n")

    def visualize_analysis(self):
        """Create comprehensive visualizations"""

        print("Generating visualizations...")

        fig = plt.figure(figsize=(20, 12))

        # 1. EPS Comparison
        ax1 = plt.subplot(3, 3, 1)
        eps_data = [self.pro_forma['eps']['acquirer'], self.pro_forma['eps']['pro_forma']]
        colors = ['blue', 'green' if self.accretion_dilution['is_accretive'] else 'red']
        bars = ax1.bar(['Acquirer\nStandalone', 'Pro Forma\nCombined'], eps_data, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('EPS Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('EPS ($)')
        ax1.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.2f}', ha='center', va='bottom', fontweight='bold')

        # 2. Revenue Breakdown
        ax2 = plt.subplot(3, 3, 2)
        revenue_components = [
            self.pro_forma['revenue']['acquirer'],
            self.pro_forma['revenue']['target'],
            self.pro_forma['revenue']['synergies']
        ]
        labels = ['Acquirer', 'Target', 'Synergies']
        colors_rev = ['#3498db', '#e74c3c', '#2ecc71']
        ax2.pie(revenue_components, labels=labels, autopct='%1.1f%%', colors=colors_rev, startangle=90)
        ax2.set_title('Pro Forma Revenue Composition', fontsize=14, fontweight='bold')

        # 3. EBITDA Waterfall
        ax3 = plt.subplot(3, 3, 3)
        ebitda_values = [
            self.pro_forma['ebitda']['acquirer'],
            self.pro_forma['ebitda']['target'],
            self.pro_forma['ebitda']['synergies']
        ]
        cumulative = np.cumsum([0] + ebitda_values)
        ax3.bar(range(len(ebitda_values)), ebitda_values, bottom=cumulative[:-1],
               color=['blue', 'orange', 'green'], alpha=0.7, edgecolor='black')
        ax3.plot([0, len(ebitda_values)-1], [cumulative[1], cumulative[-1]], 'k--', linewidth=2)
        ax3.set_xticks(range(len(ebitda_values)))
        ax3.set_xticklabels(['Acquirer\nEBITDA', 'Target\nEBITDA', 'Cost\nSynergies'])
        ax3.set_title('EBITDA Waterfall', fontsize=14, fontweight='bold')
        ax3.set_ylabel('EBITDA ($M)')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Purchase Price Premium
        ax4 = plt.subplot(3, 3, 4)
        pp = self.purchase_price_allocation
        premium_data = [pp['target_market_cap'], pp['total_consideration'] - pp['target_market_cap']]
        colors_pp = ['#95a5a6', '#e67e22']
        ax4.bar(['Target\nMarket Cap', 'Premium\nPaid'], premium_data, color=colors_pp, alpha=0.7, edgecolor='black')
        ax4.set_title(f'Purchase Price Premium: {pp["premium_percent"]:.1f}%', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Value ($M)')
        ax4.grid(True, alpha=0.3, axis='y')

        # 5. Sensitivity Analysis - Cost Synergies
        ax5 = plt.subplot(3, 3, 5)
        synergy_sens = self.sensitivity_analysis('cost_synergies', range_pct=0.5, steps=11)
        ax5.plot(synergy_sens['value']/1e6, synergy_sens['eps_change_pct'],
                linewidth=2, marker='o', color='green')
        ax5.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Breakeven')
        ax5.set_title('Sensitivity: Cost Synergies', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Cost Synergies ($M)')
        ax5.set_ylabel('EPS Change (%)')
        ax5.grid(True, alpha=0.3)
        ax5.legend()

        # 6. Sensitivity Analysis - Purchase Price
        ax6 = plt.subplot(3, 3, 6)
        price_sens = self.sensitivity_analysis('purchase_price', range_pct=0.2, steps=11)
        ax6.plot(price_sens['value'], price_sens['eps_change_pct'],
                linewidth=2, marker='o', color='red')
        ax6.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Breakeven')
        ax6.set_title('Sensitivity: Purchase Price', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Offer Price per Share ($)')
        ax6.set_ylabel('EPS Change (%)')
        ax6.grid(True, alpha=0.3)
        ax6.legend()

        # 7. Sources and Uses
        ax7 = plt.subplot(3, 3, 7)
        su = self.calculate_sources_and_uses()
        sources_data = [
            su['sources']['cash_on_hand'],
            su['sources']['new_debt'],
            su['sources']['stock_issued']
        ]
        sources_labels = ['Cash', 'Debt', 'Stock']
        colors_sources = ['#27ae60', '#e74c3c', '#3498db']
        ax7.pie([s for s in sources_data if s > 0],
               labels=[l for s, l in zip(sources_data, sources_labels) if s > 0],
               autopct='%1.1f%%', colors=colors_sources, startangle=90)
        ax7.set_title('Sources of Funds', fontsize=14, fontweight='bold')

        # 8. Goodwill Breakdown
        ax8 = plt.subplot(3, 3, 8)
        gw = self.calculate_goodwill_intangibles()
        gw_data = [
            gw['target_book_value'],
            gw['identifiable_intangibles'],
            gw['goodwill']
        ]
        gw_labels = ['Book Value', 'Intangibles', 'Goodwill']
        colors_gw = ['#34495e', '#9b59b6', '#1abc9c']
        ax8.pie(gw_data, labels=gw_labels, autopct='%1.1f%%', colors=colors_gw, startangle=90)
        ax8.set_title('Purchase Price Allocation', fontsize=14, fontweight='bold')

        # 9. Accretion/Dilution Summary
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')

        pf = self.pro_forma
        pp = self.purchase_price_allocation

        summary_text = f"""
        M&A TRANSACTION SUMMARY
        ══════════════════════════════════════

        DEAL METRICS:
        Purchase Price:        ${pp['total_consideration']/1e6:,.1f}M
        Premium:               {pp['premium_percent']:.1f}%
        EV/EBITDA Multiple:    {pp['ev_ebitda_multiple']:.2f}x

        PRO FORMA RESULTS:
        Revenue:               ${pf['revenue']['pro_forma']/1e6:,.1f}M
        EBITDA:                ${pf['ebitda']['pro_forma']/1e6:,.1f}M
        Net Income:            ${pf['net_income']['pro_forma']/1e6:,.1f}M

        EPS IMPACT:
        Standalone EPS:        ${pf['eps']['acquirer']:.2f}
        Pro Forma EPS:         ${pf['eps']['pro_forma']:.2f}

        EPS Change:            {pf['eps']['change_pct']:+.2f}%

        RESULT: {'ACCRETIVE' if self.accretion_dilution['is_accretive'] else 'DILUTIVE'}
        """

        ax9.text(0.05, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center')

        plt.tight_layout()
        filename = 'ma_merger_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Visualization saved as '{filename}'")
        plt.show()


def example_merger():
    """Example merger scenario"""

    # Acquirer data (e.g., Tech Giant acquiring smaller competitor)
    acquirer = {
        'name': 'TechCorp',
        'stock_price': 150.00,
        'shares_outstanding': 100_000_000,  # 100M shares
        'revenue': 5_000_000_000,  # $5B
        'ebitda': 1_500_000_000,  # $1.5B
        'depreciation_amortization': 200_000_000,  # $200M
        'interest_expense': 50_000_000,  # $50M
        'net_income': 900_000_000,  # $900M
        'cash': 2_000_000_000,  # $2B
        'debt': 500_000_000,  # $500M
        'total_equity': 8_000_000_000  # $8B
    }

    # Target data
    target = {
        'name': 'InnovateCo',
        'stock_price': 50.00,
        'shares_outstanding': 20_000_000,  # 20M shares
        'revenue': 800_000_000,  # $800M
        'ebitda': 200_000_000,  # $200M
        'depreciation_amortization': 30_000_000,  # $30M
        'interest_expense': 10_000_000,  # $10M
        'net_income': 120_000_000,  # $120M
        'cash': 100_000_000,  # $100M
        'debt': 200_000_000,  # $200M
        'net_debt': 100_000_000,  # $100M (debt - cash)
        'book_value': 500_000_000,  # $500M
        'total_equity': 500_000_000,
        'ppe': 300_000_000,
        'inventory': 150_000_000
    }

    # Deal structure (Cash acquisition with 30% premium)
    deal = {
        'consideration_type': 'cash',  # 'cash', 'stock', or 'mixed'
        'offer_price_per_share': 65.00,  # 30% premium
        'transaction_fees_pct': 0.02,  # 2% transaction fees
        'new_debt_issued': 0,  # Will be calculated if needed
        'debt_interest_rate': 0.045,  # 4.5% interest rate on new debt

        # Synergies
        'cost_synergies': 50_000_000,  # $50M annual cost synergies
        'revenue_synergies': 30_000_000,  # $30M revenue synergies

        # Purchase price allocation
        'intangibles_pct': 0.30,  # 30% of purchase price as identifiable intangibles
        'intangibles_life': 10,  # 10-year amortization
        'ppe_markup_pct': 0.15,  # 15% markup on PP&E
        'inventory_markdown_pct': -0.03,  # 3% markdown on inventory

        # Tax
        'tax_rate': 0.25  # 25% tax rate
    }

    return acquirer, target, deal


def main():
    """Main execution function"""

    print("="*80)
    print("M&A MERGER MODEL - ACCRETION/DILUTION ANALYSIS")
    print("="*80)

    # Load example data
    acquirer, target, deal = example_merger()

    print(f"\nAnalyzing merger: {acquirer['name']} acquiring {target['name']}")
    print(f"Deal Type: {deal['consideration_type'].upper()}")

    # Create model
    model = MergerModel(acquirer, target, deal)

    # Run analysis
    model.calculate_purchase_price()
    model.calculate_pro_forma_income_statement()

    # Generate report
    model.generate_report()

    # Create visualizations
    model.visualize_analysis()

    print("\nAnalysis complete! Check 'ma_merger_analysis.png' for visualizations.")


if __name__ == "__main__":
    main()