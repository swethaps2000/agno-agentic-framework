# agents/plot_agent.py
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from db.db_setup import setup_mongo
from agents.base_agent import agent
import re

@agent(name="PlotAgent")
class PlotAgent:
    def __init__(self):
        self.mongo_colls = setup_mongo()
        self.plots_dir = './plots'
        os.makedirs(self.plots_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def run(self, user_id: str, query: str, want_plot: bool) -> Dict:
        if not want_plot:
            return {'plot_path': None, 'plot_description': None}
        
        try:
            expense_docs = list(self.mongo_colls['expense_coll'].find({'user_id': user_id}))
            income_docs = list(self.mongo_colls['income_coll'].find({'user_id': user_id}))
            
            if not expense_docs and not income_docs:
                return {'plot_path': None, 'plot_description': "No transaction data available for plotting"}
            
            plot_analysis = self._analyze_query_for_plot(query)
            
            plot_path, plot_desc = self._create_smart_plot(
                user_id, 
                query, 
                expense_docs, 
                income_docs, 
                plot_analysis
            )
            
            return {
                'plot_path': f'/plots/{plot_path.split("/")[-1]}',
                'plot_description': plot_desc
            }
            
        except Exception as e:
            print(f"Plot generation failed: {e}")
            return {'plot_path': None, 'plot_description': f"Plot generation failed: {str(e)}"}
    
    def _analyze_query_for_plot(self, query: str) -> Dict:
        query_lower = query.lower()
        
        time_patterns = {
            'monthly': r'\b(month|monthly|last\s+\d+\s+months?)\b',
            'weekly': r'\b(week|weekly|last\s+\d+\s+weeks?)\b',
            'yearly': r'\b(year|yearly|annual|last\s+\d+\s+years?)\b',
            'daily': r'\b(day|daily|last\s+\d+\s+days?)\b'
        }
        
        time_period = None
        time_value = None
        for period, pattern in time_patterns.items():
            if re.search(pattern, query_lower):
                time_period = period
                number_match = re.search(r'(\d+)\s+' + period.rstrip('ly'), query_lower)
                if number_match:
                    time_value = int(number_match.group(1))
                break
        
        plot_type = 'category_bar'
        if any(word in query_lower for word in ['trend', 'time', 'over time', 'pattern']):
            plot_type = 'time_series'
        elif any(word in query_lower for word in ['pie', 'proportion', 'percentage']):
            plot_type = 'pie_chart'
        elif any(word in query_lower for word in ['income', 'expense', 'balance']):
            plot_type = 'income_vs_expense'
        
        categories = []
        category_keywords = ['groceries', 'rent', 'fuel', 'food', 'bills', 'shopping', 'transport', 'medical', 'investment', 'salary', 'entertainment']
        for cat in category_keywords:
            if cat in query_lower:
                categories.append(cat)
        
        return {
            'plot_type': plot_type,
            'time_period': time_period,
            'time_value': time_value,
            'categories': categories
        }
    
    def _create_smart_plot(self, user_id: str, query: str, expense_docs: List[Dict], income_docs: List[Dict], analysis: Dict) -> Tuple[str, str]:
        expense_df = pd.DataFrame(expense_docs) if expense_docs else pd.DataFrame()
        income_df = pd.DataFrame(income_docs) if income_docs else pd.DataFrame()
        
        if not expense_df.empty:
            expense_df = self._filter_data_by_query(expense_df, analysis)
        if not income_df.empty:
            income_df = self._filter_data_by_query(income_df, analysis)
        
        plot_type = analysis['plot_type']
        timestamp = int(datetime.now().timestamp())
        
        try:
            if plot_type == 'time_series':
                path = self._plot_time_series(expense_df, user_id, analysis, timestamp)
                desc = f"Time series showing spending over {analysis.get('time_period', 'time')}"
            elif plot_type == 'pie_chart':
                path = self._plot_pie_chart(expense_df, user_id, timestamp)
                desc = "Pie chart showing spending distribution by category"
            elif plot_type == 'income_vs_expense':
                path = self._plot_income_vs_expense(income_df, expense_df, user_id, timestamp)
                desc = "Income vs expense comparison with net savings/loss"
            else:
                path = self._plot_category_breakdown(expense_df, user_id, analysis, timestamp)
                desc = "Category breakdown of spending"
            
            return path, desc
            
        except Exception as e:
            print(f"Plot creation failed: {e}")
            path = self._create_empty_plot(user_id, f"Error: {str(e)}", timestamp)
            return path, "Error generating plot"
    
    def _filter_data_by_query(self, df: pd.DataFrame, analysis: Dict) -> pd.DataFrame:
        if df.empty:
            return df
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        if analysis['time_period'] and analysis['time_value']:
            end_date = datetime.now()
            if analysis['time_period'] == 'monthly':
                start_date = end_date - timedelta(days=30 * analysis['time_value'])
            elif analysis['time_period'] == 'weekly':
                start_date = end_date - timedelta(weeks=analysis['time_value'])
            elif analysis['time_period'] == 'yearly':
                start_date = end_date - timedelta(days=365 * analysis['time_value'])
            else:
                start_date = end_date - timedelta(days=analysis['time_value'])
            
            df = df[df['date'] >= start_date]
        
        if analysis['categories'] and 'category' in df.columns:
            df = df[df['category'].isin(analysis['categories'])]
        
        return df
    
    def _plot_category_breakdown(self, df: pd.DataFrame, user_id: str, analysis: Dict, timestamp: int) -> str:
        if df.empty:
            return self._create_empty_plot(user_id, "No expense data available", timestamp)
        
        agg = df.groupby('category')['amount'].sum().sort_values(ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(agg.index, agg.values, color=sns.color_palette("husl", len(agg)))
        
        ax.set_title('Spending by Category', fontsize=14, fontweight='bold')
        ax.set_ylabel('Amount (₹)', fontsize=12)
        ax.set_xlabel('Category', fontsize=12)
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'₹{height:,.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        path = os.path.join(self.plots_dir, f'{user_id}_category_{timestamp}.png')
        fig.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return path
    
    def _plot_time_series(self, df: pd.DataFrame, user_id: str, analysis: Dict, timestamp: int) -> str:
        if df.empty:
            return self._create_empty_plot(user_id, "No data for time series", timestamp)
        
        period_map = {
            'monthly': 'M',
            'weekly': 'W',
            'yearly': 'Y',
            'daily': 'D'
        }
        period = period_map.get(analysis.get('time_period'), 'D')
        
        df['period'] = df['date'].dt.to_period(period)
        grouped = df.groupby('period')['amount'].sum().reset_index()
        grouped['period_str'] = grouped['period'].astype(str)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(grouped['period_str'], grouped['amount'], marker='o', linewidth=2, markersize=6)
        
        ax.set_title(f'{analysis.get("time_period", "Daily").title()} Spending Pattern', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Period', fontsize=12)
        ax.set_ylabel('Amount (₹)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        path = os.path.join(self.plots_dir, f'{user_id}_timeseries_{timestamp}.png')
        fig.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return path
    
    def _plot_pie_chart(self, df: pd.DataFrame, user_id: str, timestamp: int) -> str:
        if df.empty:
            return self._create_empty_plot(user_id, "No data for pie chart", timestamp)
        
        agg = df.groupby('category')['amount'].sum().sort_values(ascending=False)
        
        if len(agg) > 8:
            top_categories = agg.head(8)
            others_sum = agg.tail(len(agg) - 8).sum()
            if others_sum > 0:
                top_categories['Others'] = others_sum
            agg = top_categories
        
        fig, ax = plt.subplots(figsize=(10, 10))
        colors = sns.color_palette("husl", len(agg))
        
        wedges, texts, autotexts = ax.pie(agg.values, labels=agg.index, autopct='%1.1f%%',
                                         colors=colors, startangle=90)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Spending Distribution by Category', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        path = os.path.join(self.plots_dir, f'{user_id}_pie_{timestamp}.png')
        fig.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return path
    
    def _plot_income_vs_expense(self, income_df: pd.DataFrame, expense_df: pd.DataFrame, 
                               user_id: str, timestamp: int) -> str:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        if not expense_df.empty:
            expense_df['month'] = pd.to_datetime(expense_df['date']).dt.to_period('M')
            expense_monthly = expense_df.groupby('month')['amount'].sum()
        else:
            expense_monthly = pd.Series(dtype=float)
        
        if not income_df.empty:
            income_df['month'] = pd.to_datetime(income_df['date']).dt.to_period('M')
            income_monthly = income_df.groupby('month')['amount'].sum()
        else:
            income_monthly = pd.Series(dtype=float)
        
        all_months = sorted(set(list(expense_monthly.index) + list(income_monthly.index)))
        
        if all_months:
            income_values = [income_monthly.get(month, 0) for month in all_months]
            expense_values = [expense_monthly.get(month, 0) for month in all_months]
            month_labels = [str(month) for month in all_months]
            
            x = np.arange(len(month_labels))
            width = 0.35
            
            ax1.bar(x - width/2, income_values, width, label='Income', color='green', alpha=0.7)
            ax1.bar(x + width/2, expense_values, width, label='Expense', color='red', alpha=0.7)
            
            ax1.set_title('Monthly Income vs Expense', fontweight='bold')
            ax1.set_xlabel('Month')
            ax1.set_ylabel('Amount (₹)')
            ax1.set_xticks(x)
            ax1.set_xticklabels(month_labels, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            net_values = [inc - exp for inc, exp in zip(income_values, expense_values)]
            colors = ['green' if x >= 0 else 'red' for x in net_values]
            
            ax2.bar(month_labels, net_values, color=colors, alpha=0.7)
            ax2.set_title('Monthly Savings/Loss', fontweight='bold')
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Net Amount (₹)')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(self.plots_dir, f'{user_id}_income_expense_{timestamp}.png')
        fig.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return path
    
    def _create_empty_plot(self, user_id: str, message: str, timestamp: int) -> str:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=16,
                transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax.set_title('No Data Available', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        path = os.path.join(self.plots_dir, f'{user_id}_empty_{timestamp}.png')
        fig.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return path