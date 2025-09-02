import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
import re
from typing import List, Dict, Tuple, Optional
import numpy as np

PLOTS_DIR = './plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def analyze_query_for_plot_type(query: str) -> Dict[str, any]:
    """
    Analyze the user query to determine what type of plot to generate
    """
    query_lower = query.lower()
    
    # Time-based queries
    time_patterns = {
        'monthly': r'\b(month|monthly|last\s+\d+\s+months?|past\s+\d+\s+months?)\b',
        'weekly': r'\b(week|weekly|last\s+\d+\s+weeks?|past\s+\d+\s+weeks?)\b',
        'yearly': r'\b(year|yearly|annual|last\s+\d+\s+years?|past\s+\d+\s+years?)\b',
        'daily': r'\b(day|daily|last\s+\d+\s+days?|past\s+\d+\s+days?)\b'
    }
    
    # Extract time period
    time_period = None
    time_value = None
    for period, pattern in time_patterns.items():
        if re.search(pattern, query_lower):
            time_period = period
            # Try to extract number
            number_match = re.search(r'(\d+)\s+' + period.rstrip('ly'), query_lower)
            if number_match:
                time_value = int(number_match.group(1))
            break
    
    # Plot type detection
    plot_type = 'category_bar'  # default
    
    if any(word in query_lower for word in ['trend', 'time', 'over time', 'pattern', 'timeline']):
        plot_type = 'time_series'
    elif any(word in query_lower for word in ['category', 'categories', 'breakdown', 'distribution']):
        plot_type = 'category_bar'
    elif any(word in query_lower for word in ['compare', 'comparison', 'vs', 'versus']):
        plot_type = 'comparison'
    elif any(word in query_lower for word in ['pie', 'proportion', 'percentage']):
        plot_type = 'pie_chart'
    elif any(word in query_lower for word in ['income', 'expense', 'balance']):
        plot_type = 'income_vs_expense'
    
    # Category detection
    categories = []
    category_keywords = ['groceries', 'rent', 'fuel', 'food', 'bills', 'shopping', 
                        'transport', 'medical', 'investment', 'salary', 'entertainment']
    for cat in category_keywords:
        if cat in query_lower:
            categories.append(cat)
    
    return {
        'plot_type': plot_type,
        'time_period': time_period,
        'time_value': time_value,
        'categories': categories,
        'query': query
    }

def filter_data_by_query(df: pd.DataFrame, query_analysis: Dict) -> pd.DataFrame:
    """
    Filter dataframe based on query analysis
    """
    if df.empty:
        return df
    
    # Convert date column to datetime if it's string
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Time-based filtering
    if query_analysis['time_period'] and query_analysis['time_value']:
        end_date = datetime.now()
        if query_analysis['time_period'] == 'monthly':
            start_date = end_date - timedelta(days=30 * query_analysis['time_value'])
        elif query_analysis['time_period'] == 'weekly':
            start_date = end_date - timedelta(weeks=query_analysis['time_value'])
        elif query_analysis['time_period'] == 'yearly':
            start_date = end_date - timedelta(days=365 * query_analysis['time_value'])
        elif query_analysis['time_period'] == 'daily':
            start_date = end_date - timedelta(days=query_analysis['time_value'])
        else:
            start_date = end_date - timedelta(days=90)  # default 3 months
        
        df = df[df['date'] >= start_date]
    
    # Category filtering
    if query_analysis['categories'] and 'category' in df.columns:
        df = df[df['category'].isin(query_analysis['categories'])]
    
    return df

def plot_time_series_enhanced(df: pd.DataFrame, user_id: str, query_analysis: Dict) -> str:
    """
    Enhanced time series plot based on query
    """
    if df.empty:
        return create_empty_plot(user_id, "No data found for the specified time period")
    
    # Group by date
    if query_analysis['time_period'] == 'monthly':
        df['period'] = df['date'].dt.to_period('M')
        title_period = "Monthly"
    elif query_analysis['time_period'] == 'weekly':
        df['period'] = df['date'].dt.to_period('W')
        title_period = "Weekly"
    elif query_analysis['time_period'] == 'yearly':
        df['period'] = df['date'].dt.to_period('Y')
        title_period = "Yearly"
    else:
        df['period'] = df['date'].dt.to_period('D')
        title_period = "Daily"
    
    grouped = df.groupby('period')['amount'].sum().reset_index()
    grouped['period_str'] = grouped['period'].astype(str)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(grouped['period_str'], grouped['amount'], marker='o', linewidth=2, markersize=6)
    
    time_desc = f"{query_analysis['time_value']} {query_analysis['time_period']}" if query_analysis['time_value'] else "Recent"
    ax.set_title(f'{title_period} Spending Pattern - {time_desc}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Period', fontsize=12)
    ax.set_ylabel('Amount (₹)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add value annotations on points
    for i, row in grouped.iterrows():
        ax.annotate(f'₹{row["amount"]:,.0f}', 
                   (row['period_str'], row['amount']), 
                   textcoords="offset points", 
                   xytext=(0,10), 
                   ha='center', fontsize=9)
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f'{user_id}_time_series_{int(datetime.now().timestamp())}.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return path

def plot_category_breakdown_enhanced(df: pd.DataFrame, user_id: str, query_analysis: Dict) -> str:
    """
    Enhanced category breakdown plot
    """
    if df.empty:
        return create_empty_plot(user_id, "No data found for the specified categories")
    
    # Group by category
    agg = df.groupby('category')['amount'].sum().sort_values(ascending=False)
    
    # Limit to top 10 categories for readability
    if len(agg) > 10:
        agg = agg.head(10)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(agg.index, agg.values, color=sns.color_palette("husl", len(agg)))
    
    time_desc = ""
    if query_analysis['time_value'] and query_analysis['time_period']:
        time_desc = f" - Last {query_analysis['time_value']} {query_analysis['time_period']}"
    
    ax.set_title(f'Spending by Category{time_desc}', fontsize=14, fontweight='bold')
    ax.set_ylabel('Amount (₹)', fontsize=12)
    ax.set_xlabel('Category', fontsize=12)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'₹{height:,.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    path = os.path.join(PLOTS_DIR, f'{user_id}_category_breakdown_{int(datetime.now().timestamp())}.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return path

def plot_pie_chart(df: pd.DataFrame, user_id: str, query_analysis: Dict) -> str:
    """
    Create pie chart for category distribution
    """
    if df.empty:
        return create_empty_plot(user_id, "No data found for pie chart")
    
    agg = df.groupby('category')['amount'].sum().sort_values(ascending=False)
    
    # Show top 8 categories, group rest as "Others"
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
    
    # Enhance text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    time_desc = ""
    if query_analysis['time_value'] and query_analysis['time_period']:
        time_desc = f" - Last {query_analysis['time_value']} {query_analysis['time_period']}"
    
    ax.set_title(f'Spending Distribution{time_desc}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f'{user_id}_pie_chart_{int(datetime.now().timestamp())}.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return path

def plot_income_vs_expense(income_df: pd.DataFrame, expense_df: pd.DataFrame, user_id: str, query_analysis: Dict) -> str:
    """
    Create income vs expense comparison plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Monthly totals
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
    
    # Combine data
    all_months = sorted(set(list(expense_monthly.index) + list(income_monthly.index)))
    
    if all_months:
        income_values = [income_monthly.get(month, 0) for month in all_months]
        expense_values = [expense_monthly.get(month, 0) for month in all_months]
        month_labels = [str(month) for month in all_months]
        
        # Bar chart comparison
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
        
        # Net savings/loss
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
    path = os.path.join(PLOTS_DIR, f'{user_id}_income_vs_expense_{int(datetime.now().timestamp())}.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return path

def create_empty_plot(user_id: str, message: str) -> str:
    """
    Create a plot showing no data message
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=16, 
            transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    ax.set_title('No Data Available', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    path = os.path.join(PLOTS_DIR, f'{user_id}_no_data_{int(datetime.now().timestamp())}.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return path

def generate_smart_plot(user_id: str, query: str, expense_docs: List[Dict], income_docs: List[Dict] = None) -> Tuple[str, str]:
    """
    Main function to generate appropriate plot based on query analysis
    """
    # Analyze the query
    query_analysis = analyze_query_for_plot_type(query)
    
    # Convert documents to DataFrames
    expense_df = pd.DataFrame(expense_docs) if expense_docs else pd.DataFrame()
    income_df = pd.DataFrame(income_docs) if income_docs else pd.DataFrame()
    
    # Filter data based on query
    if not expense_df.empty:
        expense_df = filter_data_by_query(expense_df, query_analysis)
    if not income_df.empty:
        income_df = filter_data_by_query(income_df, query_analysis)
    
    # Generate appropriate plot based on analysis
    plot_type = query_analysis['plot_type']
    
    try:
        if plot_type == 'time_series':
            path = plot_time_series_enhanced(expense_df, user_id, query_analysis)
            plot_description = f"Time series plot showing spending patterns over {query_analysis.get('time_period', 'time')}"
        
        elif plot_type == 'pie_chart':
            path = plot_pie_chart(expense_df, user_id, query_analysis)
            plot_description = "Pie chart showing spending distribution by category"
        
        elif plot_type == 'income_vs_expense':
            path = plot_income_vs_expense(income_df, expense_df, user_id, query_analysis)
            plot_description = "Comparison of income vs expenses with net savings/loss"
        
        else:  # default to category breakdown
            path = plot_category_breakdown_enhanced(expense_df, user_id, query_analysis)
            plot_description = f"Category breakdown of spending{' for ' + ', '.join(query_analysis['categories']) if query_analysis['categories'] else ''}"
        
        return path, plot_description
        
    except Exception as e:
        print(f"Error generating plot: {e}")
        path = create_empty_plot(user_id, f"Error generating plot: {str(e)}")
        return path, "Error generating plot"

# Legacy functions for backward compatibility
def plot_spend_by_category(df: pd.DataFrame, user_id: str):
    """Legacy function - now calls enhanced version"""
    query_analysis = {'plot_type': 'category_bar', 'time_period': None, 'time_value': None, 'categories': [], 'query': 'spending by category'}
    return plot_category_breakdown_enhanced(df, user_id, query_analysis)

def plot_time_series(df: pd.DataFrame, user_id: str):
    """Legacy function - now calls enhanced version"""
    query_analysis = {'plot_type': 'time_series', 'time_period': 'daily', 'time_value': None, 'categories': [], 'query': 'spending over time'}
    return plot_time_series_enhanced(df, user_id, query_analysis)