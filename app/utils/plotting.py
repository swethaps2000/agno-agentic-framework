import pandas as pd
import matplotlib.pyplot as plt
import os


PLOTS_DIR = './plots'
os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_spend_by_category(df: pd.DataFrame, user_id: str):
    agg = df.groupby('category')['amount'].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6,4))
    agg.plot(kind='bar', ax=ax)
    ax.set_title('Spend by Category')
    ax.set_ylabel('Amount')
    path = os.path.join(PLOTS_DIR, f'{user_id}_spend_by_category.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_time_series(df: pd.DataFrame, user_id: str):
    df2 = df.groupby('date')['amount'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(df2['date'], df2['amount'])
    ax.set_title('Spending Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Amount')
    path = os.path.join(PLOTS_DIR, f'{user_id}_timeseries.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path