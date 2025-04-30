import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from glob import glob
import warnings
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import matplotlib.patheffects as path_effects

warnings.filterwarnings('ignore')

OUTPUT_FOLDER = "steem_pro_analysis"

plt.style.use('seaborn-v0_8-whitegrid')
sns.set(style="whitegrid", font_scale=1.3)

PREMIUM_COLORS = {
    'main': ['#3366cc', '#dc3912', '#ff9900', '#109618', '#990099', '#0099c6'],
    'blue_green': ['#00429d', '#4771b2', '#73a2c6', '#a5d5d8', '#ffffe0'],
    'green_red': ['#1a9850', '#66bd63', '#a6d96a', '#d9ef8b', '#fee08b', '#fdae61', '#f46d43', '#d73027'],
    'pastel': ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5'],
    'accent': ['#4D648D', '#00A9A5', '#EC9A29', '#1A936F', '#114B5F'],
    'behavior': ['#2E86C1', '#3498DB', '#5DADE2', '#F39C12', '#E67E22', '#e6e6e6']
}

def money_formatter(x, pos):
    if abs(x) >= 1_000_000_000:
        return f'{x/1_000_000_000:.1f}B'
    elif abs(x) >= 1_000_000:
        return f'{x/1_000_000:.1f}M'
    elif abs(x) >= 1_000:
        return f'{x/1_000:.1f}K'
    else:
        return f'{x:.0f}'

def add_elegant_text(ax, x, y, text, fontsize=12, color='black', ha='center', va='center',
                  weight='normal', bbox=None, shadow=False, alpha=1.0):
    t = ax.text(x, y, text, fontsize=fontsize, color=color, ha=ha, va=va,
               weight=weight, alpha=alpha, bbox=bbox, zorder=10)

    if shadow:
        t.set_path_effects([
            path_effects.Stroke(linewidth=2.5, foreground='white'),
            path_effects.Normal()
        ])

    return t

def get_report_info():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    username = os.environ.get('USER') or os.environ.get('USERNAME') or "Analyst"
    return f"Report generated on {current_time} by {username}"

def load_and_deduplicate_data(directory_path):
    print(f"Loading files from {directory_path}...")

    csv_files = glob(os.path.join(directory_path, "*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {directory_path}")

    print(f"Found {len(csv_files)} CSV files.")

    all_data = []
    for file_path in csv_files:
        try:
            if 'steem_accounts_analysis' not in os.path.basename(file_path):
                continue

            try:
                df = pd.read_csv(file_path)
                if not all(col in df.columns for col in ['Account Name', 'Power Up Total', 'Withdrawal Total']):
                    df = pd.read_csv(file_path, header=None)
                    if len(df.columns) >= 8:
                        df.columns = ["Account Name", "Power Up Total", "Withdrawal Total", "Total Amount",
                                    "Power Up Percentage", "Withdrawal Percentage", "Behavior Category", "Error"] + list(df.columns[8:])
            except Exception as e:
                print(f"Warning: Skipping file {os.path.basename(file_path)} due to error: {str(e)}")
                continue

            all_data.append(df)
            print(f"  - Loaded {os.path.basename(file_path)} with {len(df)} records")
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")

    if not all_data:
        raise ValueError("No valid data could be loaded from CSV files")

    combined_df = pd.concat(all_data, ignore_index=True)

    original_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['Account Name'])
    dedup_count = len(combined_df)

    print(f"Removed {original_count - dedup_count} duplicate accounts.")
    print(f"Final dataset contains {dedup_count} unique accounts.")

    return combined_df

def clean_and_transform_data(df):
    numeric_cols = ['Power Up Total', 'Withdrawal Total', 'Total Amount',
                   'Power Up Percentage', 'Withdrawal Percentage']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df[numeric_cols] = df[numeric_cols].fillna(0)

    if 'Behavior Category' in df.columns:
        df = df[~df['Behavior Category'].isna()]

    df['Power Up Ratio'] = df['Power Up Total'] / (df['Power Up Total'] + df['Withdrawal Total'] + 1e-10)
    df['Net Flow'] = df['Power Up Total'] - df['Withdrawal Total']
    df['Net Flow Direction'] = np.where(df['Net Flow'] > 0, 'Positive', 'Negative')

    category_mapping = {
        "100% Power Up": "100% Power Up",
        "75% Power Up": "Strong Power Up (75-99%)",
        "50% Power Up": "Balanced Power Up (50-74%)",
        "25% Power Up": "Light Power Up (25-49%)",
        "Primarily Withdrawal": "Mostly Withdrawal (<25%)",
        "No Activity": "No Activity"
    }

    if 'Behavior Category' in df.columns:
        df['User Type'] = df['Behavior Category'].map(category_mapping).fillna(df['Behavior Category'])

    return df

def generate_statistics(df):
    results = {}

    total_accounts = len(df)
    active_accounts = df[df['Total Amount'] > 0].shape[0]
    inactive_accounts = total_accounts - active_accounts

    results["total_accounts"] = total_accounts
    results["active_accounts"] = active_accounts
    results["inactive_accounts"] = inactive_accounts
    results["active_percentage"] = (active_accounts/total_accounts)*100 if total_accounts > 0 else 0

    if active_accounts == 0:
        return results

    active_df = df[df['Total Amount'] > 0].copy()

    category_counts = active_df['User Type'].value_counts() # Use User Type for consistency
    category_percentages = (category_counts / active_accounts * 100).round(1)

    results["category_counts"] = category_counts
    results["category_percentages"] = category_percentages

    total_power_up = active_df['Power Up Total'].sum()
    total_withdrawal = active_df['Withdrawal Total'].sum()
    total_transactions = total_power_up + total_withdrawal

    results["total_power_up"] = total_power_up
    results["total_withdrawal"] = total_withdrawal
    results["total_transactions"] = total_transactions
    results["net_flow"] = total_power_up - total_withdrawal

    if total_transactions > 0:
        results["power_up_percentage"] = (total_power_up / total_transactions) * 100
        results["withdrawal_percentage"] = (total_withdrawal / total_transactions) * 100
    else:
        results["power_up_percentage"] = 0
        results["withdrawal_percentage"] = 0

    results["avg_power_up"] = active_df['Power Up Total'].mean()
    results["avg_withdrawal"] = active_df['Withdrawal Total'].mean()

    power_up_groups = [
        (active_df['Power Up Percentage'] == 100).sum(),
        ((active_df['Power Up Percentage'] >= 75) & (active_df['Power Up Percentage'] < 100)).sum(),
        ((active_df['Power Up Percentage'] >= 50) & (active_df['Power Up Percentage'] < 75)).sum(),
        ((active_df['Power Up Percentage'] >= 25) & (active_df['Power Up Percentage'] < 50)).sum(),
        ((active_df['Power Up Percentage'] > 0) & (active_df['Power Up Percentage'] < 25)).sum(),
        (active_df['Power Up Percentage'] == 0).sum()
    ]

    power_up_labels = [
        "100% Power Up",
        "Strong Power Up (75-99%)",
        "Balanced Power Up (50-74%)",
        "Light Power Up (25-49%)",
        "Minimal Power Up (1-24%)",
        "No Power Up (0%)"
    ]

    results["power_up_distribution"] = {
        "counts": power_up_groups,
        "labels": power_up_labels,
        "percentages": [(count/active_accounts)*100 for count in power_up_groups]
    }

    results["max_power_up"] = active_df['Power Up Total'].max()
    results["max_withdrawal"] = active_df['Withdrawal Total'].max()

    positive_flow_count = (active_df['Net Flow'] > 0).sum()
    negative_flow_count = (active_df['Net Flow'] < 0).sum()
    zero_flow_count = (active_df['Net Flow'] == 0).sum()

    results["positive_flow_count"] = positive_flow_count
    results["negative_flow_count"] = negative_flow_count
    results["zero_flow_count"] = zero_flow_count

    if active_accounts > 0:
        results["positive_flow_pct"] = (positive_flow_count / active_accounts) * 100
        results["negative_flow_pct"] = (negative_flow_count / active_accounts) * 100
        results["zero_flow_pct"] = (zero_flow_count / active_accounts) * 100

    return results

def create_enhanced_user_profiles_chart(df, stats, charts_dir):
    fig, ax = plt.subplots(figsize=(14, 12))
    fig.set_facecolor('#f8f9fa')

    # Use a more distinct color palette
    behavior_colors = PREMIUM_COLORS['accent'] + PREMIUM_COLORS['main']

    cats = stats["category_counts"].index.tolist()
    counts = stats["category_counts"].values

    sorted_indices = np.argsort(counts)[::-1]
    cats = [cats[i] for i in sorted_indices]
    counts = counts[sorted_indices]

    # Ensure enough colors for categories, repeat if necessary
    num_cats = len(counts)
    plot_colors = [behavior_colors[i % len(behavior_colors)] for i in range(num_cats)]


    wedges, texts, autotexts = ax.pie(
        counts,
        labels=None,
        autopct=lambda p: f'{p:.1f}%\n({int(p*sum(counts)/100):,})',
        startangle=90,
        colors=plot_colors,
        wedgeprops={'width': 0.5, 'edgecolor': 'white', 'linewidth': 2, 'alpha': 0.9},
        textprops={'fontsize': 15, 'weight': 'bold', 'color': 'white'},
        pctdistance=0.75
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(14)
        autotext.set_weight('bold')
        autotext.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground='black', alpha=0.6),
            path_effects.Normal()
        ])

    title_text = ax.set_title('STEEM User Behavior Profiles', fontsize=28, pad=30,
                            color='#154360', weight='bold')
    title_text.set_path_effects([
        path_effects.Stroke(linewidth=3, foreground='#f1f1f1'),
        path_effects.Normal()
    ])

    legend_labels = []
    for i, cat in enumerate(cats):
        legend_labels.append(f"{cat}")

    legend = ax.legend(
        wedges, legend_labels,
        title="User Behaviors",
        loc="center right",
        bbox_to_anchor=(1.3, 0.5),
        frameon=True,
        framealpha=0.8,
        facecolor='#f8f9fa',
        fontsize=13,
        title_fontsize=18
    )
    legend.get_title().set_fontweight('bold')

    center_circle = plt.Circle((0, 0), 0.3, fc='white', ec='#1A5276', linewidth=2)
    ax.add_artist(center_circle)

    center_highlight = plt.Circle((0, 0), 0.29, fc='#EBF5FB', ec=None, alpha=0.7)
    ax.add_artist(center_highlight)

    formatted_active = format(stats['active_accounts'], ',')
    formatted_total = format(stats['total_accounts'], ',')

    add_elegant_text(ax, 0, 0.08, f"Active Users", fontsize=16, color='#154360', weight='bold')
    add_elegant_text(ax, 0, -0.05, f"{formatted_active}", fontsize=20, color='#154360', weight='bold')
    add_elegant_text(ax, 0, -0.15, f"of {formatted_total} total", fontsize=14, color='#2E86C1')

    subtitle = (
        "How users manage their STEEM funds between Power Up and Withdrawal"
    )
    plt.figtext(0.5, 0.94, subtitle, ha='center', fontsize=16, style='italic', color='#2E86C1')

    explanation = (
        "Each segment shows percentage and count of users in that category.\n"
        "Larger segments represent more common behaviors."
    )
    plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=12, color='#555555')

    plt.figtext(0.5, 0.05, get_report_info(), ha='center', fontsize=10, style='italic', color='#777777')

    ax.set_aspect('equal')

    plt.tight_layout(rect=[0, 0.1, 1, 0.9])
    plt.savefig(os.path.join(charts_dir, '1_steem_user_profiles.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("Enhanced User Behavior Profiles chart created with improved design for large numbers")

def create_power_up_vs_withdrawal_chart(stats, charts_dir):
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.set_facecolor('#f5f7f9')

    transaction_data = [stats["total_power_up"], stats["total_withdrawal"]]
    labels = ['Power Up', 'Withdrawal']
    colors = ['#4CAF50', '#F44336']

    total_volume = sum(transaction_data)
    percentages = [100 * x / total_volume if total_volume > 0 else 0 for x in transaction_data]

    bars = ax.bar(
        labels,
        transaction_data,
        color=colors,
        width=0.5,
        edgecolor='white',
        linewidth=1.5
    )

    for i, bar in enumerate(bars):
        value = transaction_data[i]
        percentage = percentages[i]

        if value >= 1_000_000:
            value_text = f'{value/1_000_000:.1f}M'
        elif value >= 1_000:
            value_text = f'{value/1_000:.1f}K'
        else:
            value_text = f'{value:.1f}'

        label_text = f'{value_text}\n({percentage:.1f}%)'

        height = bar.get_height()
        y_pos = height + (total_volume * 0.01) if total_volume > 0 else 0.01

        bbox_props = dict(boxstyle="round,pad=0.3", fc='white', ec=colors[i], alpha=0.85)
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            y_pos,
            label_text,
            ha='center',
            va='bottom',
            fontsize=16,
            fontweight='bold',
            color=colors[i],
            bbox=bbox_props
        )

    ax.set_facecolor('#f8f9fa')
    ax.set_ylabel('STEEM Volume', fontsize=18, labelpad=10, fontweight='bold', color='#333333')

    ax.set_xticklabels(labels, fontsize=16, fontweight='bold')

    formatter = FuncFormatter(lambda x, pos: f'{x/1_000_000:.1f}M' if x >= 1_000_000 else
                                          (f'{x/1_000:.1f}K' if x >= 1_000 else f'{x:.0f}'))
    ax.yaxis.set_major_formatter(formatter)
    ax.tick_params(axis='y', labelsize=14)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

    title_text = ax.set_title('STEEM Power Up vs. Withdrawal Volume',
                             fontsize=24, pad=20, fontweight='bold', color='#333333')

    net_flow = stats["net_flow"]
    flow_direction = "more Power Up than Withdrawal" if net_flow > 0 else "more Withdrawal than Power Up"
    flow_color = '#4CAF50' if net_flow > 0 else '#F44336'

    if abs(net_flow) >= 1_000_000:
        net_flow_text = f'{net_flow/1_000_000:+.1f}M'
    elif abs(net_flow) >= 1_000:
        net_flow_text = f'{net_flow/1_000:+.1f}K'
    else:
        net_flow_text = f'{net_flow:+.1f}'

    highlight_text = f"Net Flow: {net_flow_text} STEEM ({flow_direction})"

    bbox_props = dict(
        boxstyle="round,pad=0.5",
        fc=flow_color,
        ec="white",
        alpha=0.8,
        lw=2
    )

    plt.figtext(
        0.5, 0.07,
        highlight_text,
        ha="center",
        fontsize=16,
        fontweight='bold',
        color='white',
        bbox=bbox_props
    )

    explanation = (
        "Power Up: Users staking their STEEM to earn rewards\n" +
        "Withdrawal: Users converting STEEM to liquid assets"
    )

    plt.figtext(0.5, 0.15, explanation, ha='center', fontsize=14, style='italic', color='#555555')

    plt.figtext(0.5, 0.01, get_report_info(), ha='center', fontsize=10, style='italic', color='#777777')

    plt.tight_layout(rect=[0, 0.17, 1, 0.95])
    plt.savefig(os.path.join(charts_dir, '2_power_up_vs_withdrawal_volume.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("Enhanced Power Up vs Withdrawal chart created with clearer labels for large numbers")

def create_active_vs_inactive_chart(stats, charts_dir):
    fig, ax = plt.subplots(figsize=(14, 12))
    fig.set_facecolor('#f8f9fa')

    activity_labels = ['Active Users', 'Inactive Users']
    activity_counts = [stats["active_accounts"], stats["inactive_accounts"]]
    activity_percentages = [
        stats["active_percentage"],
        100 - stats["active_percentage"]
    ]

    formatted_counts = [f"{count:,}" for count in activity_counts]
    formatted_total = f"{stats['total_accounts']:,}"

    colors = ['#1E88E5', '#E0E0E0']

    wedges, _ = ax.pie(
        activity_counts,
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=3),
        startangle=90,
        colors=colors
    )

    centre_circle = plt.Circle((0, 0), 0.3, fc='white', ec='#dddddd', lw=2)
    ax.add_artist(centre_circle)

    center_highlight = plt.Circle((0, 0), 0.29, fc='#E3F2FD', ec=None, alpha=0.7)
    ax.add_artist(center_highlight)

    active_bbox = dict(boxstyle="round,pad=0.5", fc=colors[0], ec='white', alpha=0.9)
    inactive_bbox = dict(boxstyle="round,pad=0.5", fc=colors[1], ec='white', alpha=0.9)

    plt.figtext(
        0.25, 0.5,
        "ACTIVE USERS",
        ha="center",
        va="bottom",
        fontsize=14,
        color='#555555',
        weight='bold',
        bbox=active_bbox
    )

    plt.figtext(
        0.25, 0.45,
        f"{formatted_counts[0]}\n({activity_percentages[0]:.1f}%)",
        ha="center",
        va="top",
        fontsize=16,
        color='#555555',
        weight='bold'
    )

    plt.figtext(
        0.75, 0.5,
        "INACTIVE USERS",
        ha="center",
        va="bottom",
        fontsize=14,
        color='#555555',
        weight='bold',
        bbox=inactive_bbox
    )

    plt.figtext(
        0.75, 0.45,
        f"{formatted_counts[1]}\n({activity_percentages[1]:.1f}%)",
        ha="center",
        va="top",
        fontsize=16,
        color='#555555',
        weight='bold'
    )

    add_elegant_text(
        ax, 0, 0.05,
        "TOTAL USERS",
        fontsize=16,
        color='#1976D2',
        weight='bold'
    )
    add_elegant_text(
        ax, 0, -0.05,
        formatted_total,
        fontsize=22,
        color='#1565C0',
        weight='bold',
        shadow=True
    )

    title_text = fig.suptitle(
        'STEEM User Activity Status',
        fontsize=28,
        y=0.95,
        fontweight='bold',
        color='#0D47A1'
    )
    title_text.set_path_effects([
        path_effects.Stroke(linewidth=2, foreground='#f1f1f1'),
        path_effects.Normal()
    ])

    explanation = (
        "Active users: Have made at least one transaction (Power Up or Withdrawal)\n"
        "Inactive users: No recorded transactions during the analysis period"
    )

    explanation_box = dict(boxstyle="round,pad=0.5", fc='#f8f9fa', ec='#dddddd', alpha=0.9)
    plt.figtext(
        0.5, 0,
        explanation,
        ha="center",
        fontsize=14,
        style='italic',
        color='#555555',
        bbox=explanation_box
    )

    plt.figtext(0.5, 0.05, get_report_info(), ha='center', fontsize=10, style='italic', color='#777777')

    plt.tight_layout(rect=[0, 0.1, 1, 0.9])
    plt.savefig(os.path.join(charts_dir, '4_user_activity_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("Enhanced User Activity Status chart created with improved visibility for large numbers")

def create_user_power_up_distribution_chart(stats, charts_dir):
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.set_facecolor('#f5f7f9')

    power_up_counts = stats["power_up_distribution"]["counts"]
    power_up_labels = stats["power_up_distribution"]["labels"]
    power_up_percentages = stats["power_up_distribution"]["percentages"]

    sorted_indices = np.argsort(power_up_counts)[::-1]
    sorted_labels = [power_up_labels[i] for i in sorted_indices]
    sorted_counts = [power_up_counts[i] for i in sorted_indices]
    sorted_percentages = [power_up_percentages[i] for i in sorted_indices]

    formatted_counts = [f"{int(count):,}" for count in sorted_counts]

    colors = ['#1a9850', '#91cf60', '#d9ef8b', '#fee08b', '#fc8d59', '#d73027']
    custom_cmap = dict(zip(sorted_labels, [colors[i % len(colors)] for i in range(len(sorted_labels))]))

    bars = ax.barh(
        sorted_labels,
        sorted_counts,
        color=[custom_cmap[label] for label in sorted_labels],
        height=0.7,
        edgecolor='white',
        linewidth=1
    )

    for i, (bar, pct, count_text) in enumerate(zip(bars, sorted_percentages, formatted_counts)):
        width = bar.get_width()
        label_x_pos = width + (max(sorted_counts) * 0.02) if max(sorted_counts) > 0 else 0.02

        bbox_props = dict(boxstyle="round,pad=0.3", fc='white', ec=custom_cmap[sorted_labels[i]], alpha=0.85)

        ax.text(
            label_x_pos,
            i,
            f"{count_text} users ({pct:.1f}%)",
            va='center',
            ha='left',
            fontsize=14,
            fontweight='bold',
            color='#333333',
            bbox=bbox_props
        )

        if width > max(sorted_counts) * 0.1 if max(sorted_counts) > 0 else 0:
            ax.text(
                width/2,
                i,
                f"{pct:.1f}%",
                va='center',
                ha='center',
                fontsize=14,
                fontweight='bold',
                color='white',
                path_effects=[path_effects.withStroke(linewidth=2, foreground='black', alpha=0.3)]
            )

    ax.set_facecolor('#f8f9fa')
    ax.set_xlabel('Number of Users', fontsize=16, labelpad=10, fontweight='bold', color='#333333')

    formatter = FuncFormatter(lambda x, pos: f'{x/1_000_000:.1f}M' if x >= 1_000_000 else
                                          (f'{x/1_000:.1f}K' if x >= 1_000 else f'{x:.0f}'))
    ax.xaxis.set_major_formatter(formatter)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

    title_text = ax.set_title('How Users Power Up Their STEEM',
                             fontsize=24, pad=20, fontweight='bold', color='#333333')
    title_text.set_path_effects([
        path_effects.Stroke(linewidth=2, foreground='#f1f1f1'),
        path_effects.Normal()
    ])

    explanation = (
        "Users are grouped by the percentage of their activity\n"
        "that goes toward powering up STEEM vs. withdrawals."
    )

    plt.figtext(
        0.5, 0.07,
        explanation,
        ha="center",
        fontsize=14,
        style='italic',
        color='#555555'
    )

    plt.figtext(0.5, 0.01, get_report_info(), ha='center', fontsize=10, style='italic', color='#777777')

    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    plt.savefig(os.path.join(charts_dir, '3_user_power_up_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("User Power Up Distribution chart created.")


def create_net_flow_dashboard_chart(stats, charts_dir):
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.set_facecolor('#f8f9fa')

    flow_labels = ['Positive Net Flow\n(More Power Up)', 'Zero Net Flow\n(Balanced)', 'Negative Net Flow\n(More Withdrawal)']
    flow_counts = [
        stats.get("positive_flow_count", 0),
        stats.get("zero_flow_count", 0),
        stats.get("negative_flow_count", 0)
    ]
    flow_percentages = [
        stats.get("positive_flow_pct", 0),
        stats.get("zero_flow_pct", 0),
        stats.get("negative_flow_pct", 0)
    ]

    formatted_counts = [f"{int(count):,}" for count in flow_counts]

    colors = ['#4CAF50', '#9E9E9E', '#F44336']

    bars = ax.barh(
        ['Net Flow Pattern'],
        [100],
        color='white',
        edgecolor='none',
        height=0.6
    )

    left = 0
    for i, (count, percentage, color) in enumerate(zip(flow_counts, flow_percentages, colors)):
        bar = ax.barh(
            ['Net Flow Pattern'],
            [percentage],
            left=left,
            color=color,
            edgecolor='white',
            height=0.6,
            linewidth=2
        )

        if percentage > 5:
            ax.text(
                left + percentage/2,
                0,
                f"{percentage:.1f}%",
                ha='center',
                va='center',
                fontsize=14,
                color='white',
                weight='bold',
                path_effects=[path_effects.withStroke(linewidth=2, foreground='black', alpha=0.2)]
            )

        left += percentage

    y_positions = [0.65, 0.5, 0.35]
    x_positions = [0.25, 0.5, 0.75]

    for i, (label, count, pct, color, x_pos, y_pos) in enumerate(zip(
        flow_labels, formatted_counts, flow_percentages, colors, x_positions, y_positions)):

        card_bbox = dict(boxstyle="round,pad=0.4", fc=color, ec='white', alpha=0.85)

        plt.figtext(
            x_pos, y_pos + 0.05,
            label,
            ha='center',
            va='center',
            fontsize=14,
            color='white',
            weight='bold',
            bbox=card_bbox
        )

        plt.figtext(
            x_pos, y_pos - 0.05,
            f"{count} users\n({pct:.1f}%)",
            ha='center',
            va='center',
            fontsize=14,
            color=color,
            weight='bold'
        )

    ax.set_yticks([])
    ax.set_xlim(0, 100)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=12)
    ax.xaxis.set_ticks_position('bottom')

    title_text = ax.set_title('STEEM User Net Flow Patterns',
                             fontsize=24, pad=20, weight='bold', color='#333333')
    title_text.set_path_effects([
        path_effects.Stroke(linewidth=2, foreground='#f1f1f1'),
        path_effects.Normal()
    ])

    explanation = (
        "This chart shows whether users are predominantly powering up or withdrawing their STEEM.\n"
        "Positive Net Flow means a user powers up more than they withdraw. Negative Net Flow means the opposite."
    )

    plt.figtext(
        0.5, 0.15,
        explanation,
        ha="center",
        fontsize=14,
        style='italic',
        color='#555555'
    )

    plt.figtext(0.5, 0.05, get_report_info(), ha='center', fontsize=10, style='italic', color='#777777')

    plt.tight_layout(rect=[0, 0.2, 1, 0.95])
    plt.savefig(os.path.join(charts_dir, '6_net_flow_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Net Flow Dashboard chart created.")

def create_pro_charts(df, stats, output_dir):
    print("\nCreating premium-quality charts...")

    charts_dir = os.path.join(output_dir, OUTPUT_FOLDER)
    os.makedirs(charts_dir, exist_ok=True)

    active_df = df[df['Total Amount'] > 0].copy()

    if len(active_df) == 0:
        print("No active accounts to plot charts for.")
        return

    plt.rcParams['figure.figsize'] = (14, 10)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['font.family'] = 'sans-serif'

    create_enhanced_user_profiles_chart(df, stats, charts_dir)
    create_power_up_vs_withdrawal_chart(stats, charts_dir)
    create_user_power_up_distribution_chart(stats, charts_dir)
    create_active_vs_inactive_chart(stats, charts_dir)
    # Removed: create_power_up_withdrawal_patterns_chart(active_df, charts_dir)
    create_net_flow_dashboard_chart(stats, charts_dir)

    print(f"Premium charts saved to {charts_dir}")

def export_unique_accounts(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"{OUTPUT_FOLDER}/unique_steem_accounts.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    df.to_csv(csv_path, index=False)

    print(f"Exported {len(df)} unique accounts to {csv_path}")
    return csv_path

def generate_markdown_report(stats, output_dir):
    charts_dir = os.path.join(output_dir, OUTPUT_FOLDER)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    username = os.environ.get('USER') or os.environ.get('USERNAME') or "Analyst"

    md_content = [
        f"# STEEM User Behavior Analysis Report",
        f"*Generated on: {timestamp}*",
        "",
        "## Executive Summary",
        f"This report analyzes the behavior of STEEM users, focusing on how they manage their funds between Power Up (staking) and Withdrawal activities. The analysis covers {stats['total_accounts']:,} total accounts.",
        "",
        "### Key Highlights",
        f"- **{stats['active_percentage']:.1f}%** of accounts are active (have made transactions)",
        f"- **{stats.get('power_up_percentage', 0):.1f}%** of transaction volume is Power Up",
        f"- **{stats.get('withdrawal_percentage', 0):.1f}%** of transaction volume is Withdrawal",
        "",
        "## Detailed Findings",
        "",
        "### User Activity Overview",
        f"Out of all {stats['total_accounts']:,} analyzed accounts:",
        f"- **Active Accounts**: {stats['active_accounts']:,} ({stats['active_percentage']:.1f}%)",
        f"- **Inactive Accounts**: {stats['inactive_accounts']:,} ({100-stats['active_percentage']:.1f}%)",
        "",
        f"![User Activity Dashboard]({OUTPUT_FOLDER}/4_user_activity_dashboard.png)",
    ]

    if "total_power_up" in stats and "total_withdrawal" in stats:
        net_flow_direction = "positive (more Power Up)" if stats['net_flow'] >= 0 else "negative (more Withdrawal)"

        md_content.extend([
            "",
            "### Transaction Volumes",
            f"The analysis reveals the following transaction patterns:",
            "",
            f"- **Total Power Up Volume**: {stats['total_power_up']:,.1f} STEEM ({stats.get('power_up_percentage', 0):.1f}%)",
            f"- **Total Withdrawal Volume**: {stats['total_withdrawal']:,.1f} STEEM ({stats.get('withdrawal_percentage', 0):.1f}%)",
            f"- **Total Transaction Volume**: {stats.get('total_transactions', 0):,.1f} STEEM",
            f"- **Net Flow**: {stats.get('net_flow', 0):+,.1f} STEEM ({net_flow_direction})",
            "",
            f"![Power Up vs Withdrawal Volume]({OUTPUT_FOLDER}/2_power_up_vs_withdrawal_volume.png)",
        ])

    if "category_counts" in stats and len(stats["category_counts"]) > 0:
        md_content.extend([
            "",
            "### User Behavior Profiles",
            "Users fall into various behavior profiles based on their Power Up vs Withdrawal patterns:",
            ""
        ])

        # Sort categories by count for reporting consistency
        sorted_categories = stats["category_counts"].sort_values(ascending=False)

        for category, count in sorted_categories.items():
            percentage = stats["category_percentages"].get(category, 0) # Use .get for safety
            md_content.append(f"- **{category}**: {count:,} users ({percentage:.1f}%)")

        md_content.extend([
            "",
            f"![User Behavior Profiles]({OUTPUT_FOLDER}/1_steem_user_profiles.png)"
        ])

    if "power_up_distribution" in stats:
        md_content.extend([
            "",
            "### Power Up Behavior Distribution",
            "When looking at how much of their activity users devote to Power Up:",
            ""
        ])

        counts = stats["power_up_distribution"]["counts"]
        labels = stats["power_up_distribution"]["labels"]
        percentages = stats["power_up_distribution"]["percentages"]

        sorted_indices = np.argsort(counts)[::-1]

        for i in sorted_indices:
            md_content.append(f"- **{labels[i]}**: {counts[i]:,} users ({percentages[i]:.1f}%)")

        md_content.extend([
            "",
            f"![Power Up Distribution]({OUTPUT_FOLDER}/3_user_power_up_distribution.png)"
        ])

    if all(k in stats for k in ["positive_flow_count", "zero_flow_count", "negative_flow_count"]):
        md_content.extend([
            "",
            "### Net Flow Patterns",
            "Looking at whether users are net Power Up or Withdrawal focused:",
            "",
            f"- **Positive Net Flow (More Power Up)**: {stats['positive_flow_count']:,} users ({stats.get('positive_flow_pct', 0):.1f}%)",
            f"- **Zero Net Flow (Balanced)**: {stats['zero_flow_count']:,} users ({stats.get('zero_flow_pct', 0):.1f}%)",
            f"- **Negative Net Flow (More Withdrawal)**: {stats['negative_flow_count']:,} users ({stats.get('negative_flow_pct', 0):.1f}%)",
            "",
            f"![Net Flow Patterns]({OUTPUT_FOLDER}/6_net_flow_dashboard.png)"
        ])

    # Removed reference to chart 5
    # md_content.extend([
    #     "",
    #     "### User Transaction Patterns",
    #     "The scatter plot below shows the relationship between Power Up and Withdrawal activities for each user:",
    #     "",
    #     f"![Power Up vs Withdrawal Patterns]({OUTPUT_FOLDER}/5_power_up_withdrawal_patterns.png)",
    #     "",
    #     "Each dot represents a user, with the position showing their Power Up amount (horizontal) versus Withdrawal amount (vertical). Larger dots indicate users with higher total transaction volumes."
    # ])

    md_content.extend([
        "",
        "## Key Insights",
        ""
    ])

    insights = []

    if stats['active_percentage'] > 75:
        insights.append("- Most accounts are active, showing high engagement in the platform.")
    elif stats['active_percentage'] > 50:
        insights.append("- More than half of all accounts are active, showing good engagement.")
    elif stats['active_percentage'] < 25:
        insights.append("- Most accounts are inactive, suggesting potential opportunities to increase engagement.")

    power_up_pct = stats.get('power_up_percentage', 0)
    withdrawal_pct = stats.get('withdrawal_percentage', 0)

    if power_up_pct > withdrawal_pct:
        if power_up_pct > 75:
            insights.append("- Users overwhelmingly prefer to Power Up their STEEM rather than withdraw it, indicating strong confidence in staking.")
        else:
            insights.append("- Users tend to Power Up more than they withdraw, showing good confidence in staking STEEM.")
    elif withdrawal_pct > 0: # Avoid insight if both are 0
        insights.append("- Withdrawal activity exceeds Power Up, which may indicate users are taking profits or moving funds elsewhere.")


    if "category_counts" in stats and len(stats["category_counts"]) > 0:
         # Ensure we use the sorted list from above
        most_common_category = sorted_categories.index[0]
        if "100% Power Up" in most_common_category:
            insights.append("- The most common user behavior is 100% Power Up, showing strong commitment to staking STEEM.")
        elif "Power Up" in most_common_category:
            insights.append(f"- The most common user behavior is '{most_common_category}', showing good support for staking STEEM.")
        elif "Withdrawal" in most_common_category.lower():
            insights.append(f"- The most common user behavior is '{most_common_category}', indicating users are primarily extracting value.")
        elif "No Activity" in most_common_category:
             insights.append(f"- The most common user behavior is '{most_common_category}'.")


    positive_flow_pct = stats.get("positive_flow_pct", 0)
    negative_flow_pct = stats.get("negative_flow_pct", 0)
    if positive_flow_pct > negative_flow_pct:
        insights.append("- More users have a positive net flow (Power Up > Withdrawal), showing overall confidence in STEEM.")
    elif negative_flow_pct > 0: # Avoid insight if both are 0
        insights.append("- More users have a negative net flow (Withdrawal > Power Up), which may indicate profit-taking or reduced confidence.")

    md_content.extend(insights)

    md_content.extend([
        "",
        "## Recommendations",
        "",
        "Based on the analysis, consider the following actions:",
        "",
        "1. **Engagement Strategy**: Focus on converting inactive accounts to active users to increase platform vitality.",
        "2. **Staking Incentives**: Promote the benefits of Power Up to encourage more users to stake their STEEM.",
        "3. **User Retention**: Understand why some users are predominantly withdrawing and address those concerns.",
        "4. **Community Building**: Highlight success stories from users who have benefited from Power Up strategies.",
        "5. **Regular Monitoring**: Continue tracking these metrics over time to identify trends and measure improvements.",
        "",
        "---",
        "",
        f"*This report was automatically generated on {timestamp}.*",
        f"*All charts can be found in the `{OUTPUT_FOLDER}` folder.*"
    ])

    report_path = os.path.join(output_dir, f"{OUTPUT_FOLDER}/steem_analysis_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, 'w') as f:
        f.write('\n'.join(md_content))

    print(f"Comprehensive markdown report generated and saved to {report_path}")
    return report_path

def analyze_steem_data(input_directory, output_directory):
    try:
        print("\n" + "="*80)
        print(f"{'STEEM USER BEHAVIOR PREMIUM ANALYSIS':^80}")
        print("="*80 + "\n")

        df = load_and_deduplicate_data(input_directory)

        print("\nCleaning and transforming data...")
        df = clean_and_transform_data(df)

        csv_path = export_unique_accounts(df, output_directory)

        print("\nGenerating in-depth statistics...")
        stats = generate_statistics(df)

        os.makedirs(output_directory, exist_ok=True)

        create_pro_charts(df, stats, output_directory)

        report_path = generate_markdown_report(stats, output_directory)

        print("\n" + "="*80)
        print(f"{'PREMIUM ANALYSIS COMPLETED SUCCESSFULLY!':^80}")
        print("="*80)
        print(f"\n📊 Charts are saved in: {os.path.join(output_directory, OUTPUT_FOLDER)}")
        print(f"📝 Report is saved as: {report_path}")
        print(f"📋 Unique accounts dataset saved as: {csv_path}")
        print(f"\n{'Analysis completed on: '+datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^80}")

    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║       🔍 STEEM USER BEHAVIOR PREMIUM ANALYSIS 🔍        ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "-"*70)

    input_dir = input("Enter path to input directory containing CSV files: ").strip()
    if not input_dir:
        input_dir = "./steem_input_data"
        print(f"Using default input directory: {input_dir}")

    output_dir = input("Enter path for output charts and report: ").strip()
    if not output_dir:
        output_dir = "./steem_analysis_results"
        print(f"Using default output directory: {output_dir}")

    analyze_steem_data(input_dir, output_dir)
