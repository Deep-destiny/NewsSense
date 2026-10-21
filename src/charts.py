import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

def generate_distribution_chart(df):
    """
    Renders an in-memory, publication-ready sentiment distribution chart.
    Returns a Base64-encoded PNG image string for zero-disk web rendering.
    """
    if 'sentiment' not in df.columns or df.empty:
        return None

    # Aggregate counts across all three classes
    counts = df['sentiment'].value_counts()
    
    # Ensure all three standard classes exist in a consistent order
    categories = ['Positive', 'Negative', 'Neutral']
    values = [counts.get(cat, 0) for cat in categories]
    
    colors = ['#10b981', '#ef4444', '#64748b']  # Emerald, Crimson, Slate

    # Setup figure with clean typography and modern aspect ratio
    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=110)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    bars = ax.bar(categories, values, color=colors, width=0.55, edgecolor='none', zorder=3)

    # Annotate counts on top of bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold', color='#1e293b')

    # Styling spines and gridlines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1')
    ax.spines['bottom'].set_color('#cbd5e1')
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='#94a3b8', zorder=0)
    ax.set_axisbelow(True)

    # Set integer ticks on Y-axis
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(axis='x', colors='#334155', labelsize=10)
    ax.tick_params(axis='y', colors='#64748b', labelsize=9)

    plt.title('Sentiment Class Breakdown', fontsize=12, fontweight='600', color='#0f172a', pad=12)
    plt.tight_layout()

    # Zero-disk I/O: Pipe directly to in-memory byte stream
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=110)
    buf.seek(0)
    plt.close(fig)

    return base64.b64encode(buf.getvalue()).decode('utf-8')
