import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

def generate_distribution_chart(df):
    """
    Renders an in-memory, publication-ready sentiment distribution chart
    styled for the dark cyberpunk / fintech aesthetic.
    Returns a Base64-encoded PNG image string for zero-disk web rendering.
    """
    if 'sentiment' not in df.columns or df.empty:
        return None

    # Aggregate counts across all three classes
    counts = df['sentiment'].value_counts()
    
    # Ensure all three standard classes exist in a consistent order
    categories = ['Positive', 'Negative', 'Neutral']
    values = [counts.get(cat, 0) for cat in categories]
    
    # Cyberpunk palette matching NEWSAI.html: Bullish neon emerald, Bearish neon crimson, Neutral cobalt blue
    colors = ['#00e5a0', '#ff4560', '#7b9cff']

    # Setup figure with transparent/dark background and crisp modern typography
    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=120)
    fig.patch.set_facecolor('#050710')
    fig.patch.set_alpha(0.0)  # Transparent figure background for glassmorphic containers
    ax.set_facecolor('#050710')
    ax.set_alpha(0.0)

    bars = ax.bar(categories, values, color=colors, width=0.52, edgecolor='none', zorder=3)

    # Annotate counts on top of bars in neon glow color
    for bar, color in zip(bars, colors):
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold', color='#e8eaf2')

    # Styling spines and gridlines for dark mode
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('rgba(255,255,255,0.12)')
    ax.spines['bottom'].set_color('rgba(255,255,255,0.12)')
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.15, color='#ffffff', zorder=0)
    ax.set_axisbelow(True)

    # Set integer ticks on Y-axis and style tick labels
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(axis='x', colors='#a0aec0', labelsize=10)
    ax.tick_params(axis='y', colors='#718096', labelsize=9)

    plt.title('Sentiment Class Breakdown', fontsize=12, fontweight='700', color='#e8eaf2', pad=14, fontfamily='sans-serif')
    plt.tight_layout()

    # Zero-disk I/O: Pipe directly to in-memory byte stream with transparency
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=120)
    buf.seek(0)
    plt.close(fig)

    return base64.b64encode(buf.getvalue()).decode('utf-8')
