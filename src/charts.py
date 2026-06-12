import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

def generate_distribution_chart(df):
    if 'sentiment' not in df.columns or df.empty:
        return None
    
    counts = df['sentiment'].value_counts()
    colors = {'Positive': '#198754', 'Negative': '#dc3545', 'Neutral': '#6c757d'}
    chart_colors = [colors.get(x, '#000000') for x in counts.index]
    
    plt.figure(figsize=(6, 4))
    counts.plot(kind='bar', color=chart_colors)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    return base64.b64encode(buf.getvalue()).decode('utf-8')
