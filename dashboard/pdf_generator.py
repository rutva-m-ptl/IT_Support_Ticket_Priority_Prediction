import io
import pandas as pd

def generate_csv_report(df_tickets):
    """Generates a downloadable CSV summary report."""
    output = io.BytesIO()
    summary_df = df_tickets.groupby(['priority', 'product_area']).size().reset_index(name='ticket_count')
    csv_bytes = summary_df.to_csv(index=False).encode('utf-8')
    return csv_bytes

def get_executive_summary_data(df_tickets):
    """Calculates high-level metrics for the executive report."""
    total = len(df_tickets)
    urgent_count = len(df_tickets[df_tickets['priority'] == 'urgent']) if 'priority' in df_tickets.columns else 0
    high_count = len(df_tickets[df_tickets['priority'] == 'high']) if 'priority' in df_tickets.columns else 0
    
    return {
        'total_tickets': total,
        'urgent_tickets': urgent_count,
        'high_tickets': high_count,
        'sla_risk_percentage': round((urgent_count / total * 100), 2) if total > 0 else 0
    }