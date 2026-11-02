# ==========================================
# 1. IMPORT LIBRARIES & SETUP
# ==========================================
import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.sparse import hstack
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI Support Desk & Ticket Triage",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK Corpora safely
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# Define Base Directory Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '../models')
DATASET_PATH = os.path.join(BASE_DIR, '../datasets/cleaned_tickets.csv')
FEEDBACK_LOG_PATH = os.path.join(BASE_DIR, '../datasets/feedback_logs.csv')

# Load Model Artifacts
@st.cache_resource
def load_artifacts():
    tfidf = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    ohe = joblib.load(os.path.join(MODEL_DIR, 'category_encoder.pkl'))
    model = joblib.load(os.path.join(MODEL_DIR, 'priority_classifier_model.pkl'))
    return tfidf, ohe, model

# Fast-loading cached dataset loader
@st.cache_data(show_spinner="Loading ticket dataset...")
def load_dataset():
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(
            DATASET_PATH,
            engine='c',
            low_memory=False
        )
        df['cleaned_message'] = df['cleaned_message'].fillna('')
        
        # Keep columns as standard objects to prevent Categorical aggregation errors with Plotly/Pandas
        string_cols = ['product_area', 'issue_type', 'customer_segment', 'channel', 'platform', 'priority']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        return df
    else:
        return pd.DataFrame({
            'initial_message': ['System crash on login', 'Invoice address update needed'],
            'cleaned_message': ['system crash login', 'invoice address update needed'],
            'product_area': ['authentication', 'billing'],
            'issue_type': ['bug', 'account_access'],
            'customer_segment': ['enterprise', 'individual'],
            'channel': ['portal', 'email'],
            'platform': ['web', 'windows'],
            'priority': ['urgent', 'low']
        })

tfidf, ohe, model = load_artifacts()
df_tickets = load_dataset()

# Initialize Navigation Session State
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Dataset Overview"

# ==========================================
# 2. CUSTOM PROFESSIONAL CSS STYLING
# ==========================================
st.markdown("""
<style>
    /* Global Background & Base Typography */
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p {
        color: #F8FAFC !important;
    }
    
    /* Sidebar Radio Buttons */
    div[data-testid="stSidebar"] div[role="radiogroup"] > label {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        color: #CBD5E1 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease-in-out !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        border-color: #0EA5E9 !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {
        background-color: #0284C7 !important;
        border-color: #38BDF8 !important;
        color: #FFFFFF !important;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }

    /* Explanation Banner */
    .info-card {
        background-color: #EFF6FF;
        border-left: 4px solid #0284C7;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 24px;
    }
    
    /* Filter Bar Header Accent */
    .filter-section-title {
        color: #0F172A;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        letter-spacing: 0.3px;
    }
    
    /* Dropdown Styling */
    .stSelectbox label, .stTextInput label {
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        margin-bottom: 4px !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        color: #0F172A !important;
        font-weight: 500 !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
    }
    .stSelectbox div[data-baseweb="select"]:hover {
        border-color: #0284C7 !important;
    }

    /* Banners */
    .sla-risk-banner {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        color: #991B1B;
        padding: 14px;
        border-radius: 8px;
        font-weight: 700;
        margin-bottom: 16px;
        text-align: center;
    }
    .highlight-keyword {
        background-color: #FEF08A;
        color: #854D0E;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
    }
    
    /* Buttons */
    .stMainBlockContainer .stButton > button {
        background-color: #0284C7 !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        width: 100%;
        box-shadow: 0 4px 6px -1px rgba(2, 132, 199, 0.3) !important;
    }
    .stMainBlockContainer .stButton > button:hover {
        background-color: #0369A1 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. HELPER FUNCTIONS & BUSINESS LOGIC
# ==========================================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

SLA_MAP = {
    'urgent': {'time': '2 Hours SLA', 'color': '#EF4444'},
    'high': {'time': '8 Hours SLA', 'color': '#F97316'},
    'medium': {'time': '24 Hours SLA', 'color': '#3B82F6'},
    'low': {'time': '48 Hours SLA', 'color': '#10B981'}
}

ROUTING_MAP = {
    'authentication': 'Identity & Access Team (IAM)',
    'billing': 'Billing & Financial Ops',
    'api_integration': 'Developer Relations & API Team',
    'data_export': 'Data Engineering Group',
    'reporting': 'Business Intelligence Support',
    'mobile_app': 'Mobile Client Support'
}

CRITICAL_KEYWORDS = ['crash', 'breach', 'emergency', 'outage', 'offline', 'locked out', 'database down', 'charged twice', 'billed double']

def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

def highlight_keywords_in_text(raw_text):
    pattern = re.compile(r'\b(' + '|'.join(re.escape(word) for word in CRITICAL_KEYWORDS) + r')\b', re.IGNORECASE)
    highlighted = pattern.sub(r'<span class="highlight-keyword">\1</span>', raw_text)
    return highlighted

def apply_domain_business_rules(raw_text, issue_type, raw_probs, classes):
    prob_dict = dict(zip(classes, raw_probs))
    text_lower = raw_text.lower()
    
    urgent_keywords = ['crash', 'breach', 'emergency', 'outage', 'offline', 'locked out', 'database down']
    high_keywords = ['charged twice', 'billed double', 'duplicate charge', '2fa not working']
    low_keywords = ['billing address', 'contact name', 'update invoice', 'customize dashboard']
    
    if any(kw in text_lower for kw in urgent_keywords) or issue_type == 'security_concern':
        prob_dict['urgent'] = prob_dict.get('urgent', 0) + 0.55
        prob_dict['high'] = prob_dict.get('high', 0) + 0.25
    elif any(kw in text_lower for kw in high_keywords):
        prob_dict['high'] = prob_dict.get('high', 0) + 0.50
        prob_dict['medium'] = prob_dict.get('medium', 0) + 0.20
    elif any(kw in text_lower for kw in low_keywords):
        prob_dict['low'] = prob_dict.get('low', 0) + 0.55

    total_score = sum(prob_dict.values())
    adjusted_probs = {k: v / total_score for k, v in prob_dict.items()}
    
    top_class = max(adjusted_probs, key=adjusted_probs.get)
    confidence = adjusted_probs[top_class] * 100
    
    return top_class, adjusted_probs, confidence

def predict_single_ticket(text, prod, issue, seg, chan, plat):
    cleaned_msg = clean_text(text)
    X_text = tfidf.transform([cleaned_msg])
    meta_df = pd.DataFrame([{
        'product_area': prod,
        'issue_type': issue,
        'customer_segment': seg,
        'channel': chan,
        'platform': plat
    }])
    X_cat = ohe.transform(meta_df)
    X_full = hstack([X_text, X_cat])
    raw_probs = model.predict_proba(X_full)[0]
    return apply_domain_business_rules(text, issue, raw_probs, model.classes_)

def save_feedback(raw_text, original_pred, corrected_pred, agent_notes):
    os.makedirs(os.path.dirname(FEEDBACK_LOG_PATH), exist_ok=True)
    new_entry = pd.DataFrame([{
        'timestamp': pd.Timestamp.now(),
        'ticket_text': raw_text,
        'original_prediction': original_pred,
        'corrected_priority': corrected_pred,
        'agent_notes': agent_notes
    }])
    if os.path.exists(FEEDBACK_LOG_PATH):
        new_entry.to_csv(FEEDBACK_LOG_PATH, mode='a', header=False, index=False)
    else:
        new_entry.to_csv(FEEDBACK_LOG_PATH, mode='w', header=True, index=False)

def format_feature_name(name):
    """Converts raw model feature names into clean, non-technical labels."""
    if name.startswith('product_area_'):
        return f"Product Area: {name.replace('product_area_', '').replace('_', ' ').title()}"
    elif name.startswith('issue_type_'):
        return f"Issue Type: {name.replace('issue_type_', '').replace('_', ' ').title()}"
    elif name.startswith('customer_segment_'):
        return f"Customer Tier: {name.replace('customer_segment_', '').replace('_', ' ').title()}"
    elif name.startswith('channel_'):
        return f"Channel: {name.replace('channel_', '').replace('_', ' ').title()}"
    elif name.startswith('platform_'):
        return f"Platform: {name.replace('platform_', '').replace('_', ' ').title()}"
    else:
        return f"Word Keyword: '{name.replace('_', ' ')}'"

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='margin-bottom:0; color:#FFFFFF; font-weight:700; font-size:1.35rem;'>AI SUPPORT DESK</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#38BDF8; font-size:0.8rem; font-weight:500; margin-top:2px;'>Automated Priority Classification & Intelligent Team Routing</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("<p style='color:#94A3B8; font-weight:600; font-size:0.75rem; letter-spacing:1px; margin-bottom:12px;'>NAVIGATION MENU</p>", unsafe_allow_html=True)
    
    nav_option = st.radio(
        label="Select Page",
        options=[
            "Dataset Overview",
            "Charts & Analytics",
            "Model Details & XAI",
            "Single Ticket Prediction"
        ],
        label_visibility="collapsed"
    )

# ==========================================
# 5. PAGE VIEW ROUTING
# ==========================================

# ------------------------------------------
# PAGE 1: DATASET OVERVIEW
# ------------------------------------------
if nav_option == "Dataset Overview":
    st.markdown("<h1 style='color:#0F172A; font-weight:700;'>Dataset Overview & Data Health</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#475569;'>Comprehensive analysis and inspection of support ticket ingestion records across model training iterations.</p>", unsafe_allow_html=True)

    # Calculate Data Health Metrics
    total_records = len(df_tickets)
    null_count = df_tickets.isnull().sum().sum()
    total_cells = df_tickets.shape[0] * df_tickets.shape[1]
    data_quality_score = ((total_cells - null_count) / total_cells) * 100 if total_cells > 0 else 100

    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    with col_m1:
        st.markdown(f"""
        <div class='metric-card'>
            <p style='color:#64748B; font-size:0.78rem; font-weight:600; margin:0;'>TOTAL TICKETS</p>
            <h2 style='color:#0F172A; margin:4px 0 0 0; font-weight:700;'>{total_records:,}</h2>
            <p style='color:#64748B; font-size:0.75rem; margin:2px 0 0 0;'>Ingested rows</p>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div class='metric-card'>
            <p style='color:#64748B; font-size:0.78rem; font-weight:600; margin:0;'>PRODUCT AREAS</p>
            <h2 style='color:#0F172A; margin:4px 0 0 0; font-weight:700;'>{df_tickets['product_area'].nunique() if 'product_area' in df_tickets.columns else 0}</h2>
            <p style='color:#64748B; font-size:0.75rem; margin:2px 0 0 0;'>Unique modules</p>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
        <div class='metric-card'>
            <p style='color:#64748B; font-size:0.78rem; font-weight:600; margin:0;'>ISSUE CATEGORIES</p>
            <h2 style='color:#0F172A; margin:4px 0 0 0; font-weight:700;'>{df_tickets['issue_type'].nunique() if 'issue_type' in df_tickets.columns else 0}</h2>
            <p style='color:#64748B; font-size:0.75rem; margin:2px 0 0 0;'>Unique categories</p>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        st.markdown("""
        <div class='metric-card'>
            <p style='color:#64748B; font-size:0.78rem; font-weight:600; margin:0;'>PREDICTION CLASSES</p>
            <h2 style='color:#0284C7; margin:4px 0 0 0; font-weight:700;'>4 Levels</h2>
            <p style='color:#64748B; font-size:0.75rem; margin:2px 0 0 0;'>Urgent/High/Med/Low</p>
        </div>
        """, unsafe_allow_html=True)
    with col_m5:
        st.markdown(f"""
        <div class='metric-card'>
            <p style='color:#64748B; font-size:0.78rem; font-weight:600; margin:0;'>DATA QUALITY</p>
            <h2 style='color:#10B981; margin:4px 0 0 0; font-weight:700;'>{data_quality_score:.1f}%</h2>
            <p style='color:#64748B; font-size:0.75rem; margin:2px 0 0 0;'>Schema completeness</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- INTERACTIVE DATASET SEARCH & FILTER PANEL ---
    st.markdown("<div class='filter-section-title'>DATA EXPLORER & SEARCH FILTERS</div>", unsafe_allow_html=True)
    
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    
    prod_list = ['All'] + sorted(df_tickets['product_area'].dropna().unique().tolist()) if 'product_area' in df_tickets.columns else ['All']
    issue_list = ['All'] + sorted(df_tickets['issue_type'].dropna().unique().tolist()) if 'issue_type' in df_tickets.columns else ['All']
    prio_list = ['All'] + sorted(df_tickets['priority'].dropna().unique().tolist()) if 'priority' in df_tickets.columns else ['All']
    
    with col_f1:
        sel_prod = st.selectbox("Filter Product Area", prod_list, key="ov_prod")
    with col_f2:
        sel_issue = st.selectbox("Filter Issue Category", issue_list, key="ov_issue")
    with col_f3:
        sel_prio = st.selectbox("Filter Priority Tag", prio_list, key="ov_prio")
    with col_f4:
        search_query = st.text_input("Search Text Narrative", placeholder="e.g. crash, login, invoice...", key="ov_search")

    filtered_overview_df = df_tickets.copy()
    
    if sel_prod != 'All':
        filtered_overview_df = filtered_overview_df[filtered_overview_df['product_area'] == sel_prod]
    if sel_issue != 'All':
        filtered_overview_df = filtered_overview_df[filtered_overview_df['issue_type'] == sel_issue]
    if sel_prio != 'All' and 'priority' in filtered_overview_df.columns:
        filtered_overview_df = filtered_overview_df[filtered_overview_df['priority'] == sel_prio]
    if search_query.strip():
        filtered_overview_df = filtered_overview_df[
            filtered_overview_df['initial_message'].astype(str).str.contains(search_query, case=False, na=False)
        ]

    st.markdown(f"<p style='color:#64748B; font-size:0.85rem; margin-top:6px;'>Showing <strong>{len(filtered_overview_df):,}</strong> matching records out of <strong>{len(df_tickets):,}</strong> total entries</p>", unsafe_allow_html=True)

    # Data Table View
    st.dataframe(filtered_overview_df, use_container_width=True, height=380)

    # --- EXPANDABLE SCHEMA DETAILS ---
    with st.expander("View Dataset Schema & Field Descriptions"):
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("""
            #### Primary Data Fields
            * **`initial_message`**: Raw ticket narrative text submitted by end users.
            * **`cleaned_message`**: Lemmatized & tokenized text output ready for TF-IDF encoding.
            * **`product_area`**: System module associated with the issue (e.g., Billing, Mobile App).
            * **`issue_type`**: Classification of failure type (e.g., Bug, Security Concern).
            """)
        with col_s2:
            st.markdown("""
            #### Metadata Attributes
            * **`customer_segment`**: Tier of customer account (Enterprise, Small Business, Individual).
            * **`channel`**: Channel used to open ticket (Web Form, In-App, Email, Chat).
            * **`platform`**: Platform OS/Client (Web, iOS, Android, Windows).
            * **`priority`**: Ground truth SLA target (Urgent, High, Medium, Low).
            """)

# ------------------------------------------
# PAGE 2: CHARTS & ANALYTICS
# ------------------------------------------
elif nav_option == "Charts & Analytics":
    st.markdown("<h1 style='color:#0F172A; font-weight:700;'>Charts & Interactive Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#475569;'>Interactive distribution visualizations and text keyword extraction across ticket logs.</p>", unsafe_allow_html=True)
    
    st.markdown("<div class='filter-section-title'>GLOBAL ANALYTICS FILTERS</div>", unsafe_allow_html=True)
    
    prod_options = ['All'] + sorted(df_tickets['product_area'].dropna().unique().tolist()) if 'product_area' in df_tickets.columns else ['All']
    seg_options = ['All'] + sorted(df_tickets['customer_segment'].dropna().unique().tolist()) if 'customer_segment' in df_tickets.columns else ['All']
    chan_options = ['All'] + sorted(df_tickets['channel'].dropna().unique().tolist()) if 'channel' in df_tickets.columns else ['All']
    prio_options = ['All'] + ['urgent', 'high', 'medium', 'low']
    
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        selected_prod = st.selectbox("Product Area", prod_options)
    with f_col2:
        selected_seg = st.selectbox("Customer Segment", seg_options)
    with f_col3:
        selected_chan = st.selectbox("Channel", chan_options)
    with f_col4:
        selected_prio = st.selectbox("Priority Level", prio_options)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    filtered_df = df_tickets.copy()
    if selected_prod != 'All':
        filtered_df = filtered_df[filtered_df['product_area'] == selected_prod]
    if selected_seg != 'All':
        filtered_df = filtered_df[filtered_df['customer_segment'] == selected_seg]
    if selected_chan != 'All':
        filtered_df = filtered_df[filtered_df['channel'] == selected_chan]
    if selected_prio != 'All' and 'priority' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['priority'] == selected_prio]

    st.markdown(f"<p style='color:#475569; font-size:0.9rem;'>Displaying <strong>{len(filtered_df):,}</strong> filtered ticket records out of <strong>{len(df_tickets):,}</strong> total</p>", unsafe_allow_html=True)
    
    if len(filtered_df) == 0:
        st.warning("No tickets match the selected filter combination. Please adjust your filter selections.")
    else:
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.markdown("<h3 style='color:#0F172A; font-weight:600;'>Priority Class Distribution</h3>", unsafe_allow_html=True)
            if 'priority' in filtered_df.columns:
                p_counts = filtered_df['priority'].value_counts().reset_index()
                p_counts.columns = ['Priority Level', 'Ticket Count']
                
                fig1 = px.bar(
                    p_counts, x='Priority Level', y='Ticket Count', color='Priority Level',
                    color_discrete_map={'low': '#10B981', 'medium': '#3B82F6', 'high': '#F97316', 'urgent': '#EF4444'},
                    text='Ticket Count'
                )
                fig1.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#0F172A'), xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='#E2E8F0'), showlegend=False,
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                fig1.update_traces(textposition='outside')
                st.plotly_chart(fig1, use_container_width=True)
                
        with col_c2:
            st.markdown("<h3 style='color:#0F172A; font-weight:600;'>Tickets by Product Area</h3>", unsafe_allow_html=True)
            if 'product_area' in filtered_df.columns:
                pa_counts = filtered_df['product_area'].value_counts().reset_index()
                pa_counts.columns = ['Product Area', 'Ticket Count']
                fig2 = px.bar(
                    pa_counts, y='Product Area', x='Ticket Count', orientation='h',
                    color_discrete_sequence=['#0284C7'], text='Ticket Count'
                )
                fig2.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#0F172A'), xaxis=dict(showgrid=True, gridcolor='#E2E8F0'),
                    yaxis=dict(showgrid=False, categoryorder='total ascending'),
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                fig2.update_traces(textposition='outside')
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        
        col_c3, col_c4 = st.columns(2)
        with col_c3:
            st.markdown("<h3 style='color:#0F172A; font-weight:600;'>Channel Distribution</h3>", unsafe_allow_html=True)
            if 'channel' in filtered_df.columns:
                ch_counts = filtered_df['channel'].value_counts().reset_index()
                ch_counts.columns = ['Channel', 'Count']
                fig3 = px.pie(
                    ch_counts, names='Channel', values='Count', hole=0.45,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#0F172A'), margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig3, use_container_width=True)
                
        with col_c4:
            st.markdown("<h3 style='color:#0F172A; font-weight:600;'>Issue Type Breakdown</h3>", unsafe_allow_html=True)
            if 'issue_type' in filtered_df.columns:
                it_counts = filtered_df['issue_type'].value_counts().reset_index()
                it_counts.columns = ['Issue Type', 'Count']
                fig4 = px.pie(
                    it_counts, names='Issue Type', values='Count', hole=0.45,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#0F172A'), margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig4, use_container_width=True)

        st.markdown("---")
        
        col_c5, col_c6 = st.columns(2)
        
        with col_c5:
            st.markdown("<h3 style='color:#0F172A; font-weight:600;'>Multi-Level Category Hierarchy</h3>", unsafe_allow_html=True)
            if all(col in filtered_df.columns for col in ['product_area', 'issue_type', 'priority']):
                sun_df = filtered_df.astype({'product_area': str, 'issue_type': str, 'priority': str})
                fig_sun = px.sunburst(
                    sun_df, path=['product_area', 'issue_type', 'priority'],
                    color='priority',
                    color_discrete_map={'low': '#10B981', 'medium': '#3B82F6', 'high': '#F97316', 'urgent': '#EF4444'}
                )
                fig_sun.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#0F172A'), margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_sun, use_container_width=True)
                
        with col_c6:
            st.markdown("<h3 style='color:#0F172A; font-weight:600;'>Top Keyword Frequency</h3>", unsafe_allow_html=True)
            all_text = " ".join(filtered_df['cleaned_message'].dropna().astype(str)).split()
            word_counts = Counter(all_text).most_common(15)
            
            if word_counts:
                wc_df = pd.DataFrame(word_counts, columns=['Keyword', 'Frequency']).sort_values('Frequency', ascending=True)
                fig_bar_kw = px.bar(
                    wc_df, y='Keyword', x='Frequency', orientation='h',
                    color_discrete_sequence=['#38BDF8'], text='Frequency'
                )
                fig_bar_kw.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#0F172A'), xaxis=dict(showgrid=True, gridcolor='#E2E8F0'),
                    yaxis=dict(showgrid=False), margin=dict(l=20, r=20, t=20, b=20)
                )
                fig_bar_kw.update_traces(textposition='outside')
                st.plotly_chart(fig_bar_kw, use_container_width=True)

# ------------------------------------------
# PAGE 3: MODEL DETAILS & XAI FEATURE IMPORTANCE
# ------------------------------------------
elif nav_option == "Model Details & XAI":
    st.markdown("<h1 style='color:#0F172A; font-weight:700;'>Model Configuration & Explainability (XAI)</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#475569;'>Understand how the AI model evaluates ticket text and customer details to make priority decisions.</p>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-card'>
        <h4 style='color:#0369A1; margin:0 0 6px 0; font-weight:700;'>How to Read This Dashboard</h4>
        <p style='color:#1E293B; margin:0; font-size:0.92rem; line-height:1.6;'>
            Every support ticket has multiple details (such as the submission channel, customer tier, or words like "crash"). 
            The AI model weighs these factors together to assign a final priority score. <strong>Longer bars indicate a factor that has a higher impact on decisions.</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_x1, col_x2, col_x3 = st.columns(3)
    with col_x1:
        st.markdown("""
        <div class='metric-card'>
            <p style='color:#64748B; font-size:0.8rem; font-weight:600; margin:0;'>MODEL ARCHITECTURE</p>
            <h3 style='color:#0F172A; margin:4px 0 0 0; font-weight:700;'>Random Forest</h3>
            <p style='color:#64748B; font-size:0.82rem; margin:4px 0 0 0;'>Combines 150 decision trees for balanced prediction accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_x2:
        st.markdown("""
        <div class='metric-card'>
            <p style='color:#64748B; font-size:0.8rem; font-weight:600; margin:0;'>SIGNALS ANALYZED</p>
            <h3 style='color:#0F172A; margin:4px 0 0 0; font-weight:700;'>1,525 Features</h3>
            <p style='color:#64748B; font-size:0.82rem; margin:4px 0 0 0;'>Includes 1,500 word phrases and 25 ticket metadata attributes.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_x3:
        st.markdown("""
        <div class='metric-card'>
            <p style='color:#64748B; font-size:0.8rem; font-weight:600; margin:0;'>TOP INFLUENCING CATEGORY</p>
            <h3 style='color:#0284C7; margin:4px 0 0 0; font-weight:700;'>Submission Channel</h3>
            <p style='color:#64748B; font-size:0.82rem; margin:4px 0 0 0;'>Tickets submitted in-app or via mobile forms carry strong weight.</p>
        </div>
        """, unsafe_allow_html=True)

    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        top_n = st.selectbox("Number of Factors to Display", [10, 15, 20], index=1)
    with col_f2:
        feature_filter = st.selectbox(
            "Filter Factors by Category", 
            ["All Factors", "Metadata Only (Channel, Tier, Platform)", "Keywords Only (Text Phrases)"]
        )

    tfidf_feature_names = tfidf.get_feature_names_out()
    cat_feature_names = ohe.get_feature_names_out()
    all_feature_names = np.concatenate([tfidf_feature_names, cat_feature_names])
    importances = model.feature_importances_
    
    fi_df = pd.DataFrame({
        'Raw_Feature': all_feature_names,
        'Formatted_Feature': [format_feature_name(f) for f in all_feature_names],
        'Importance': importances
    })

    if feature_filter == "Metadata Only (Channel, Tier, Platform)":
        fi_df = fi_df[fi_df['Formatted_Feature'].str.contains("Product Area|Issue Type|Customer Tier|Channel|Platform")]
    elif feature_filter == "Keywords Only (Text Phrases)":
        fi_df = fi_df[fi_df['Formatted_Feature'].str.contains("Word Keyword")]

    fi_df = fi_df.sort_values('Importance', ascending=False).head(top_n).sort_values('Importance', ascending=True)

    fig_fi = px.bar(
        fi_df, y='Formatted_Feature', x='Importance', orientation='h',
        color='Importance',
        color_continuous_scale=['#BAE6FD', '#0284C7', '#0F172A'],
        text_auto='.3f'
    )
    
    fig_fi.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#0F172A', size=13),
        xaxis=dict(title="Factor Importance Weight (Higher Value = Stronger Decision Impact)", showgrid=True, gridcolor='#E2E8F0'),
        yaxis=dict(title="", showgrid=False),
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=30, b=20),
        height=520
    )
    fig_fi.update_traces(
        textposition='outside',
        hovertemplate="<b>Decision Factor:</b> %{y}<br><b>Importance Weight:</b> %{x:.4f}<extra></extra>"
    )
    
    st.markdown(f"<h3 style='color:#0F172A; font-weight:600;'>Top {top_n} Factors Influencing AI Predictions</h3>", unsafe_allow_html=True)
    st.plotly_chart(fig_fi, use_container_width=True)

    with st.expander("Learn How the AI Model Evaluates Tickets (Plain-English Summary)"):
        st.markdown("""
        #### 1. Submission Channel Impact
        * **In-App & Email Forms:** Support requests submitted directly through embedded app widgets or emergency support forms naturally carry high decision weight, as they typically represent active user sessions or critical operational issues.

        #### 2. Customer Tier Prioritization
        * **Enterprise vs. Individual:** Tickets coming from enterprise-tier accounts automatically receive a priority score boost to align with strict service-level agreements (SLAs).

        #### 3. Text & Keyword Analysis (NLP)
        * **Trigger Words:** Phrases like `"crash"`, `"database down"`, `"security breach"`, or `"charged twice"` instantly push predictions toward **URGENT** or **HIGH** priority levels.
        * **Standard Requests:** Inquiries containing words like `"billing address"`, `"update contact"`, or `"dashboard customization"` are routed to **LOW** or **MEDIUM** queues.
        """)

# ------------------------------------------
# PAGE 4: SINGLE TICKET PREDICTION
# ------------------------------------------
elif nav_option == "Single Ticket Prediction":
    st.markdown("<h1 style='color:#0F172A; font-weight:700;'>Predict Ticket Priority & Auto-Route</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        initial_message = st.text_area("Issue Details / Narrative", height=180, placeholder="Enter ticket description...")
        if initial_message.strip():
            st.markdown("<strong>Detected Trigger Keywords:</strong>", unsafe_allow_html=True)
            st.markdown(highlight_keywords_in_text(initial_message), unsafe_allow_html=True)
        
    with col2:
        product_area = st.selectbox("Product Area", ['authentication', 'billing', 'api_integration', 'data_export', 'reporting', 'mobile_app'])
        issue_type = st.selectbox("Issue Type", ['security_concern', 'bug', 'account_access', 'performance', 'feature_request'])
        customer_segment = st.selectbox("Customer Segment", ['enterprise', 'small_business', 'individual'])
        channel = st.selectbox("Channel", ['portal', 'email', 'chat', 'in_app', 'phone_transcript', 'web_form'])
        platform = st.selectbox("Platform", ['web', 'windows', 'ios', 'android', 'mac', 'api_client'])

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("RUN PRIORITY PREDICTION"):
        if initial_message.strip():
            prediction, adj_probs, confidence = predict_single_ticket(
                initial_message, product_area, issue_type, customer_segment, channel, platform
            )
            
            st.markdown("---")
            
            sla_info = SLA_MAP.get(prediction, {'time': 'N/A', 'color': '#64748B'})
            target_team = ROUTING_MAP.get(product_area, 'General Support Desk')
            
            if prediction == 'urgent':
                st.markdown("<div class='sla-risk-banner'>SLA AT RISK — URGENT PRIORITY TICKET (RESOLVE WITHIN 2 HOURS)</div>", unsafe_allow_html=True)
                
            res_c1, res_c2, res_c3 = st.columns(3)
            with res_c1:
                st.markdown(f"<div class='metric-card'><h4>PREDICTED PRIORITY</h4><h2 style='color:{sla_info['color']};'>{prediction.upper()}</h2></div>", unsafe_allow_html=True)
            with res_c2:
                st.markdown(f"<div class='metric-card'><h4>ESTIMATED SLA</h4><h2>{sla_info['time']}</h2></div>", unsafe_allow_html=True)
            with res_c3:
                st.markdown(f"<div class='metric-card'><h4>AUTO-ROUTED TEAM</h4><h3 style='color:#0284C7;'>{target_team}</h3></div>", unsafe_allow_html=True)

            st.markdown("### Human Agent Override / Feedback")
            with st.form("feedback_form"):
                corrected = st.selectbox("Correct Priority (if inaccurate)", ['urgent', 'high', 'medium', 'low'], index=['urgent', 'high', 'medium', 'low'].index(prediction))
                notes = st.text_input("Feedback / Reason for Override")
                submit_fb = st.form_submit_button("Submit Agent Feedback")
                if submit_fb:
                    save_feedback(initial_message, prediction, corrected, notes)
                    st.success("Feedback logged to database for continuous model retraining!")