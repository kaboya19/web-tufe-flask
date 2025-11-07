from flask import Flask, render_template, request, send_file, flash, redirect, url_for, jsonify
import pandas as pd
import numpy as np
import json
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils
from datetime import datetime
import csv
from dateutil.parser import parse
import os
from gspread.exceptions import APIError, SpreadsheetNotFound
import base64
from oauth2client.service_account import ServiceAccountCredentials
import gspread
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pywebpush import webpush, WebPushException
import sqlite3
from cryptography.hazmat.primitives import serialization
import base64

app = Flask(__name__)
app.secret_key = os.urandom(24).hex()  # Güvenli, rastgele bir secret key oluştur

# Web Push Notifications Configuration
VAPID_PUBLIC_KEY = os.environ.get('VAPID_PUBLIC_KEY', '')
VAPID_PRIVATE_KEY = os.environ.get('VAPID_PRIVATE_KEY', '')
VAPID_CLAIM_EMAIL = os.environ.get('VAPID_CLAIM_EMAIL', 'mailto:webtufe@example.com')

def get_vapid_private_key_for_webpush():
    """
    Get VAPID private key in the format that pywebpush expects.
    pywebpush expects base64 URL-safe encoded DER format (original format from generate_vapid_keys.py)
    """
    if not VAPID_PRIVATE_KEY:
        return None
    try:
        # If it's already in PEM format, convert it to DER then to base64 URL-safe
        if '-----BEGIN' in VAPID_PRIVATE_KEY:
            # Load PEM format
            private_key = serialization.load_pem_private_key(
                VAPID_PRIVATE_KEY.encode('utf-8'),
                password=None
            )
            # Convert to DER
            private_key_der = private_key.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            # Convert to base64 URL-safe (without padding)
            private_key_b64 = base64.urlsafe_b64encode(private_key_der).decode('utf-8').rstrip('=')
            return private_key_b64
        
        # If it's already in base64 URL-safe format (most common case)
        # Just return it as is, but ensure it's properly formatted
        # Remove any whitespace
        key = VAPID_PRIVATE_KEY.strip().replace('\n', '').replace('\r', '')
        
        # Validate it's base64 URL-safe
        try:
            # Add padding if needed for validation
            padding = len(key) % 4
            if padding:
                test_key = key + ('=' * (4 - padding))
            else:
                test_key = key
            # Try to decode to verify it's valid
            base64.urlsafe_b64decode(test_key)
            return key
        except Exception:
            # If validation fails, try to convert from DER
            # This handles edge cases
            padding = len(key) % 4
            if padding:
                private_key_b64 = key + ('=' * (4 - padding))
            else:
                private_key_b64 = key
            
            private_key_der = base64.urlsafe_b64decode(private_key_b64)
            private_key = serialization.load_der_private_key(
                private_key_der,
                password=None
            )
            # Convert back to base64 URL-safe
            der_bytes = private_key.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            return base64.urlsafe_b64encode(der_bytes).decode('utf-8').rstrip('=')
            
    except Exception as e:
        print(f"Error processing VAPID private key: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

# Initialize database for push subscriptions
def init_push_db():
    conn = sqlite3.connect('push_subscriptions.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            endpoint TEXT UNIQUE NOT NULL,
            p256dh TEXT NOT NULL,
            auth TEXT NOT NULL,
            user_agent TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database on startup
init_push_db()




def get_turkish_month(date_str):
    aylar = ['Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran', 'Temmuz', 'Ağustos', 'Eylül', 'Ekim', 'Kasım', 'Aralık']
    try:
        if len(date_str) == 7:  # 'YYYY-MM'
            y, m = date_str.split('-')
            return aylar[int(m)-1]
        else:  # 'YYYY-MM-DD'
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            return aylar[date_obj.month-1]
    except:
        return date_str
#test
def get_google_credentials():
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    
    # Get credentials from environment variable
    credentials_json = os.environ.get('GOOGLE_CREDENTIALS_BASE64')
    if not credentials_json:
        raise ValueError("GOOGLE_CREDENTIALS_BASE64 environment variable is not set")
    
    # Decode base64 and parse the JSON string from environment variable
    try:
        decoded_json = base64.b64decode(credentials_json).decode('utf-8')
        credentials_dict = json.loads(decoded_json)
        creds = ServiceAccountCredentials.from_json_keyfile_dict(credentials_dict, scope)
        return creds
    except Exception as e:
        raise ValueError(f"Failed to decode credentials: {str(e)}")

def get_google_credentials_2():
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    
    # Get credentials from environment variable
    credentials_json = os.environ.get('GOOGLE_CREDENTIALS_2_BASE64')
    if not credentials_json:
        raise ValueError("GOOGLE_CREDENTIALS_2_BASE64 environment variable is not set")
    
    # Decode base64 and parse the JSON string from environment variable
    try:
        decoded_json = base64.b64decode(credentials_json).decode('utf-8')
        credentials_dict = json.loads(decoded_json)
        creds = ServiceAccountCredentials.from_json_keyfile_dict(credentials_dict, scope)
        return creds
    except Exception as e:
        raise ValueError(f"Failed to decode credentials: {str(e)}")

def get_google_sheets_data():
    # Google Sheets API setup
    creds = get_google_credentials()
    client = gspread.authorize(creds)
    
    # Open the spreadsheet
    spreadsheet = client.open_by_key('14iiu_MQwtMxHTFt6ceyFhkk6v0OL-wuoQS1IGPzSpNE')
    
    # Get the specific worksheet by title
    worksheet = spreadsheet.worksheet('Ana Gruplar Aylık Değişimler')
    
    # Get all values
    data = worksheet.get_all_values()
    
    # Convert to DataFrame
    df = pd.DataFrame(data[1:], columns=data[0])

    df=pd.read_csv("gruplaraylık.csv",index_col=0)
    
    # Get the last column
    last_col = df.columns[-1]
    
    # Get Turkish month name from the last column date
    month_name = get_turkish_month(last_col)
    
    # Create a dictionary of category and value pairs
    categories = df.iloc[:, 0].tolist()  # First column contains categories
    values = df.iloc[:,-1].tolist()
    
    # Convert values to float and handle any non-numeric values
    def convert_value(v):
        try:
            # Remove any whitespace and replace comma with dot
            v = str(v).strip().replace(',', '.')
            # Handle negative numbers
            if v.startswith('-'):
                return -float(v[1:])
            return float(v)
        except:
            return 0
    
    values = [convert_value(v) for v in values]
    
    # Create pairs and sort by value (highest to lowest)
    pairs = list(zip(categories, values))
    pairs.sort(key=lambda x: x[1], reverse=False)
    
    return pairs, month_name

def get_tufe_data():
    # Google Sheets API setup
    """creds = get_google_credentials_2()
    client = gspread.authorize(creds)
    
    spreadsheet = client.open_by_key('14iiu_MQwtMxHTFt6ceyFhkk6v0OL-wuoQS1IGPzSpNE')
    
    worksheet = spreadsheet.worksheet('Web TÜFE')
    
    data = worksheet.get_all_values()"""

    df=pd.read_csv('tüfe.csv').rename(columns={"Unnamed: 0":"Tarih"})
    
    # Convert to DataFrame
    #df = pd.DataFrame(data[1:], columns=data[0])
    
    # Convert date column to datetime
    df['Tarih'] = pd.to_datetime(df['Tarih'])
    
    # Sort by date
    df = df.sort_values('Tarih')
    
    # Convert TÜFE values to float
    
    return df

def get_monthly_change():
    # Google Sheets API setup
    """creds = get_google_credentials_2()
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key('14iiu_MQwtMxHTFt6ceyFhkk6v0OL-wuoQS1IGPzSpNE')
    worksheet = spreadsheet.get_worksheet_by_id(767776936)
    data = worksheet.get_all_values()
    df = pd.DataFrame(data[1:], columns=data[0])"""

    df=pd.read_csv("gruplaraylık.csv",index_col=0)
   
    last_col = df.columns[-1]
    tufe_row = df[df.iloc[:,0].str.strip().str.lower() == 'web tüfe']
    if not tufe_row.empty:
        value = tufe_row[last_col].values[0]
        try:
            value = float(str(value).replace(',', '.'))
        except:
            value = 0
    else:
        value = 0
    return value, last_col

def get_tufe_vs_tuik_bar_data():
    # Google Sheets API setup
    """creds = get_google_credentials_2()
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key('14iiu_MQwtMxHTFt6ceyFhkk6v0OL-wuoQS1IGPzSpNE')
    worksheet = spreadsheet.get_worksheet_by_id(767776936)
    data = worksheet.get_all_values()
    df = pd.DataFrame(data[1:], columns=data[0])"""

    df=pd.read_csv("gruplaraylık.csv",index_col=0)
    # Get all months (columns except the first)
    months = df.columns[1:]
    # Get TÜFE row
    tufe_row = df[df.iloc[:,0].str.strip().str.lower() == 'web tüfe']
    tufe_values = []
    tufe_months = []
    for col in months:
        val = tufe_row[col].values[0] if not tufe_row.empty else None
        try:
            val = float(str(val).replace(',', '.'))
        except:
            val = None
        tufe_values.append(val)
        tufe_months.append(col)
    # Read TUİK values from CSV
    tuik_df = pd.read_csv('tüik.csv', index_col=0)
    tuik_df.index = pd.to_datetime(tuik_df.index)
    # Her ayın son günündeki değeri al
    tuik_monthly_last = tuik_df.resample('M').last()
    tuik_monthly_last['pct_change'] = tuik_monthly_last['TÜİK'].pct_change() * 100
    tuik_monthly_last = tuik_monthly_last.iloc[1:]  # İlk ayın değişimi NaN olur, atla
    tuik_monthly_last.index = tuik_monthly_last.index.strftime('%Y-%m')
    # TÜFE aylarını 'YYYY-MM' formatına çevir
    tufe_months_fmt = []
    for m in tufe_months:
        if len(m) == 7:
            tufe_months_fmt.append(m)
        elif len(m) == 10:
            tufe_months_fmt.append(m[:7])
        else:
            tufe_months_fmt.append(m)
    # Filtreleme ve eşleştirme
    filtered_months = []
    filtered_tufe = []
    filtered_tuik = []
    for i, m in enumerate(tufe_months_fmt):
        if m >= '2025-02':
            filtered_months.append(m)
            filtered_tufe.append(tufe_values[i])
            if m in tuik_monthly_last.index:
                val = tuik_monthly_last.loc[m, 'pct_change']
                if pd.isna(val):
                    filtered_tuik.append(0)
                else:
                    filtered_tuik.append(val)
            else:
                filtered_tuik.append(0)
    # TÜFE'nin olduğu tüm aylar ve değerleri
    month_labels = [f"{safe_get_turkish_month(m)} {m[:4]}" for m in filtered_months]
    bar_months = month_labels
    bar_tufe = filtered_tufe
    # TÜİK: aynı sırada, yoksa None, 0 ise de None
    tuik_month_map = {m: (v if v != 0 else None) for m, v in zip(filtered_months, filtered_tuik)}
    bar_tuik = [tuik_month_map.get(m, None) for m in filtered_months]
    return bar_months, bar_tufe, bar_tuik

def safe_get_turkish_month(m):
    if len(m) == 7:  # 'YYYY-MM'
        return get_turkish_month(m + '-01')
    elif len(m) == 10:  # 'YYYY-MM-DD'
        return get_turkish_month(m)
    else:
        return m

def get_ana_gruplar_data():
    # Google Sheets API setup
    """creds = get_google_credentials_2()
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key('14iiu_MQwtMxHTFt6ceyFhkk6v0OL-wuoQS1IGPzSpNE')
    worksheet = spreadsheet.get_worksheet_by_id(564638736)
    data = worksheet.get_all_values()
    df = pd.DataFrame(data[1:], columns=data[0])"""

    df=pd.read_csv("gruplar_int.csv").rename(columns={"Unnamed: 0":"Tarih"})
    df['Tarih'] = pd.to_datetime(df['Tarih'])
    df = df.sort_values('Tarih')
    return df

def get_ana_grup_monthly_change(grup_adi):
    """creds = get_google_credentials_2()
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key('14iiu_MQwtMxHTFt6ceyFhkk6v0OL-wuoQS1IGPzSpNE')
    worksheet = spreadsheet.get_worksheet_by_id(767776936)
    data = worksheet.get_all_values()
    df = pd.DataFrame(data[1:], columns=data[0])"""

    df=pd.read_csv("gruplaraylık.csv",index_col=0)
    last_col = df.columns[-1]
    row = df[df.iloc[:,0].str.strip().str.lower() == grup_adi.strip().lower()]
    if not row.empty:
        value = row[last_col].values[0]
        try:
            value = float(str(value).replace(',', '.'))
        except:
            value = 0
    else:
        value = 0
    return value, last_col

def get_ana_grup_all_monthly_changes(grup_adi):
    """creds = get_google_credentials_2()
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key('14iiu_MQwtMxHTFt6ceyFhkk6v0OL-wuoQS1IGPzSpNE')
    worksheet = spreadsheet.get_worksheet_by_id(767776936)
    data = worksheet.get_all_values()
    df = pd.DataFrame(data[1:], columns=data[0])"""

    df=pd.read_csv("gruplaraylık.csv",index_col=0)
    months = df.columns[1:]
    row = df[df.iloc[:,0].str.strip().str.lower() == grup_adi.strip().lower()]
    values = []
    for col in months:
        val = row[col].values[0] if not row.empty else None
        try:
            val = float(str(val).replace(',', '.'))
        except:
            val = None
        values.append(val)
    # Ay etiketleri Türkçe
    month_labels = [f"{safe_get_turkish_month(m)} {m[:4]}" for m in months]
    bar_colors = ['#118AB2'] * len(month_labels)
    font_size = 18 if len(month_labels) < 10 else 16
    return month_labels, values, bar_colors, font_size

def is_date(string):
    try:
        parse(string, dayfirst=True)
        return True
    except:
        return False

def get_last_update_date():
    try:
        with open('time.txt', 'r') as f:
            date_str = f.read().strip()
            date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M')
            return date_obj.strftime('%d.%m.%Y %H:%M')
    except Exception as e:
        return "Bilgi mevcut değil"

def create_monthly_graph(tufe_data):
    try:
        # Convert string data to list of dictionaries if needed
        if isinstance(tufe_data, str):
            tufe_data = json.loads(tufe_data)
        
        # Debug print
        print("TUFE Data Structure:", type(tufe_data))
        print("TUFE Data Sample:", tufe_data[:2] if isinstance(tufe_data, list) else tufe_data)
        
        # Extract data
        if isinstance(tufe_data, list):
            dates = []
            web_tufe = []
            tuik_tufe = []
            for row in tufe_data:
                if isinstance(row, dict):
                    dates.append(row.get('Tarih', ''))
                    web_tufe.append(float(row.get('Web TÜFE', 0)))
                    tuik_tufe.append(float(row.get('TÜİK TÜFE', 0)))
                else:
                    print(f"Unexpected row type: {type(row)}, row: {row}")
        else:
            raise ValueError(f"Unexpected data type: {type(tufe_data)}")
        
        # Create figure
        fig = go.Figure()
        
        # Add traces
        fig.add_trace(go.Scatter(
            x=dates,
            y=web_tufe,
            name='Web TÜFE',
            line=dict(color='#EF476F', width=3),
            hovertemplate='%{y:.2f}%<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=tuik_tufe,
            name='TÜİK TÜFE',
            line=dict(color='#118AB2', width=3),
            hovertemplate='%{y:.2f}%<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Aylık TÜFE Değişim Oranları',
                font=dict(size=24, family='Inter, sans-serif', color='#2B2D42'),
                y=0.95
            ),
            xaxis=dict(
                title='Tarih',
                title_font=dict(size=14, family='Inter, sans-serif', color='#2B2D42'),
                tickfont=dict(size=14, family='Inter, sans-serif', color='#2B2D42'),
                gridcolor='#E9ECEF'
            ),
            yaxis=dict(
                title='Değişim (%)',
                title_font=dict(size=14, family='Inter, sans-serif', color='#2B2D42'),
                tickfont=dict(size=14, family='Inter, sans-serif', color='#2B2D42'),
                gridcolor='#E9ECEF',
                tickformat='.2f'
            ),
            legend=dict(
                font=dict(size=12, family='Inter, sans-serif', color='#2B2D42'),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#E9ECEF',
                borderwidth=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='x unified',
            height=500,
            margin=dict(l=20, r=20, t=80, b=20)
        )
        
        return fig
    except Exception as e:
        print(f"Error in create_monthly_graph: {str(e)}")
        raise

def create_bar_graph(tufe_vs_tuik_data):
    try:
        # Convert string data to list of dictionaries if needed
        if isinstance(tufe_vs_tuik_data, str):
            tufe_vs_tuik_data = json.loads(tufe_vs_tuik_data)
        
        # Debug print
        print("Bar Data Structure:", type(tufe_vs_tuik_data))
        print("Bar Data Sample:", tufe_vs_tuik_data[:2] if isinstance(tufe_vs_tuik_data, list) else tufe_vs_tuik_data)
        
        # Extract data
        if isinstance(tufe_vs_tuik_data, list):
            categories = []
            web_tufe_values = []
            tuik_values = []
            for row in tufe_vs_tuik_data:
                if isinstance(row, dict):
                    categories.append(row.get('Grup', ''))
                    web_tufe_values.append(float(row.get('Web TÜFE', 0)))
                    tuik_values.append(float(row.get('TÜİK', 0)))
                else:
                    print(f"Unexpected row type: {type(row)}, row: {row}")
        else:
            raise ValueError(f"Unexpected data type: {type(tufe_vs_tuik_data)}")
        
        # Create figure
        fig = go.Figure()
        
        # Add bars for Web TÜFE
        fig.add_trace(go.Bar(
            y=categories,
            x=web_tufe_values,
            name='Web TÜFE',
            orientation='h',
            marker_color='#EF476F',
            text=[f'{v:+.2f}%' for v in web_tufe_values],
            textposition='outside',
            hovertemplate='%{x:+.2f}%<extra></extra>'
        ))
        
        # Add bars for TÜİK
        fig.add_trace(go.Bar(
            y=categories,
            x=tuik_values,
            name='TÜİK',
            orientation='h',
            marker_color='#118AB2',
            text=[f'{v:+.2f}%' for v in tuik_values],
            textposition='outside',
            hovertemplate='%{x:+.2f}%<extra></extra>'
        ))
        
        # Calculate x-axis range
        if web_tufe_values and tuik_values:
            all_values = web_tufe_values + tuik_values
            x_min = min(all_values)
            x_max = max(all_values)
            x_range = x_max - x_min
            x_margin = x_range * 0.2 if x_range != 0 else abs(x_max) * 0.2 if x_max != 0 else 1
        else:
            x_min, x_max, x_margin = -1, 1, 0.2
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Ana Gruplar TÜFE Karşılaştırması',
                font=dict(size=24, family='Inter, sans-serif', color='#2B2D42'),
                y=0.95
            ),
            xaxis=dict(
                title='Değişim (%)',
                title_font=dict(size=14, family='Inter, sans-serif', color='#2B2D42'),
                tickfont=dict(size=14, family='Inter, sans-serif', color='#2B2D42'),
                gridcolor='#E9ECEF',
                range=[x_min - x_margin, x_max + x_margin]
            ),
            yaxis=dict(
                title='Ana Gruplar',
                title_font=dict(
                    size=14,
                    family='Inter, sans-serif',
                    color='#2B2D42'
                ),
                tickfont=dict(
                    size=14,
                    family='Inter, sans-serif',
                    color='#2B2D42'
                ),
                gridcolor='#E9ECEF'
            ),
            legend=dict(
                font=dict(size=12, family='Inter, sans-serif', color='#2B2D42'),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#E9ECEF',
                borderwidth=1,
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=600,
            margin=dict(l=20, r=20, t=100, b=20),
            barmode='group',
            bargap=0.15,
            bargroupgap=0.1
        )
        
        return fig
    except Exception as e:
        print(f"Error in create_bar_graph: {str(e)}")
        raise

def is_mobile_device(user_agent):
    """
    User agent string'ini analiz ederek mobil cihaz olup olmadığını belirler
    """
    if not user_agent:
        return False
    
    user_agent = user_agent.lower()
    mobile_keywords = [
        'mobile', 'android', 'iphone', 'ipad', 'ipod', 'blackberry', 
        'windows phone', 'palm', 'symbian', 'nokia', 'samsung',
        'htc', 'lg', 'motorola', 'sony', 'kindle', 'tablet'
    ]
    
    return any(keyword in user_agent for keyword in mobile_keywords)

from flask import send_from_directory
@app.route('/ads.txt')
def ads_txt():
    return send_from_directory('static', 'ads.txt')

@app.route('/')
def redirect_page():
    # Manuel seçim için force parametresi
    force_manual = request.args.get('manual')
    if force_manual:
        return render_template('redirect.html')
    
    # User agent'ı kontrol et
    user_agent = request.headers.get('User-Agent', '')
    
    if is_mobile_device(user_agent):
        # Mobil cihaz - mobil siteye yönlendir
        return redirect('https://webtufemobile.onrender.com?v=2')
    else:
        # Masaüstü cihaz - ana sayfaya yönlendir
        return redirect('/ana-sayfa')

def get_monthly_group_data_for_date(date_str):
    """Get monthly group data for a specific date"""
    try:
        df = pd.read_csv("gruplaraylık.csv", index_col=0)
        
        # Check if the date exists in columns
        if date_str not in df.columns:
            return None, None
            
        # Get the data for the selected date
        data_pairs = []
        for idx, row in df.iterrows():
            group_name = row.iloc[0]  # First column is group name
            value = row[date_str]
            try:
                value = float(str(value).replace(',', '.'))
                data_pairs.append((group_name, value))
            except:
                continue
                
        # Get Turkish month name
        month_name = get_turkish_month(date_str)
        return data_pairs, month_name
    except Exception as e:
        print(f"Error in get_monthly_group_data_for_date: {e}")
        return None, None

def get_available_dates():
    """Get list of available dates from monthly data"""
    try:
        df = pd.read_csv("gruplaraylık.csv", index_col=0)
        # Return column names except the first one (group names), sorted in reverse order (newest first)
        dates = df.columns[1:].tolist()
        dates.sort(reverse=True)
        return dates
    except Exception as e:
        print(f"Error in get_available_dates: {e}")
        return []

@app.route('/ana-sayfa', methods=['GET', 'POST'])
def ana_sayfa():
    try:
        # Get available dates for the dropdown
        available_dates = get_available_dates()
        selected_date = None
        
        # Handle POST request (date selection)
        show_contrib = True
        if request.method == 'POST':
            selected_date = request.form.get('selected_date')
            # Checkbox sends value only when checked
            show_contrib = 'show_contrib' in request.form
            if selected_date:
                # Get data for selected date
                data_pairs, month_name = get_monthly_group_data_for_date(selected_date)
                if data_pairs is None:
                    # Fallback to default data if selected date not found
                    data_pairs, month_name = get_google_sheets_data()
                    flash('Seçilen tarih için veri bulunamadı, varsayılan veri gösteriliyor.', 'warning')
            else:
                # No date selected, use default data
                data_pairs, month_name = get_google_sheets_data()
        else:
            # GET request, use latest available date by default
            if available_dates:
                selected_date = available_dates[0]  # First item is the latest due to reverse sort
                data_pairs, month_name = get_monthly_group_data_for_date(selected_date)
                if data_pairs is None:
                    # Fallback to default data if latest date not found
                    data_pairs, month_name = get_google_sheets_data()
                    selected_date = None  # Clear selected_date to show we're using default data
            else:
                # No dates available, use default data
                data_pairs, month_name = get_google_sheets_data()
                selected_date = None
        
        # Sort data pairs by value in descending order for better visualization
        data_pairs_sorted = sorted(data_pairs, key=lambda x: x[1], reverse=True)
        categories = [pair[0] for pair in data_pairs_sorted]
        values = [pair[1] for pair in data_pairs_sorted]
        
        # Get last update date
        last_update = get_last_update_date()
        
        # Prepare left (monthly change) data
        left_categories = [pair[0] for pair in data_pairs_sorted]
        left_values = [pair[1] for pair in data_pairs_sorted]
        
        valid_values = [v for _, v in data_pairs if v is not None]

        if valid_values:
            x_min = min(valid_values)
            x_max = max(valid_values)

            # Marj hesapla
            x_range = x_max - x_min
            x_margin = x_range * 0.3 if x_range != 0 else abs(x_max) * 0.3

            x_min_with_margin = x_min - x_margin
            x_max_with_margin = x_max + x_margin

            # Sıfıra yaklaşma kontrolü
            if x_min >= 0:
                x_min_with_margin = max(0, x_min - x_margin)
            if x_max <= 0:
                x_max_with_margin = min(0, x_max + x_margin)

        # If user disabled contributions, render single chart and return early
        if not show_contrib:
            def compute_range_single(values):
                vals = [v for v in values if v is not None]
                if not vals:
                    return [0, 1]
                vmin, vmax = min(vals), max(vals)
                vr = vmax - vmin
                m = vr * 0.3 if vr != 0 else max(abs(vmax), abs(vmin)) * 0.3
                xmin, xmax = vmin - m, vmax + m
                if vmin >= 0:
                    xmin = max(0, vmin - m)
                return [xmin, xmax]

            single_range = compute_range_single(left_values)
            left_colors = ['#EF476F' if c == 'Web TÜFE' else '#118AB2' for c in left_categories]
            single_fig = go.Figure()
            single_fig.add_trace(go.Bar(
                y=left_categories,
                x=left_values,
                orientation='h',
                marker=dict(color=left_colors, line=dict(width=0)),
                text=[f'<b>{v:+.2f}%</b>' for v in left_values],
                textposition='outside',
                textfont=dict(size=15, family='Inter, sans-serif', color='#2B2D42'),
                cliponaxis=False,
                hovertemplate='%{y}: %{x:+.2f}%<extra></extra>'
            ))
            chart_title = f'Web TÜFE {month_name} Ayı Ana Grup Artış Oranları'
            single_fig.update_layout(
                title=dict(text=''),
                xaxis=dict(title='Değişim (%)', gridcolor='#E9ECEF', zerolinecolor='#E9ECEF', range=single_range,
                           title_font=dict(size=14, family='Inter, sans-serif', color='#2B2D42'),
                           tickfont=dict(size=12, family='Inter, sans-serif', color='#2B2D42')),
                yaxis=dict(title='Grup', autorange='reversed', gridcolor='#E9ECEF',
                           title_font=dict(size=14, family='Inter, sans-serif', color='#2B2D42'),
                           tickfont=dict(size=14, family='Arial Black, sans-serif', color='#2B2D42')),
                showlegend=False, plot_bgcolor='white', paper_bgcolor='white', height=640,
                margin=dict(l=40, r=80, t=30, b=50), hovermode='y unified',
                hoverlabel=dict(bgcolor='white', font_size=12, font_family='Inter, sans-serif', font_color='#2B2D42')
            )
            graphJSON = json.dumps(single_fig, cls=plotly.utils.PlotlyJSONEncoder)
            return render_template('index.html', 
                                 graphJSON=graphJSON, 
                                 active_page='ana_sayfa', 
                                 last_update=last_update,
                                 available_dates=available_dates,
                                 selected_date=selected_date,
                                 sorted_group_data=data_pairs_sorted,
                                 show_contrib=show_contrib,
                                 chart_title=chart_title)

        # Build contribution series from katkıpayları.csv (right side)
        contribGraphJSON = None
        contrib_df = None
        try:
            contrib_df = pd.read_csv('katkıpayları.csv', index_col=0)
        except Exception:
            contrib_df = None

        right_values = None
        if contrib_df is not None:
            if selected_date and selected_date in contrib_df.index:
                row = contrib_df.loc[selected_date]
            else:
                row = contrib_df.iloc[-1]
            if 'Web TÜFE' in row.index:
                row = row.drop('Web TÜFE')
            # Reindex to match left order where possible
            try:
                row = row.reindex(left_categories)
            except Exception:
                pass
            right_values = row.tolist()

        # Create combined subplot: 1 row, 2 columns (left bars, right bars) with shared y-axis
        # Leave a visible middle gutter (for centered labels) by increasing spacing
        # Create two symmetric panels with a wide center gutter for labels
        fig = make_subplots(rows=1, cols=2, column_widths=[0.37, 0.37],
                             specs=[[{"type": "bar"}, {"type": "bar"}]],
                             horizontal_spacing=0.26, shared_yaxes=True)

        # Left bars (monthly change)
        left_colors = ['#EF476F' if c == 'Web TÜFE' else '#118AB2' for c in left_categories]
        fig.add_trace(go.Bar(
            y=left_categories,
            x=left_values,
            orientation='h',
            marker=dict(color=left_colors, line=dict(width=0)),
            name='Aylık değişim',
            text=[f'<b>{v:+.2f}%</b>' for v in left_values],
            textposition='outside',
            textfont=dict(size=15, family='Inter, sans-serif', color='#2B2D42'),
            cliponaxis=False,
            hovertemplate='%{y}: %{x:+.2f}%<extra></extra>'
        ), row=1, col=1)

        # Right bars (contributions), if available
        if right_values is not None:
            fig.add_trace(go.Bar(
                y=left_categories,
                x=right_values,
                orientation='h',
                marker=dict(color='#118AB2', line=dict(width=0)),
                name='Aylık etkiler',
                text=[f'<b>{(v if pd.notna(v) else 0):+.2f}</b>' for v in right_values],
                textposition='outside',
                textfont=dict(size=15, family='Inter, sans-serif', color='#2B2D42'),
                cliponaxis=False,
                hovertemplate='%{y}: %{x:+.2f} puan<extra></extra>'
            ), row=1, col=2)

        # Determine ranges for both sides
        def compute_range(values):
            vals = [v for v in values if v is not None]
            if not vals:
                return [0, 1]
            vmin, vmax = min(vals), max(vals)
            vr = vmax - vmin
            m = vr * 0.3 if vr != 0 else max(abs(vmax), abs(vmin)) * 0.3
            xmin, xmax = vmin - m, vmax + m
            if vmin >= 0:
                xmin = max(0, vmin - m)
            return [xmin, xmax]

        # Data-driven ranges with margin; widen left bound a bit if negative labels would clip
        left_range = compute_range(left_values)
        right_range = compute_range(right_values) if right_values is not None else [0, 1]

        # Update shared y-axis and both x-axes
        # Put y tick labels in the center gutter by placing them on right side of the left subplot
        fig.update_yaxes(title_text='Grup', autorange='reversed', side='right',
                         title_font=dict(size=14, family='Inter, sans-serif', color='#2B2D42'),
                         tickfont=dict(size=14, family='Arial Black, sans-serif', color='#2B2D42'),
                         showticklabels=True, row=1, col=1)
        # Hide y tick labels on the right subplot so labels appear only in the middle
        fig.update_yaxes(showticklabels=False, row=1, col=2)
        fig.update_xaxes(title_text='Değişim (%)', range=left_range, gridcolor='#E9ECEF', zerolinecolor='#E9ECEF',
                         tickfont=dict(size=12, family='Inter, sans-serif', color='#2B2D42'), row=1, col=1)
        fig.update_xaxes(title_text='Katkı (puan)', range=right_range, gridcolor='#E9ECEF', zerolinecolor='#E9ECEF',
                         tickfont=dict(size=12, family='Inter, sans-serif', color='#2B2D42'), row=1, col=2)

        # Layout - leave title empty to place heading in template row with controls
        chart_title = f'Web TÜFE {month_name} Ayı Ana Grup Aylık Değişimleri ve Katkıları'
        fig.update_layout(
            title=dict(text=''),
            barmode='overlay', bargap=0.25,
            showlegend=False,
            legend=dict(orientation='h', yanchor='bottom', y=0.02, xanchor='center', x=0.5),
            plot_bgcolor='white', paper_bgcolor='white', height=640,
            margin=dict(l=40, r=40, t=30, b=50), hovermode='y unified',
            hoverlabel=dict(bgcolor='white', font_size=12, font_family='Inter, sans-serif', font_color='#2B2D42')
        )

        # Export combined
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return render_template('index.html', 
                             graphJSON=graphJSON, 
                             active_page='ana_sayfa', 
                             last_update=last_update,
                             available_dates=available_dates,
                             selected_date=selected_date,
                             sorted_group_data=data_pairs_sorted,
                             show_contrib=show_contrib,
                             chart_title=chart_title)
    except Exception as e:
        flash(f'Bir hata oluştu: {str(e)}', 'error')
        available_dates = get_available_dates()
        return render_template('index.html', 
                             available_dates=available_dates,
                             selected_date=None,
                             sorted_group_data=[],
                             show_contrib=True)

@app.route('/tufe', methods=['GET', 'POST'])
def tufe():
    # Initialize graphJSON
    graphJSON = None
    # Get TÜFE data
    df = get_tufe_data()
    # Get last date and last value
    last_date = df['Tarih'].iloc[-1]
    last_value = df['Web TÜFE'].iloc[-1]
    # Calculate change rate
    change_rate = last_value - 100
    # Get monthly change and last column date
    monthly_change, last_col_date = get_monthly_change()
    # Get Turkish month name
    month_name = get_turkish_month(last_col_date)
    
    # Get madde names from Google Sheet
    """creds = get_google_credentials_2()
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key('14iiu_MQwtMxHTFt6ceyFhkk6v0OL-wuoQS1IGPzSpNE')
    worksheet = spreadsheet.get_worksheet_by_id(459488282)
    data = worksheet.get_all_values()
    df_madde = pd.DataFrame(data[1:], columns=data[0])"""
    import pandas as pd
    df_madde=pd.read_csv("endeksler_int.csv").rename(columns={"Unnamed: 0":"Tarih"})
    madde_names = df_madde.columns[1:].tolist()  # Get column names as madde names
    
    selected_madde = request.form.get('madde') if request.method == 'POST' else 'TÜFE'
    
    if selected_madde == 'TÜFE':
        # Filter dates to show only first day of each month
        df['month'] = df['Tarih'].dt.to_period('M')
        first_days = df.groupby('month').first()
        
        # Read TÜİK data from tuikytd.csv
        tuik_df = None
        try:
            tuik_df = pd.read_csv('tuikytd.csv', index_col=0)
            tuik_df.index = pd.to_datetime(tuik_df.index)
            tuik_df = tuik_df.sort_index()
        except Exception as e:
            print(f"TÜİK verisi okunamadı: {e}")
        
        # Create line plot
        fig = go.Figure()
        
        # Add Web TÜFE line
        fig.add_trace(go.Scatter(
            x=df['Tarih'],
            y=df['Web TÜFE'],
            mode='lines',
            name='Web TÜFE',
            line=dict(
                color='#EF476F',
                width=3
            ),
            hovertemplate='%{customdata[0]}<br>Web TÜFE: %{customdata[1]:+.2f}%' + '<extra></extra>',
            customdata=[[f"{date.strftime('%d')} {get_turkish_month(date.strftime('%Y-%m-%d'))} {date.strftime('%Y')}", y-100] for date, y in zip(df['Tarih'], df['Web TÜFE'])]
        ))
        
        # Add TÜİK TÜFE line if data is available
        if tuik_df is not None and 'Genel' in tuik_df.columns:
            # Filter TÜİK data to match Web TÜFE date range
            tuik_filtered = tuik_df[tuik_df.index >= df['Tarih'].min()]
            tuik_filtered = tuik_filtered[tuik_filtered.index <= df['Tarih'].max()]
            
            if not tuik_filtered.empty:
                fig.add_trace(go.Scatter(
                    x=tuik_filtered.index,
                    y=tuik_filtered['Genel'],
                    mode='lines',
                    name='TÜİK TÜFE',
                    line=dict(
                        color='#118AB2',
                        width=3,
                        shape='hv'  # Step grafik
                    ),
                    hovertemplate='%{customdata[0]}<br>TÜİK TÜFE: %{customdata[1]:+.2f}%' + '<extra></extra>',
                    customdata=[[f"{date.strftime('%d')} {get_turkish_month(date.strftime('%Y-%m-%d'))} {date.strftime('%Y')}", y-100] for date, y in zip(tuik_filtered.index, tuik_filtered['Genel'])]
                ))
        
        # Update layout with modern theme
        fig.update_layout(
            # Removed invalid width property
            title=dict(
                text='Web Tüketici Fiyat Endeksi',
                font=dict(
                    size=24,
                    family='Inter, sans-serif',
                    color='#2B2D42'
                ),
                y=0.95
            ),
            xaxis=dict(
                title='Tarih',
                title_font=dict(
                    size=14,
                    family='Inter, sans-serif',
                    color='#2B2D42'
                ),
                tickfont=dict(
                    size=14,
                    family='Inter, sans-serif',
                    color='#2B2D42'
                ),
                gridcolor='#E9ECEF',
                zerolinecolor='#E9ECEF',
                tickformat='%d %B %Y',
                tickangle=0,
                ticktext=[f"{get_turkish_month(date.strftime('%Y-%m-%d'))} {date.strftime('%Y')}" for date in first_days['Tarih'][1:]],
                tickvals=first_days['Tarih'][1:]
            ),
            yaxis=dict(
                title='TÜFE (%)',
                title_font=dict(
                    size=14,
                    family='Inter, sans-serif',
                    color='#2B2D42'
                ),
                tickfont=dict(
                    size=14,
                    family='Inter, sans-serif',
                    color='#2B2D42'
                ),
                gridcolor='#E9ECEF'
            ),
            showlegend=True,
            legend=dict(
                font=dict(size=12, family='Inter, sans-serif', color='#2B2D42'),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#E9ECEF',
                borderwidth=1,
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=600,
            margin=dict(l=20, r=20, t=80, b=20),
            hovermode='closest',
            hoverlabel=dict(
                bgcolor='white',
                font_size=12,
                font_family='Inter, sans-serif',
                namelength=-1
            )
        )
        
        # Convert plot to JSON
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Prepare bar chart data
        bar_months, bar_tufe, bar_tuik = get_tufe_vs_tuik_bar_data()
        # Bar chart
        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(
            x=bar_months,
            y=bar_tufe,
            name='Web TÜFE',
            marker_color='#EF476F',
            text = [f'<b>{v:.2f}</b>' if v is not None else '' for v in bar_tufe],
            textposition='outside',
            textfont=dict(size=14, color='#EF476F', family='Inter, sans-serif'),
            width=0.35,
            hovertemplate='Web TÜFE: %{y:.2f}<extra></extra>',
            cliponaxis=False
        ))
        bar_fig.add_trace(go.Bar(
            x=bar_months,
            y=bar_tuik,
            name='TÜİK TÜFE',
            marker_color='#118AB2',
            text = [f'<b>{v:.2f}</b>' if v is not None else '' for v in bar_tuik],
            textposition='outside',
            textfont=dict(size=14, color='#118AB2', family='Inter, sans-serif'),
            width=0.35,
            hovertemplate='TÜİK TÜFE: %{y:.2f}<extra></extra>',
            cliponaxis=False
        ))
        bar_fig.update_layout(
            barmode='group',
            title=dict(
                text='Aylık Web TÜFE ve TÜİK TÜFE Karşılaştırması',
                font=dict(size=20, family='Inter, sans-serif', color='#2B2D42'),
                y=0.95
            ),
            xaxis=dict(
                title='Ay',
                title_font=dict(size=14, family='Inter, sans-serif', color='#2B2D42'),
                tickfont=dict(size=12, family='Inter, sans-serif', color='#2B2D42'),
                gridcolor='#E9ECEF'
            ),
            yaxis=dict(
                title='Değişim (%)',
                title_font=dict(size=14, family='Inter, sans-serif', color='#2B2D42'),
                tickfont=dict(size=12, family='Inter, sans-serif', color='#2B2D42'),
                gridcolor='#E9ECEF'
            ),
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=max(min(len(bar_months) * 40, 800), 400),
            margin=dict(l=10, r=10, t=40, b=20),
            hovermode='x',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        bar_graphJSON = json.dumps(bar_fig, cls=plotly.utils.PlotlyJSONEncoder)
        # Line chart
        line_fig = go.Figure()
        line_fig.add_trace(go.Scatter(
            x=bar_months,
            y=bar_tufe,
            name='Web TÜFE',
            mode='lines+markers',
            line=dict(color='#EF476F', width=3),
            marker=dict(size=8, color='#EF476F'),
            text=[f'   {v:.2f}' if v is not None and v < 0 else (f'{v:.2f}' if v is not None else '') for v in bar_tufe],
            textposition='top center',
            textfont=dict(size=12, color='#EF476F', family='Inter, sans-serif'),
            hovertemplate='Web TÜFE: %{y:.2f}<extra></extra>'
        ))
        line_fig.add_trace(go.Scatter(
            x=bar_months,
            y=bar_tuik,
            name='TÜİK TÜFE',
            mode='lines+markers',
            line=dict(color='#118AB2', width=3),
            marker=dict(size=8, color='#118AB2'),
            text=[f'   {v:.2f}' if v is not None and v < 0 else (f'{v:.2f}' if v is not None else '') for v in bar_tuik],
            textposition='top center',
            textfont=dict(size=12, color='#118AB2', family='Inter, sans-serif'),
            hovertemplate='TÜİK TÜFE: %{y:.2f}<extra></extra>'
        ))
        line_fig.update_layout(
            title=dict(
                text='Aylık Web TÜFE ve TÜİK TÜFE Karşılaştırması',
                font=dict(size=20, family='Inter, sans-serif', color='#2B2D42'),
                y=0.95
            ),
            xaxis=dict(
                title='Ay',
                title_font=dict(size=14, family='Inter, sans-serif', color='#2B2D42'),
                tickfont=dict(size=12, family='Inter, sans-serif', color='#2B2D42'),
                gridcolor='#E9ECEF'
            ),
            yaxis=dict(
                title='Değişim (%)',
                title_font=dict(size=14, family='Inter, sans-serif', color='#2B2D42'),
                tickfont=dict(size=12, family='Inter, sans-serif', color='#2B2D42'),
                gridcolor='#E9ECEF'
            ),
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=max(min(len(bar_months) * 40, 800), 400),
            margin=dict(l=10, r=10, t=40, b=20),
            hovermode='x',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        line_graphJSON = json.dumps(line_fig, cls=plotly.utils.PlotlyJSONEncoder)
        # Ensure last_date is a datetime object
        if isinstance(last_date, str):
            try:
                last_date = datetime.strptime(last_date, '%Y-%m-%d')
            except:
                try:
                    last_date = datetime.strptime(last_date, '%d.%m.%Y')
                except:
                    pass
        print('Dropdown selected:', selected_madde)
        print('Normalized:', selected_madde.strip().lower())
        print('Endeks columns:', list(df.columns))
        print('Degisim index:', list(df.index))
        print('Endeks eşleşme:', selected_madde.strip().lower() in df.columns)
        print('Degisim eşleşme:', selected_madde.strip().lower() in df.index)
        # Bar grafiği başlığında turkish_month kullanılmadan hemen önce tanımla
        try:
            selected_date_obj = datetime.strptime(last_col_date, '%Y-%m')
            turkish_month = get_turkish_month(selected_date_obj.strftime('%Y-%m-%d'))
        except Exception:
            turkish_month = last_col_date
        return render_template('tufe.html', graphJSON=graphJSON,
            last_date=last_date,
            change_rate=change_rate,
            month_name=month_name,
            monthly_change=monthly_change,
            bar_graphJSON=bar_graphJSON,
            line_graphJSON=line_graphJSON,
            active_page='tufe',
            madde_names=madde_names,
            selected_madde=selected_madde
        )
    else:
        bar_labels=[]
        bar_values=[]
        bar_colors=[]
        # Madde seçiliyse endeks grafiği için 459488282 ID'li tabloyu oku
        """worksheet_endeks = spreadsheet.get_worksheet_by_id(459488282)
        data_endeks = worksheet_endeks.get_all_values()
        df_endeks = pd.DataFrame(data_endeks[1:], columns=data_endeks[0])"""
        df_endeks=pd.read_csv("endeksler_int.csv").rename(columns={"Unnamed: 0":"Tarih"})
        df_endeks = df_endeks.set_index('Tarih')
        df_endeks.columns = df_endeks.columns.str.strip().str.lower()
        df_endeks.index = pd.to_datetime(df_endeks.index)
        norm2orig = {col.strip().lower(): col for col in df_endeks.columns}
        selected_madde_norm = selected_madde.strip().lower()
        print('Dropdown selected:', selected_madde)
        print('Normalized:', selected_madde_norm)
        print('Endeks columns:', list(df_endeks.columns))
        if selected_madde_norm in norm2orig:
            print("Endeks bulundu")
            real_col = norm2orig[selected_madde_norm]
            endeks_seri = df_endeks[real_col].apply(lambda v: float(str(v).replace(',', '.')) if v not in [None, ''] else None)
            endeks_dates = df_endeks.index
            
            # Toplam değişim oranını hesapla
            total_change = endeks_seri.iloc[-1] - 100 if not endeks_seri.empty else None
            
            # Aylık değişim oranını al
            """monthly_change_worksheet = spreadsheet.get_worksheet_by_id(1103913248)
            monthly_change_data = monthly_change_worksheet.get_all_values()
            df_monthly = pd.DataFrame(monthly_change_data[1:], columns=monthly_change_data[0])"""
            df_monthly=pd.read_csv("maddeleraylık.csv",index_col=0)
            df_monthly[df_monthly.columns[0]] = df_monthly[df_monthly.columns[0]].str.strip().str.lower()
            monthly_row = df_monthly[df_monthly.iloc[:,0] == selected_madde_norm]
            monthly_change = None
            if not monthly_row.empty:
                try:
                    monthly_change = float(str(monthly_row.iloc[:,-1].values[0]).replace(',', '.'))
                except:
                    monthly_change = None
            
            print("Endeks serisi uzunluğu:", len(endeks_seri))
            print("Endeks tarihleri uzunluğu:", len(endeks_dates))
            print("İlk birkaç endeks değeri:", endeks_seri.head().tolist())
            
            turkish_dates = [f"{d.day} {get_turkish_month(d.strftime('%Y-%m-%d'))} {d.year}" for d in endeks_dates]
            aybasi_tarihler = [d for d in endeks_dates if d.day == 1]
            tickvals = aybasi_tarihler
            ticktext = [f"{get_turkish_month(d.strftime('%Y-%m-%d'))} {d.year}" for d in aybasi_tarihler]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=endeks_dates,
                y=endeks_seri,
                mode='lines',
                name=selected_madde,
                customdata=turkish_dates,
                line=dict(color='#EF476F', width=3),
                hovertemplate='%{customdata}<br>' + f'{selected_madde}: ' + '%{y:.2f}<extra></extra>'
            ))
            
            # TÜFE grafiği ile aynı stil
            fig.update_layout(
                title=dict(
                    text=f'{selected_madde} Endeksi',
                    font=dict(size=24, family='Inter, sans-serif', color='#2B2D42'),
                    y=0.95
                ),
                xaxis=dict(
                    title='Tarih',
                    title_font=dict(
                        size=14,
                        family='Inter, sans-serif',
                        color='#2B2D42'
                    ),
                    tickfont=dict(
                        size=14,
                        family='Inter, sans-serif',
                        color='#2B2D42'
                    ),
                    gridcolor='#E9ECEF',
                    zerolinecolor='#E9ECEF',
                    tickvals=tickvals,
                    ticktext=ticktext,
                    tickangle=0
                ),
                yaxis=dict(
                    title='Endeks',
                    title_font=dict(
                        size=14,
                        family='Inter, sans-serif',
                        color='#2B2D42'
                    ),
                    tickfont=dict(
                        size=14,
                        family='Inter, sans-serif',
                        color='#2B2D42'
                    ),
                    gridcolor='#E9ECEF'
                ),
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=600,
                margin=dict(l=20, r=20, t=80, b=20),
                hovermode='closest',
                hoverlabel=dict(
                    bgcolor='white',
                    font_size=12,
                    font_family='Inter, sans-serif',
                    namelength=-1
                )
            )
            
            print("Grafik oluşturuldu")
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            print("GraphJSON oluşturuldu, uzunluk:", len(graphJSON))
            
            # Aylık değişim grafiği için verileri hazırla
            monthly_changes = []
            monthly_dates = []
            if not monthly_row.empty:
                for col in df_monthly.columns[1:]:  # İlk sütun madde adı
                    try:
                        value = float(str(monthly_row[col].values[0]).replace(',', '.'))
                        monthly_changes.append(value)
                        # Tarihi YYYY-MM formatına çevir
                        date_obj = datetime.strptime(col, '%Y-%m-%d')
                        monthly_dates.append(f"{get_turkish_month(date_obj.strftime('%Y-%m-%d'))} {date_obj.year}")
                    except:
                        monthly_changes.append(None)
                        monthly_dates.append(col)
            
            # TÜİK verilerini al
            tuik_changes = []
            try:
                data_tuik = pd.read_excel("harcama gruplarina gore endeks sonuclari.xlsx")
                cols = data_tuik.iloc[2]
                data_tuik.columns = cols
                data_tuik = data_tuik.iloc[4:, 4:16]
                # Tarih indeksini ay başına göre ayarla
                data_tuik = data_tuik.set_index(pd.date_range(start="2005-01-01", freq="MS", periods=len(data_tuik)))
                data_tuik = data_tuik.pct_change().dropna().loc["2025-02":].round(4) * 100
                data_tuik.columns = [
                    "Gıda ve alkolsüz içecekler", "Alkollü içecekler ve tütün", "Giyim ve ayakkabı", "Konut",
                    "Ev eşyası", "Sağlık", "Ulaştırma", "Haberleşme", "Eğlence ve kültür", "Eğitim",
                    "Lokanta ve oteller", "Çeşitli mal ve hizmetler"
                ]
                # TÜİK verilerini aylık değişim tarihleriyle eşleştir
                for date in monthly_dates:
                    try:
                        month, year = date.split()
                        month_map = {
                            'Ocak': '01', 'Şubat': '02', 'Mart': '03', 'Nisan': '04',
                            'Mayıs': '05', 'Haziran': '06', 'Temmuz': '07', 'Ağustos': '08',
                            'Eylül': '09', 'Ekim': '10', 'Kasım': '11', 'Aralık': '12'
                        }
                        date_str = f"{year}-{month_map[month]}-01"
                        date_obj = pd.to_datetime(date_str)
                        if selected_madde in data_tuik.columns and date_obj in data_tuik.index:
                            tuik_value = data_tuik.loc[date_obj, selected_madde]
                            tuik_changes.append(tuik_value)
                        else:
                            tuik_changes.append(None)
                    except Exception as e:
                        print(f"TÜİK verisi eşleştirme hatası: {e}")
                        tuik_changes.append(None)
            except Exception as e:
                print("TÜİK verisi okunamadı:", e)
                tuik_changes = [None] * len(monthly_dates)
            
            # Sort bars by value descending (highest first)
            sorted_data = sorted(zip(bar_labels, bar_values, bar_colors), key=lambda x: x[1], reverse=False)
            bar_labels = [x[0] for x in sorted_data]
            bar_values = [x[1] for x in sorted_data]
            bar_colors = [x[2] for x in sorted_data]
            
            # Aylık değişim bar grafiği
            bar_months = monthly_dates  # bar_months değişkenini burada tanımla
            bar_labels = bar_months
            bar_fig = go.Figure()
            bar_texts = []
            text_colors = []
            text_positions = []
            threshold = max(abs(v) for v in monthly_changes) * 0.8 if monthly_changes else 1
            for v in monthly_changes:
                bar_texts.append(f'<b>{v:.2f}</b>')

                
                
                text_colors.append('#2B2D42')
                text_positions.append('outside')
            y_min = min(monthly_changes)
            y_max = max(monthly_changes)

            # Marj hesapla (üstteki text'ler için makul marj)
            y_range = y_max - y_min
            y_margin_bottom = y_range * 0.1 if y_range != 0 else abs(y_max) * 0.1
            y_margin_top = y_range * 0.2 if y_range != 0 else abs(y_max) * 0.2  # 20% üst marj

            # Marjlı sınırlar
            y_min_with_margin = y_min - y_margin_bottom
            y_max_with_margin = y_max + y_margin_top

            # Sıfıra yaklaşma kontrolü
            if y_min >= 0:
                y_min_with_margin = max(0, y_min - y_margin_bottom)
            if y_max <= 0:
                y_max_with_margin = min(0, y_max + y_margin_top)
            fig = go.Figure(go.Bar(
                x=monthly_dates,
                y=monthly_changes,
                name=selected_madde,
                marker_color='#EF476F',
                text=bar_texts,
                textposition=text_positions,
                textfont=dict(
                    size=16,
                    color=text_colors,
                    family='Inter, sans-serif'
                ),
                width=0.6,
                hovertemplate='%{x}<br>Değişim: %{y:.2f}%<extra></extra>',
                cliponaxis=False
            ))
            
            # y ekseni aralığını hesapla (negatif ve pozitif değerler için)
            valid_changes = [v for v in monthly_changes if v is not None]
            y_min = min(valid_changes + [0])
            y_max = max(valid_changes + [0])
            yaxis_range = [y_min * 1.2 if y_min < 0 else 0, y_max * 1.2 if y_max > 0 else 1]

            
            
            fig.update_layout(
                title=dict(
                    text=f'{selected_madde} Aylık Değişimler',
                    font=dict(
                        size=20,
                        family='Inter, sans-serif',
                        color='#2B2D42'
                    ),
                    y=0.98,
                    x=0.5,
                    xanchor='center',
                    yanchor='top'
                ),
                xaxis=dict(
                    title='Ay',
                    title_font=dict(
                        size=14,
                        family='Inter, sans-serif',
                        color='#2B2D42'
                    ),
                    tickfont=dict(
                        size=12,
                        family='Inter, sans-serif',
                        color='#2B2D42'
                    ),
                    tickangle=0
                ),
                yaxis=dict(
                    title='Değişim (%)',
                    title_font=dict(
                        size=14,
                        family='Inter, sans-serif',
                        color='#2B2D42'
                    ),
                    tickfont=dict(
                        size=12,
                        family='Inter, sans-serif',
                        color='#2B2D42'
                    ),
                    gridcolor='#E9ECEF',
                    range=[y_min_with_margin, y_max_with_margin]  # Y ekseni aralığını ayarla
                ),
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=450,  # Yüksekliği artırdık
                margin=dict(l=10, r=40, t=60, b=20),  # Sağ marjı artırdık (10 -> 40)
                hovermode='x'
            )
            
            bar_graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            # Aylık değişim line grafiği
            line_fig = go.Figure()
            line_fig.add_trace(go.Scatter(
                x=monthly_dates,
                y=monthly_changes,
                mode='lines+markers+text',
                name=selected_madde,
                line=dict(color='#EF476F', width=3),
                marker=dict(size=8, color='#EF476F'),
                text=[f'{v:.2f}' if v is not None else '' for v in monthly_changes],
                textposition='top center',
                textfont=dict(size=12, color='#EF476F', family='Inter, sans-serif'),
                hovertemplate='%{x}<br>Değişim: %{y:.2f}%<extra></extra>',
                cliponaxis=False
            ))
            
            line_fig.update_layout(
                title=dict(
                    text=f'{selected_madde} Aylık Değişim Oranları',
                    font=dict(
                        size=20,
                        family='Inter, sans-serif',
                        color='#2B2D42'
                    ),
                    y=0.95
                ),
                xaxis=dict(
                    title='Ay',
                    title_font=dict(
                        size=14,
                        family='Inter, sans-serif',
                        color='#2B2D42'
                    ),
                    tickfont=dict(
                        size=12,
                        family='Inter, sans-serif',
                        color='#2B2D42'
                    ),
                    tickangle=0
                ),
                yaxis=dict(
                    title='Değişim (%)',
                    title_font=dict(
                        size=14,
                        family='Inter, sans-serif',
                        color='#2B2D42'
                    ),
                    tickfont=dict(
                        size=12,
                        family='Inter, sans-serif',
                        color='#2B2D42'
                    ),
                    gridcolor='#E9ECEF',
                    range=[y_min_with_margin, y_max_with_margin]
                ),
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=450,  # Yüksekliği artırdık
                margin=dict(l=10, r=40, t=60, b=20),  # Sağ marjı artırdık (10 -> 40)
                hovermode='x'
            )
            line_graphJSON = line_fig.to_json()
            
            # Get the last month from the monthly CSV file for individual items
            last_month_from_csv = None
            if not df_monthly.empty:
                # Get the last column (excluding the first column which is the group name)
                last_column = df_monthly.columns[-1]
                try:
                    # Parse the date from the column name (format: YYYY-MM-DD)
                    date_obj = datetime.strptime(last_column, '%Y-%m-%d')
                    last_month_from_csv = get_turkish_month(date_obj.strftime('%Y-%m-%d'))
                except:
                    last_month_from_csv = None
            
            return render_template('tufe.html',
                graphJSON=graphJSON,
                last_date=endeks_dates[-1] if not endeks_dates.empty else None,
                change_rate=total_change,
                month_name=last_month_from_csv if last_month_from_csv else (get_turkish_month(endeks_dates[-1].strftime('%Y-%m-%d')) if not endeks_dates.empty else None),
                monthly_change=monthly_change,
                bar_graphJSON=bar_graphJSON,
                line_graphJSON=line_graphJSON,
                active_page='tufe',
                madde_names=madde_names,
                selected_madde=selected_madde,
                no_data=False
            )
        else:
            return render_template('tufe.html',
                graphJSON=None,
                last_date=None,
                change_rate=None,
                month_name=None,
                monthly_change=None,
                active_page='tufe',
                madde_names=madde_names,
                selected_madde=selected_madde,
                aylik_degisim_graphJSON=None,
                no_data=True
            )

@app.route('/ana-gruplar', methods=['GET', 'POST'])
def ana_gruplar():
    line_graphJSON = 'null'
    bar_graphJSON = 'null'
    bar_labels = []
    bar_values = []
    bar_colors = []
    turkish_month = ''
    df = get_ana_gruplar_data()
    grup_adlari = [col for col in df.columns if col not in ['Tarih', 'Web TÜFE']]
    selected_group = request.form.get('group') if request.method == 'POST' else grup_adlari[0]

    # Google Sheets bağlantısı
    """creds = get_google_credentials_2()
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key('14iiu_MQwtMxHTFt6ceyFhkk6v0OL-wuoQS1IGPzSpNE')"""

    # Line plot için veriler
    tarih = df['Tarih']
    values = df[selected_group]
    total_change = values.iloc[-1] - 100
    monthly_change, last_col_date = get_ana_grup_monthly_change(selected_group)
    month_name = get_turkish_month(last_col_date)
    aybasi_tarihler = df['Tarih'][df['Tarih'].dt.is_month_start]
    tickvals = aybasi_tarihler
    ticktext = [f"{get_turkish_month(d.strftime('%Y-%m-%d'))} {d.year}" for d in aybasi_tarihler]
    
    # Read TÜİK data from tuikytd.csv
    tuik_df = None
    tuik_column_name = None
    try:
        tuik_df = pd.read_csv('tuikytd.csv', index_col=0)
        tuik_df.index = pd.to_datetime(tuik_df.index)
        tuik_df = tuik_df.sort_index()
        
        # Map Web TÜFE group names to TÜİK column names
        group_mapping = {
            'Gıda ve alkolsüz içecekler': 'Gıda ve alkolsüz içecekler',
            'Alkollü içecekler ve tütün': 'Alkollü içecekler ve tütün',
            'Giyim ve ayakkabı': 'Giyim ve ayakkabı',
            'Konut,Su,Elektrik,Gaz ve Diğer Yakıtlar': 'Konut,Su,Elektrik,Gaz ve Diğer Yakıtlar',
            'Ev eşyası': 'Ev eşyası',
            'Sağlık': 'Sağlık',
            'Ulaştırma': 'Ulaştırma',
            'Haberleşme': 'Haberleşme',
            'Eğlence ve kültür': 'Eğlence ve kültür',
            'Eğitim': 'Eğitim',
            'Lokanta ve oteller': 'Lokanta ve oteller',
            'Çeşitli mal ve hizmetler': 'Çeşitli mal ve hizmetler'
        }
        
        tuik_column_name = group_mapping.get(selected_group)
        
    except Exception as e:
        print(f"TÜİK verisi okunamadı: {e}")
    
    fig = go.Figure()
    customdata = [f"{d.day} {get_turkish_month(d.strftime('%Y-%m-%d'))} {d.year}" for d in tarih]

    # Add Web TÜFE line
    fig.add_trace(go.Scatter(
        x=tarih,
        y=values,
        mode='lines',
        name=f'Web TÜFE - {selected_group}',
        line=dict(color='#EF476F', width=3),
        customdata=customdata,
        hovertemplate='<b>%{customdata}</b><br>' + f'Web TÜFE - {selected_group}: ' + '%{y:.2f}<extra></extra>'
    ))
    
    # Add TÜİK line if data is available
    if tuik_df is not None and tuik_column_name and tuik_column_name in tuik_df.columns:
        # Filter TÜİK data to match Web TÜFE date range
        tuik_filtered = tuik_df[tuik_df.index >= tarih.min()]
        tuik_filtered = tuik_filtered[tuik_filtered.index <= tarih.max()]
        
        if not tuik_filtered.empty:
            tuik_customdata = [f"{d.day} {get_turkish_month(d.strftime('%Y-%m-%d'))} {d.year}" for d in tuik_filtered.index]
            fig.add_trace(go.Scatter(
                x=tuik_filtered.index,
                y=tuik_filtered[tuik_column_name],
                mode='lines',
                name=f'TÜİK - {selected_group}',
                line=dict(
                    color='#118AB2',
                    width=3,
                    shape='hv'  # Step grafik
                ),
                customdata=tuik_customdata,
                hovertemplate='<b>%{customdata}</b><br>' + f'TÜİK - {selected_group}: ' + '%{y:.2f}<extra></extra>'
            ))

    fig.update_layout(
    title=dict(
        text=f'{selected_group} Endeksi',
        font=dict(
            size=24,
            family='Inter, sans-serif',
            color='#2B2D42'
        ),
        y=0.95
    ),
    xaxis=dict(
        title='Tarih',
        title_font=dict(
            size=14,
            family='Inter, sans-serif',
            color='#2B2D42'
        ),
        tickfont=dict(
            size=14,
            family='Inter, sans-serif',
            color='#2B2D42'
        ),
        gridcolor='#E9ECEF',
        zerolinecolor='#E9ECEF',
        tickvals=tickvals,
        ticktext=ticktext,
        tickangle=0,
        hoverformat=''
    ),
    yaxis=dict(
        title='Endeks',
        title_font=dict(
            size=14,
            family='Inter, sans-serif',
            color='#2B2D42'
        ),
        tickfont=dict(
            size=14,
            family='Inter, sans-serif',
            color='#2B2D42'
        ),
        gridcolor='#E9ECEF'
    ),
    showlegend=True,
    legend=dict(
        font=dict(size=12, family='Inter, sans-serif', color='#2B2D42'),
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='#E9ECEF',
        borderwidth=1,
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
    height=600,
    margin=dict(l=20, r=20, t=80, b=20),
    hovermode='x unified',
    hoverlabel=dict(
        bgcolor='white',
        font_size=12,
        font_family='Inter, sans-serif',
        namelength=-1
    )
)

    # --- Aylık değişim bar ve line grafik verileri ---
    monthly_changes = []
    monthly_dates = []
    """worksheet = spreadsheet.get_worksheet_by_id(767776936)
    data = worksheet.get_all_values()
    df_monthly = pd.DataFrame(data[1:], columns=data[0])"""
    df_monthly=pd.read_csv("gruplaraylık.csv",index_col=0)
    df_monthly[df_monthly.columns[0]] = df_monthly[df_monthly.columns[0]].str.strip().str.lower()
    selected_group_norm = selected_group.strip().lower()
    monthly_row = df_monthly[df_monthly.iloc[:,0] == selected_group_norm]

    if not monthly_row.empty:
        for col in df_monthly.columns[1:]:
            try:
                value = float(str(monthly_row[col].values[0]).replace(',', '.'))
                monthly_changes.append(value)
                date_obj = datetime.strptime(col, '%Y-%m-%d')
                monthly_dates.append(f"{get_turkish_month(date_obj.strftime('%Y-%m-%d'))} {date_obj.year}")
            except Exception as e:
                monthly_changes.append(None)
                monthly_dates.append(col)
    else:
        # Eğer veri yoksa, en azından boş listeler gönder
        monthly_changes = []
        monthly_dates = []

    # Bar grafik
    bar_fig = go.Figure()
    # Web TÜFE
    bar_fig.add_trace(go.Bar(
        x=monthly_dates,
        y=monthly_changes,
        name='Web TÜFE',
        marker_color='#EF476F',
        text = [f'<b>{v:.2f}</b>' if v is not None else '' for v in monthly_changes],
        textposition='outside',
        textfont=dict(size=14, color='#2B2D42', family='Inter, sans-serif'),
        width=0.35,
        hovertemplate='%{x}<br>Web TÜFE: %{y:.2f}%<extra></extra>'
    ))

    tuik_changes = []
    try:
        # tuikaylik.csv dosyasından TÜİK verilerini oku
        tuik_df = pd.read_csv("tuikaylik.csv", index_col=0)
        tuik_df.index = pd.to_datetime(tuik_df.index).strftime("%Y-%m")
        
        # TÜİK verilerini aylık değişim tarihleriyle eşleştir
        for date in monthly_dates:
            try:
                month, year = date.split()
                month_map = {
                    'Ocak': '01', 'Şubat': '02', 'Mart': '03', 'Nisan': '04',
                    'Mayıs': '05', 'Haziran': '06', 'Temmuz': '07', 'Ağustos': '08',
                    'Eylül': '09', 'Ekim': '10', 'Kasım': '11', 'Aralık': '12'
                }
                date_str = f"{year}-{month_map[month]}"  # YYYY-MM formatı
                
                if selected_group in tuik_df.columns and date_str in tuik_df.index:
                    tuik_value = tuik_df.loc[date_str, selected_group]
                    tuik_changes.append(tuik_value)
                else:
                    tuik_changes.append(None)
            except Exception as e:
                print(f"TÜİK verisi eşleştirme hatası: {e}")
                tuik_changes.append(None)
    except Exception as e:
        print("TÜİK verisi okunamadı:", e)
        tuik_changes = [None] * len(monthly_dates)
    # TÜİK
    bar_fig.add_trace(go.Bar(
        x=monthly_dates,
        y=tuik_changes,
        name='TÜİK',
        marker_color='#118AB2',
        text = [f'<b>{v:.2f}</b>' if v is not None else '' for v in tuik_changes],
        textposition='outside',
        textfont=dict(size=14, color='#118AB2', family='Inter, sans-serif'),
        width=0.35,
        hovertemplate='%{x}<br>TÜİK: %{y:.2f}%<extra></extra>'
    ))
    combined_values = monthly_changes + tuik_changes
    valid_values = [v for v in combined_values if v is not None]

    if valid_values:
        y_min = min(valid_values)
        y_max = max(valid_values)

        # Marj hesapla
        y_range = y_max - y_min
        y_margin = y_range * 0.2 if y_range != 0 else abs(y_max) * 0.2

        # Marjlı sınırlar
        y_min_with_margin = y_min - y_margin
        y_max_with_margin = y_max + y_margin

        # Sıfıra yaklaşma kontrolü
        if y_min >= 0:
            y_min_with_margin = max(0, y_min - y_margin)
        if y_max <= 0:
            y_max_with_margin = min(0, y_max + y_margin)
    bar_fig.update_layout(
        barmode='group',
        title=dict(
            text=f'{selected_group} Aylık Değişim Oranları',
            font=dict(size=20, family='Inter, sans-serif', color='#2B2D42'),
            y=0.95
        ),
        xaxis=dict(
            title='Ay',
            title_font=dict(
                size=14,
                family='Inter, sans-serif',
                color='#2B2D42'
            ),
            tickfont=dict(
                size=12,
                family='Inter, sans-serif',
                color='#2B2D42'
            ),
            gridcolor='#E9ECEF'
        ),
        yaxis=dict(
            title='Değişim (%)',
            title_font=dict(
                size=14,
                family='Inter, sans-serif',
                color='#2B2D42'
            ),
            tickfont=dict(
                size=12,
                family='Inter, sans-serif',
                color='#2B2D42'
            ),
            gridcolor='#E9ECEF',
            range=[y_min_with_margin, y_max_with_margin]
        ),
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400,
        margin=dict(l=20, r=20, t=80, b=20),
        hovermode='x'
    )
    bar_graphJSON = bar_fig.to_json()

    # Line grafik
    line_fig = go.Figure()
    # Web TÜFE
    line_fig.add_trace(go.Scatter(
        x=monthly_dates,
        y=monthly_changes,
        mode='lines+markers',
        name='Web TÜFE',
        line=dict(color='#EF476F', width=3),
        marker=dict(size=8, color='#EF476F'),
        hovertemplate='%{x}<br>Web TÜFE: %{y:.2f}%<extra></extra>'
    ))
    # TÜİK
    line_fig.add_trace(go.Scatter(
        x=monthly_dates,
        y=tuik_changes,
        mode='lines+markers',
        name='TÜİK',
        line=dict(color='#118AB2', width=3),
        marker=dict(size=8, color='#118AB2'),
        hovertemplate='%{x}<br>TÜİK: %{y:.2f}%<extra></extra>'
    ))
    line_fig.update_layout(
        height=400,
        title=dict(
            text=f'{selected_group} Aylık Değişim Oranları',
            font=dict(size=20, family='Inter, sans-serif', color='#2B2D42'),
            y=0.95
        ),
        xaxis=dict(
            title='Ay',
            title_font=dict(size=14, family='Inter, sans-serif', color='#2B2D42'),
            tickfont=dict(size=12, family='Inter, sans-serif', color='#2B2D42'),
            gridcolor='#E9ECEF',
            zerolinecolor='#E9ECEF',
            tickangle=0
        ),
        yaxis=dict(
            title='Değişim (%)',
            title_font=dict(size=14, family='Inter, sans-serif', color='#2B2D42'),
            tickfont=dict(size=12, family='Inter, sans-serif', color='#2B2D42'),
            gridcolor='#E9ECEF'
        ),
        showlegend=True,
        legend=dict(
            font=dict(size=12, family='Inter, sans-serif', color='#2B2D42'),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#E9ECEF',
            borderwidth=1,
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=80, b=20),
        hovermode='x'
    )
    line_graphJSON = line_fig.to_json()

    return render_template('ana_gruplar.html',
        graphJSON=fig.to_json(),
        grup_adlari=grup_adlari,
        selected_group=selected_group,
        total_change=total_change,
        month_name=month_name,
        monthly_change=monthly_change,
        last_value=values.iloc[-1] if not values.empty else None,
        last_date=tarih.iloc[-1].strftime('%d.%m.%Y') if not tarih.empty else '',
        active_page='ana_gruplar',
        bar_graphJSON=bar_graphJSON,
        line_graphJSON=line_graphJSON
    )

@app.route('/harcama-gruplari', methods=['GET', 'POST'])
def harcama_gruplari():
    print("\nHarcama Grupları Route Başladı")
    print("Method:", request.method)
    
    df = get_ana_gruplar_data()
    grup_adlari = [col for col in df.drop("Web TÜFE",axis=1).columns if col != 'Tarih']
    print("Grup adları:", grup_adlari)
    
    selected_group = request.form.get('group') if request.method == 'POST' else grup_adlari[0]
    selected_date = request.form.get('date') if request.method == 'POST' else None
    # Contribution controls (defaults: show=True, type='ana')
    show_contrib = False if request.method != 'POST' else ('show_contrib' in request.form)
    contrib_type = request.form.get('contrib_type', 'ana')
    print("Seçilen grup:", selected_group)
    print("Seçilen tarih:", selected_date)
    
    # Ortak: harcama grupları ve tarih seçenekleri için worksheet ve dataframe
    """creds = get_google_credentials_2()
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key('14iiu_MQwtMxHTFt6ceyFhkk6v0OL-wuoQS1IGPzSpNE')
    harcama_worksheet = spreadsheet.get_worksheet_by_id(1927818004)
    harcama_data = harcama_worksheet.get_all_values()
    df_harcama = pd.DataFrame(harcama_data[1:], columns=harcama_data[0])"""
    df_harcama=pd.read_csv("harcama_gruplarıaylık.csv",index_col=0)
    date_options = df_harcama.columns[1:].tolist()
    # Tarih seçeneklerini YYYY-MM formatına çevir (görüntüleme için)
    formatted_date_options = []
    date_mapping = {}  # YYYY-MM -> YYYY-MM-DD eşleştirmesi
    for date in date_options:
        try:
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            formatted_date = date_obj.strftime('%Y-%m')
            formatted_date_options.append(formatted_date)
            date_mapping[formatted_date] = date  # Orijinal tarihi sakla
        except:
            formatted_date_options.append(date)
            date_mapping[date] = date
    # Tarihleri yeniden eskiye doğru sırala
    formatted_date_options.sort(reverse=True)

    harcama_gruplari = []
    bar_labels = []
    bar_values = []
    bar_colors = []
    bar_graphJSON = None
    contrib_graphJSON = None
    combined_graphJSON = None
    monthly_bar_graphJSON = None
    monthly_line_graphJSON = None
    line_graphJSON = None

    if not selected_date:
        selected_date = formatted_date_options[0]

    # --- Harcama grupları ve grafik verilerini HER ZAMAN hazırla (GET ve POST fark etmeksizin) ---
    try:
        urunler = pd.read_csv('ürünler.csv')
        urunler['Ana Grup'] = urunler['Ana Grup'].str.strip().str.lower()
        urunler['Grup'] = urunler['Grup'].str.strip().str.lower()
        selected_group_norm = selected_group.strip().lower()
        harcama_gruplari = urunler[urunler["Ana Grup"] == selected_group_norm]["Grup"].unique().tolist()
    except Exception as e:
        print("ürünler.csv okuma hatası:", e)
        harcama_gruplari = []

    # Seçili tarihi orijinal formata çevir
    sheet_date = date_mapping[selected_date]
    # Ana grup değişimini 767776936 tablosundan al
    """ana_grup_worksheet = spreadsheet.get_worksheet_by_id(767776936)
    ana_grup_data = ana_grup_worksheet.get_all_values()
    df_ana_grup = pd.DataFrame(ana_grup_data[1:], columns=ana_grup_data[0])"""
    df_ana_grup=pd.read_csv("gruplaraylık.csv",index_col=0)
    df_ana_grup[df_ana_grup.columns[0]] = df_ana_grup[df_ana_grup.columns[0]].str.strip().str.lower()
    ana_grup_row = df_ana_grup[df_ana_grup.iloc[:,0] == selected_group_norm]
    ana_grup_value = None
    if not ana_grup_row.empty:
        try:
            ana_grup_value = float(str(ana_grup_row[sheet_date].values[0]).replace(',', '.'))
        except Exception as e:
            print('Ana grup değeri alınırken hata:', e)
    # Harcama gruplarının değerlerini al
    df_harcama[df_harcama.columns[0]] = df_harcama[df_harcama.columns[0]].str.strip().str.lower()
    bar_labels = []
    bar_values = []
    bar_colors = []
    for grup in harcama_gruplari:
        grup_norm = grup.strip().lower()
        row = df_harcama[df_harcama.iloc[:,0] == grup_norm]
        if not row.empty:
            try:
                value = float(str(row[sheet_date].values[0]).replace(',', '.'))
                bar_labels.append(grup.title())
                bar_values.append(value)
                bar_colors.append('#EF476F' if grup_norm == selected_group_norm else '#118AB2')
            except Exception as e:
                print(f'Grup {grup} için değer alınırken hata:', e)
    # Ana grup da listede yoksa ekle
    if selected_group_norm not in [g.strip().lower() for g in bar_labels] and ana_grup_value is not None:
        bar_labels.append(selected_group.title())
        bar_values.append(ana_grup_value)
        bar_colors.append('#EF476F')
    # Sort bars by value descending (highest first)
    sorted_data = sorted(zip(bar_labels, bar_values, bar_colors), key=lambda x: x[1], reverse=False)
    bar_labels = [x[0] for x in sorted_data]
    bar_values = [x[1] for x in sorted_data]
    bar_colors = [x[2] for x in sorted_data]
    # Barh grafik - harcama_gruplari ile aynı şekilde
    try:
        selected_date_obj = datetime.strptime(selected_date, '%Y-%m')
        turkish_month = get_turkish_month(selected_date_obj.strftime('%Y-%m-%d'))
    except Exception:
        turkish_month = selected_date
    valid_bar_values = [v for v in bar_values if v is not None]
    x_min = min(valid_bar_values + [0])
    x_max = max(valid_bar_values + [0])
    
    # Dynamic xaxis_range calculation based on data distribution
    data_range = x_max - x_min
    if data_range == 0:
        # If all values are the same, create a small range
        xaxis_range = [x_min - 0.1, x_max + 0.1]
    else:
        # Calculate margins based on data magnitude and distribution
        margin_factor = 0.15  # 15% margin
        margin = data_range * margin_factor
        
        # Ensure minimum margin for small ranges
        min_margin = 0.5
        margin = max(margin, min_margin)
        
        # Apply asymmetric margins for better text fitting
        if x_min >= 0:
            # All positive values
            x_min_with_margin = max(0, x_min - margin * 0.5)
            x_max_with_margin = x_max + margin
        elif x_max <= 0:
            # All negative values
            x_min_with_margin = x_min - margin
            x_max_with_margin = min(0, x_max + margin * 0.5)
        else:
            # Mixed positive and negative values
            x_min_with_margin = x_min - margin
            x_max_with_margin = x_max + margin
        
        xaxis_range = [x_min_with_margin, x_max_with_margin]
    
    # Ensure range is not inverted
    if xaxis_range[0] > xaxis_range[1]:
        xaxis_range = [xaxis_range[1], xaxis_range[0]]
    
    fig = go.Figure(go.Bar(
        y=bar_labels,
        x=bar_values,
        orientation='h',
        marker_color=bar_colors,
        cliponaxis=False,
        hovertemplate='%{y}: %{x:.2f}<extra></extra>'
    ))
    
    # Calculate minimal text offset to place text exactly at bar end
    range_span = xaxis_range[1] - xaxis_range[0]
    text_offset = range_span * 0.001  # Reduced to 0.1% for minimal gap
    
    for i, value in enumerate(bar_values):
        # Determine text position based on value sign and magnitude
        if value >= 0:
            # For positive values, place text to the right of the bar
            text_x = value + text_offset
            align_anchor = 'left'
        else:
            # For negative values, place text to the left of the bar
            text_x = value - text_offset
            align_anchor = 'right'
        
        fig.add_annotation(
            x=text_x,
            y=bar_labels[i],
            text=f"<b>{value:.2f}%</b>",
            showarrow=False,
            font=dict(size=15, family="Inter Bold, Inter, sans-serif", color="#2B2D42"),
            align=align_anchor,
            xanchor=align_anchor,
            yanchor='middle'
        )
    fig.update_layout(
        title=dict(
            text=f'{turkish_month} Ayı Harcama Grupları Değişimi',
            font=dict(
                size=20,
                family='Inter, sans-serif',
                color='#2B2D42'
            ),
            y=0.99,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        xaxis=dict(
            title='Değişim (%)',
            title_font=dict(
                size=14,
                family='Inter, sans-serif',
                color='#2B2D42'
            ),
            tickfont=dict(
                size=14,
                family='Inter, sans-serif',
                color='#2B2D42'
            ),
            gridcolor='#E9ECEF',
            zerolinecolor='#E9ECEF',
            range=xaxis_range
        ),
        yaxis=dict(
            title='Harcama Grubu',
            title_font=dict(
                size=14,
                family='Inter, sans-serif',
                color='#2B2D42'
            ),
            tickfont=dict(
                size=14,
                family='Inter, sans-serif',
                color='#2B2D42'
            ),
            gridcolor='#E9ECEF'
        ),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=max(min(len(bar_labels) * 70, 1800), 500),
        margin=dict(l=10, r=10, t=40, b=20),
        hovermode='y unified',
        hoverlabel=dict(bgcolor='white', font_size=12, font_family='Inter, sans-serif', font_color='#2B2D42')
    )
    bar_graphJSON = fig.to_json()

    # Build contribution chart for selected ana grup's harcama grupları
    if show_contrib:
        try:
            # Select correct CSV by contribution type
            file_path = 'harcamagrupları_manşetkatkı.csv' if contrib_type == 'ana' else 'harcamagrupları_katkı.csv'
            df_katki = pd.read_csv(file_path, index_col=0)
            # choose date row
            target_date = sheet_date if sheet_date in df_katki.index else (sheet_date[:7] if sheet_date[:7] in df_katki.index else df_katki.index[-1])
            row = df_katki.loc[target_date]
            # normalize headers
            row.index = [str(c).strip().lower() for c in row.index]
            labels = []
            values = []
            for g in harcama_gruplari:
                key = str(g).strip().lower()
                if key in row.index:
                    try:
                        val = float(str(row[key]).replace(',', '.'))
                        labels.append(g.title())
                        values.append(val)
                    except Exception:
                        pass
            # sort ascending
            pairs = sorted(zip(labels, values), key=lambda x: x[1])
            labels = [p[0] for p in pairs]
            values = [p[1] for p in pairs]
            def xr(vals):
                if not vals:
                    return [0, 1]
                vmin, vmax = min(vals), max(vals)
                rng = vmax - vmin
                m = rng * 0.25 if rng != 0 else max(abs(vmax), abs(vmin)) * 0.25
                xmin, xmax = vmin - m, vmax + m
                if vmin >= 0:
                    xmin = max(0, vmin - m)
                if vmax <= 0:
                    xmax = min(0, vmax + m)
                return [xmin, xmax]
            x_range = xr(values)
            # Build combined symmetric chart like homepage
            left_categories = bar_labels
            left_values = bar_values
            # Align right values to left categories; if a category is missing in contribution data
            # and it is the selected ana grup, fill with ana_grup_value to keep rows aligned
            right_map = {labels[i].strip().lower(): values[i] for i in range(len(labels))}
            right_values = []
            for name in left_categories:
                key = name.strip().lower()
                if key in right_map:
                    val = right_map[key]
                    # Ana grup satırında katkı değeri çizilmesin (boş bırak)
                    if key == selected_group_norm:
                        right_values.append(None)
                    else:
                        right_values.append(val)
                elif key == selected_group_norm and ana_grup_value is not None:
                    # Ana grup için katkıyı boş bırak
                    right_values.append(None)
                else:
                    right_values.append(0.0)

            # Sıralama: aylık değişimi büyükten küçüğe (üstte en büyük)
            combined = list(zip(left_categories, left_values, right_values))
            combined.sort(key=lambda x: (x[1] if x[1] is not None else float('-inf')), reverse=True)
            left_categories = [c for c, _, _ in combined]
            left_values = [v for _, v, _ in combined]
            right_values = [rv for _, _, rv in combined]

            from plotly.subplots import make_subplots
            # Compute dynamic gutter width between subplots based on longest label length
            try:
                max_label_len = max(len(str(n)) for n in left_categories)
            except Exception:
                max_label_len = 20
            gutter = min(0.50, 0.12 + 0.006 * max_label_len)
            comb = make_subplots(rows=1, cols=2, column_widths=[0.46, 0.46],
                                 specs=[[{"type": "bar"}, {"type": "bar"}]],
                                 horizontal_spacing=gutter, shared_yaxes=True)

            left_colors = ['#118AB2' for _ in left_categories]
            # Highlight ana grup if present
            for i, name in enumerate(left_categories):
                if name.strip().lower() == selected_group_norm:
                    left_colors[i] = '#EF476F'

            comb.add_trace(go.Bar(
                y=left_categories, x=left_values, orientation='h',
                marker=dict(color=left_colors, line=dict(width=0)),
                name='Aylık değişim',
                text=[f"<b>{v:+.2f}%</b>" for v in left_values],
                textposition='outside', textfont=dict(size=15, family='Inter, sans-serif', color='#2B2D42'),
                cliponaxis=False,
                hovertemplate='%{y}: %{x:+.2f}%<extra></extra>'
            ), row=1, col=1)

            # Build text labels safely; leave empty when value is None (ana grup satırı)
            right_text = [f"<b>{v:+.2f}</b>" if v is not None else '' for v in right_values]
            comb.add_trace(go.Bar(
                y=left_categories, x=right_values, orientation='h',
                marker=dict(color='#118AB2', line=dict(width=0)),
                name='Katkı',
                text=right_text,
                textposition='outside', textfont=dict(size=15, family='Inter, sans-serif', color='#2B2D42'),
                cliponaxis=False,
                hovertemplate='%{y}: %{x:+.2f} puan<extra></extra>'
            ), row=1, col=2)

            # Ranges
            def compute_range_left(vals):
                if not vals:
                    return [0, 1]
                vmin, vmax = min(vals), max(vals)
                rng = vmax - vmin
                m = rng * 0.3 if rng != 0 else max(abs(vmax), abs(vmin)) * 0.3
                xmin, xmax = vmin - m, vmax + m
                if vmin >= 0:
                    xmin = max(0, vmin - m)
                if vmax <= 0:
                    xmax = min(0, vmax + m)
                return [xmin, xmax]
            comb.update_yaxes(title_text='Harcama Grubu', autorange='reversed', side='right',
                               title_font=dict(size=14, family='Inter, sans-serif', color='#2B2D42'),
                               tickfont=dict(size=14, family='Arial Black, sans-serif', color='#2B2D42'),
                               showticklabels=True, row=1, col=1)
            comb.update_yaxes(showticklabels=False, row=1, col=2)
            comb.update_xaxes(title_text='Değişim (%)', range=compute_range_left(left_values), gridcolor='#E9ECEF', zerolinecolor='#E9ECEF',
                               tickfont=dict(size=12, family='Inter, sans-serif', color='#2B2D42'), row=1, col=1)
            # Let Plotly autoscale right panel; do not force x-range. We only increased gutter for labels.
            comb.update_xaxes(title_text='Katkı (puan)', gridcolor='#E9ECEF', zerolinecolor='#E9ECEF',
                               tickfont=dict(size=12, family='Inter, sans-serif', color='#2B2D42'), row=1, col=2)
            comb.update_layout(title=dict(text=''), barmode='overlay', bargap=0.25, showlegend=False,
                               plot_bgcolor='white', paper_bgcolor='white',
                               height=max(min(len(left_categories) * 70, 1800), 500),
                               margin=dict(l=30, r=30, t=30, b=30))
            combined_graphJSON = comb.to_json()
        except Exception as e:
            print('Katkı grafiği oluşturulamadı:', e)

    selected_harcama_grubu = request.form.get('harcama_grubu') if request.method == 'POST' else None
    harcama_grubu_adlari = harcama_gruplari if harcama_gruplari else []
    harcama_grubu_endeks_graphJSON = None
    harcama_grubu_total_change = None
    harcama_grubu_monthly_change = None

    # Endeks grafiği için harcama grubu seçildiyse veriyi oku ve çiz
    if selected_harcama_grubu:
        toplam_baslik=""
        son_ay=""
        try:
            """worksheet_endeks = spreadsheet.get_worksheet_by_id(2103865002)
            data_endeks = worksheet_endeks.get_all_values()
            df_endeks = pd.DataFrame(data_endeks[1:], columns=data_endeks[0])"""
            df_endeks=pd.read_csv("harcama_grupları.csv").rename(columns={"Unnamed: 0":"Tarih"})
            df_endeks['Tarih'] = pd.to_datetime(df_endeks['Tarih'])
            print('Seçilen harcama grubu:', selected_harcama_grubu)
            print('Endeks tablosu sütunları:', list(df_endeks.columns))
            # Sütun adlarını normalize et
            col_map = {col.strip().lower(): col for col in df_endeks.columns}
            selected_norm = selected_harcama_grubu.strip().lower()
            if selected_norm in col_map:
                real_col = col_map[selected_norm]
                values = df_endeks[real_col]
                dates = df_endeks['Tarih']
                # Türkçe ay isimleriyle x ekseni için
                turkish_months = [f"{d.day} {get_turkish_month(d.strftime('%Y-%m-%d'))} {d.year}" for d in dates]
                tickvals = dates[::max(1, len(dates)//8)]  # 8 aralıkta göster
                ticktext = [f"{get_turkish_month(d.strftime('%Y-%m-%d'))} {d.year}" for d in tickvals]
                # Değişim başlıkları için
                first_date = dates.iloc[0]
                last_date = dates.iloc[-1]
                toplam_baslik = f"{first_date.strftime('%d.%m.%Y')} - {last_date.strftime('%d.%m.%Y')}"
                harcama_grubu_total_change = values.iloc[-1] - values.iloc[0]
                
                # --- Fix: Son ay değişimi ve ay ismi harcama_gruplarıaylık.csv'den alınacak ---
                """worksheet_harcama = spreadsheet.get_worksheet_by_id(1927818004)
                data_harcama = worksheet_harcama.get_all_values()
                df_harcama = pd.DataFrame(data_harcama[1:], columns=data_harcama[0])"""
                df_harcama=pd.read_csv("harcama_gruplarıaylık.csv",index_col=0)
                df_harcama[df_harcama.columns[0]] = df_harcama[df_harcama.columns[0]].str.strip().str.lower()
                row = df_harcama[df_harcama.iloc[:,0] == selected_norm]
                harcama_grubu_monthly_change = None
                son_ay = None
                if not row.empty:
                    last_col = df_harcama.columns[-1]
                    try:
                        harcama_grubu_monthly_change = float(str(row[last_col].values[0]).replace(',', '.'))
                        # Get the month name from the last column of the monthly change CSV
                        son_ay = get_turkish_month(last_col) + f" {datetime.strptime(last_col, '%Y-%m-%d').year}"
                    except:
                        harcama_grubu_monthly_change = None
                        son_ay = None
                # --- End Fix ---
                
                # Read TÜİK data from tuikytd.csv for spending groups
                tuik_dfy = None
                tuik_column_name = None
                try:
                    tuik_dfy = pd.read_csv('tuikytd.csv', index_col=0)
                    tuik_dfy.index = pd.to_datetime(tuik_dfy.index)
                    tuik_dfy = tuik_dfy.sort_index()
                    
                    tuik_dfy.columns=tuik_dfy.columns.str.lower()
                    
                except Exception as e:
                    print(f"TÜİK verisi okunamadı: {e}")
                
                fig_endeks = go.Figure()
                
                # Add Web TÜFE line
                fig_endeks.add_trace(go.Scatter(
                    x=dates,
                    y=values,
                    mode='lines',
                    name=f'Web TÜFE - {selected_harcama_grubu.title()}',
                    line=dict(color='#EF476F', width=3),
                    marker=dict(size=8, color='#EF476F'),
                    hovertemplate='%{x|%d.%m.%Y}<br>Web TÜFE: %{y:.2f}<extra></extra>'
                ))
                
                # Add TÜİK line if data is available
                if tuik_dfy is not None:
                    # Filter TÜİK data to match Web TÜFE date range
                    tuik_filtered = tuik_dfy[tuik_dfy.index >= dates.min()]
                    tuik_filtered = tuik_filtered[tuik_filtered.index <= dates.max()]
                    
                    if not tuik_filtered.empty:
                        fig_endeks.add_trace(go.Scatter(
                            x=tuik_filtered.index,
                            y=tuik_filtered[selected_harcama_grubu],
                            mode='lines',
                            name=f'TÜİK - {selected_harcama_grubu.title()}',
                            line=dict(
                                color='#118AB2',
                                width=3,
                                shape='hv'  # Step grafik
                            ),
                            hovertemplate='%{x|%d.%m.%Y}<br>TÜİK: %{y:.2f}<extra></extra>'
                        ))
                fig_endeks.update_layout(
                    title=dict(
                        text=f'{selected_harcama_grubu.title()} Endeksi',
                        font=dict(size=18, family='Inter, sans-serif', color='#2B2D42'),
                        y=0.98
                    ),
                    xaxis=dict(
                        title='Tarih',
                        title_font=dict(size=12, family='Inter, sans-serif', color='#2B2D42'),
                        tickfont=dict(size=12, family='Inter, sans-serif', color='#2B2D42'),
                        gridcolor='#E9ECEF',
                        zerolinecolor='#E9ECEF',
                        tickangle=0,
                        tickvals=tickvals,
                        ticktext=ticktext
                    ),
                    yaxis=dict(
                        title='Endeks',
                        title_font=dict(size=12, family='Inter, sans-serif', color='#2B2D42'),
                        tickfont=dict(size=12, family='Inter, sans-serif', color='#2B2D42'),
                        gridcolor='#E9ECEF'
                    ),
                    showlegend=True,
                    legend=dict(
                        font=dict(size=12, family='Inter, sans-serif', color='#2B2D42'),
                        bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='#E9ECEF',
                        borderwidth=1,
                        orientation='h',
                        yanchor='bottom',
                        y=1.02,
                        xanchor='right',
                        x=1
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    height=400,
                    margin=dict(l=20, r=20, t=80, b=20),
                    hovermode='x unified',
                    hoverlabel=dict(
                        bgcolor='white',
                        font_size=12,
                        font_family='Inter, sans-serif',
                        namelength=-1
                    )
                )
                harcama_grubu_endeks_graphJSON = fig_endeks.to_json()

                # Monthly change graphs
                try:
                    # Get monthly changes from harcama_gruplarıaylık.csv
                    df_harcama = pd.read_csv("harcama_gruplarıaylık.csv", index_col=0)
                    df_harcama[df_harcama.columns[0]] = df_harcama[df_harcama.columns[0]].str.strip().str.lower()
                    row = df_harcama[df_harcama.iloc[:,0] == selected_norm]
                    
                    if not row.empty:
                        # Get all monthly changes
                        monthly_changes = []
                        monthly_dates = []
                        for col in df_harcama.columns[1:]:
                            try:
                                value = float(str(row[col].values[0]).replace(',', '.'))
                                monthly_changes.append(value)
                                # Convert date format
                                date_obj = datetime.strptime(col, '%Y-%m-%d')
                                monthly_dates.append(f"{get_turkish_month(date_obj.strftime('%Y-%m-%d'))} {date_obj.year}")
                            except:
                                continue

                        if monthly_changes and monthly_dates:  # Veri varsa grafikleri oluştur
                            print(monthly_changes)
                            
                            # TÜİK verilerini al
                            tuik_changes = []
                            try:
                                # tuikaylik.csv dosyasından TÜİK verilerini oku
                                tuik_df = pd.read_csv("tuikaylik.csv", index_col=0)
                                tuik_df.index = pd.to_datetime(tuik_df.index).strftime("%Y-%m")
                                tuik_df.columns=tuik_df.columns.str.lower()
                                
                                print(f"Selected harcama grubu: {selected_harcama_grubu}")
                                print(f"TÜIK CSV columns: {list(tuik_df.columns[:10])}...")  # İlk 10 sütunu göster
                                print(f"Harcama grubu in TÜIK columns: {selected_harcama_grubu in tuik_df.columns}")
                                
                                # TÜİK verilerini aylık değişim tarihleriyle eşleştir
                                for date in monthly_dates:
                                    try:
                                        month, year = date.split()
                                        month_map = {
                                            'Ocak': '01', 'Şubat': '02', 'Mart': '03', 'Nisan': '04',
                                            'Mayıs': '05', 'Haziran': '06', 'Temmuz': '07', 'Ağustos': '08',
                                            'Eylül': '09', 'Ekim': '10', 'Kasım': '11', 'Aralık': '12'
                                        }
                                        date_str = f"{year}-{month_map[month]}"  # YYYY-MM formatı
                                        
                                        if selected_harcama_grubu in tuik_df.columns and date_str in tuik_df.index:
                                            tuik_value = tuik_df.loc[date_str, selected_harcama_grubu]
                                            tuik_changes.append(tuik_value)
                                            print(f"TÜİK value found for {date_str}: {tuik_value}")
                                        else:
                                            tuik_changes.append(None)
                                            print(f"TÜİK value not found for {date_str}")
                                    except Exception as e:
                                        print(f"TÜİK verisi eşleştirme hatası: {e}")
                                        tuik_changes.append(None)
                            except Exception as e:
                                print("TÜİK verisi okunamadı:", e)
                                tuik_changes = [None] * len(monthly_dates)
                            
                            # Bar graph
                            bar_fig = go.Figure()
                            bar_fig.add_trace(go.Bar(
                                x=monthly_dates,
                                y=monthly_changes,
                                name='Web TÜFE',
                                marker_color='#EF476F',
                                text = [f'<b>{v:.2f}</b>' if v is not None else '' for v in monthly_changes],
                                textposition='outside',
                                textfont=dict(size=14, color='#2B2D42', family='Inter, sans-serif'),
                                width=0.35,
                                hovertemplate='%{x}<br>Web TÜFE: %{y:.2f}%<extra></extra>'
                            ))
                            
                            # TÜİK bar ekle
                            bar_fig.add_trace(go.Bar(
                                x=monthly_dates,
                                y=tuik_changes,
                                name='TÜİK',
                                marker_color='#118AB2',
                                text = [f'<b>{v:.2f}</b>' if v is not None else '' for v in tuik_changes],
                                textposition='outside',
                                textfont=dict(size=14, color='#118AB2', family='Inter, sans-serif'),
                                width=0.35,
                                hovertemplate='%{x}<br>TÜİK: %{y:.2f}%<extra></extra>'
                            ))

                            # Calculate y-axis range with margin (including TUIK data)
                            combined_values = monthly_changes + tuik_changes
                            valid_values = [v for v in combined_values if v is not None]
                            y_min = min(valid_values) if valid_values else 0
                            y_max = max(valid_values) if valid_values else 0
                            y_range = y_max - y_min
                            y_margin = y_range * 0.2 if y_range != 0 else abs(y_max) * 0.2
                            y_min_with_margin = y_min - y_margin
                            y_max_with_margin = y_max + y_margin
                            
                            # Sıfıra yaklaşma kontrolü
                            if y_min >= 0:
                                y_min_with_margin = max(0, y_min - y_margin)
                            if y_max <= 0:
                                y_max_with_margin = min(0, y_max + y_margin)

                            bar_fig.update_layout(
                                barmode='group',
                                title=dict(
                                    text=f'{selected_harcama_grubu.title()} Aylık Değişim Oranları',
                                    font=dict(size=20, family='Inter, sans-serif', color='#2B2D42'),
                                    y=0.95
                                ),
                                xaxis=dict(
                                    title='Ay',
                                    title_font=dict(
                                        size=14,
                                        family='Inter, sans-serif',
                                        color='#2B2D42'
                                    ),
                                    tickfont=dict(
                                        size=12,
                                        family='Inter, sans-serif',
                                        color='#2B2D42'
                                    ),
                                    gridcolor='#E9ECEF'
                                ),
                                yaxis=dict(
                                    title='Değişim (%)',
                                    title_font=dict(
                                        size=14,
                                        family='Inter, sans-serif',
                                        color='#2B2D42'
                                    ),
                                    tickfont=dict(
                                        size=12,
                                        family='Inter, sans-serif',
                                        color='#2B2D42'
                                    ),
                                    gridcolor='#E9ECEF',
                                    range=[y_min_with_margin, y_max_with_margin]
                                ),
                                showlegend=True,
                                legend=dict(
                                    orientation='h',
                                    yanchor='bottom',
                                    y=1.02,
                                    xanchor='right',
                                    x=1
                                ),
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                height=400,
                                margin=dict(l=20, r=20, t=80, b=20),
                                hovermode='x'
                            )
                            monthly_bar_graphJSON = json.dumps(bar_fig, cls=plotly.utils.PlotlyJSONEncoder)

                            # Line graph
                            line_fig = go.Figure()
                            line_fig.add_trace(go.Scatter(
                                x=monthly_dates,
                                y=monthly_changes,
                                mode='lines+markers',
                                name='Web TÜFE',
                                line=dict(color='#EF476F', width=3),
                                marker=dict(size=8, color='#EF476F'),
                                hovertemplate='%{x}<br>Web TÜFE: %{y:.2f}%<extra></extra>'
                            ))
                            
                            # TÜİK line ekle
                            line_fig.add_trace(go.Scatter(
                                x=monthly_dates,
                                y=tuik_changes,
                                mode='lines+markers',
                                name='TÜİK',
                                line=dict(color='#118AB2', width=3),
                                marker=dict(size=8, color='#118AB2'),
                                hovertemplate='%{x}<br>TÜİK: %{y:.2f}%<extra></extra>'
                            ))

                            line_fig.update_layout(
                                height=400,
                                title=dict(
                                    text=f'{selected_harcama_grubu.title()} Aylık Değişim Oranları',
                                    font=dict(size=20, family='Inter, sans-serif', color='#2B2D42'),
                                    y=0.95
                                ),
                                xaxis=dict(
                                    title='Ay',
                                    title_font=dict(size=14, family='Inter, sans-serif', color='#2B2D42'),
                                    tickfont=dict(size=12, family='Inter, sans-serif', color='#2B2D42'),
                                    gridcolor='#E9ECEF',
                                    zerolinecolor='#E9ECEF',
                                    tickangle=0
                                ),
                                yaxis=dict(
                                    title='Değişim (%)',
                                    title_font=dict(size=14, family='Inter, sans-serif', color='#2B2D42'),
                                    tickfont=dict(size=12, family='Inter, sans-serif', color='#2B2D42'),
                                    gridcolor='#E9ECEF'
                                ),
                                showlegend=True,
                                legend=dict(
                                    font=dict(size=12, family='Inter, sans-serif', color='#2B2D42'),
                                    bgcolor='rgba(255,255,255,0.8)',
                                    bordercolor='#E9ECEF',
                                    borderwidth=1,
                                    orientation='h',
                                    yanchor='bottom',
                                    y=1.02,
                                    xanchor='right',
                                    x=1
                                ),
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                margin=dict(l=20, r=20, t=80, b=20),
                                hovermode='x'
                            )
                            monthly_line_graphJSON = json.dumps(line_fig, cls=plotly.utils.PlotlyJSONEncoder)
                        else:
                            monthly_bar_graphJSON = None
                            monthly_line_graphJSON = None
                    else:
                        monthly_bar_graphJSON = None
                        monthly_line_graphJSON = None
                except Exception as e:
                    print('Aylık değişim grafikleri oluşturulamadı:', e)
                    bar_graphJSON = None
                    line_graphJSON = None
            else:
                print('Eşleşen sütun bulunamadı!')
        except Exception as e:
            print('Harcama grubu endeks grafiği oluşturulamadı:', e)

    return render_template('harcama_gruplari.html',
        grup_adlari=grup_adlari,
        selected_group=selected_group,
        harcama_gruplari=harcama_gruplari,
        date_options=formatted_date_options,  # Yeni formatlanmış tarih listesi
        selected_date=selected_date,
        active_page='harcama_gruplari',
        harcama_grubu_adlari=harcama_grubu_adlari,
        selected_harcama_grubu=selected_harcama_grubu,
        harcama_grubu_endeks_graphJSON=harcama_grubu_endeks_graphJSON,
        harcama_grubu_total_change=harcama_grubu_total_change,
        harcama_grubu_monthly_change=harcama_grubu_monthly_change,
        toplam_baslik=toplam_baslik if selected_harcama_grubu else None,
        son_ay=son_ay if selected_harcama_grubu else None,
        bar_graphJSON=bar_graphJSON if not selected_harcama_grubu else None,
        line_graphJSON=line_graphJSON if selected_harcama_grubu else None,
        monthly_bar_graphJSON=monthly_bar_graphJSON if selected_harcama_grubu else None,
        monthly_line_graphJSON=monthly_line_graphJSON if selected_harcama_grubu else None,
        show_contrib=show_contrib,
        contrib_type=contrib_type,
        contrib_graphJSON=contrib_graphJSON,
        combined_graphJSON=combined_graphJSON
    )

@app.route('/maddeler', methods=['GET', 'POST'])
def maddeler():
    # Ana grup adlarını al
    df = get_ana_gruplar_data()
    grup_adlari = [col for col in df.drop("Web TÜFE",axis=1).columns if col != 'Tarih']
    selected_group = request.form.get('group') if request.method == 'POST' else grup_adlari[0]
    # ürünler.csv'den madde adlarını al
    urunler = pd.read_csv('ürünler.csv')
    urunler['Ana Grup'] = urunler['Ana Grup'].str.strip().str.lower()
    urunler['Ürün'] = urunler['Ürün'].str.strip().str.lower()
    selected_group_norm = selected_group.strip().lower()
    madde_adlari = urunler[urunler["Ana Grup"] == selected_group_norm]["Ürün"].unique().tolist()
    # 1103913248 ID'li tablodan değişim oranlarını oku
    """creds = get_google_credentials_2()
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key('14iiu_MQwtMxHTFt6ceyFhkk6v0OL-wuoQS1IGPzSpNE')
    worksheet = spreadsheet.get_worksheet_by_id(1103913248)
    data = worksheet.get_all_values()
    df_madde = pd.DataFrame(data[1:], columns=data[0])"""
    df_madde=pd.read_csv("maddeleraylık.csv",index_col=0)
    df_madde[df_madde.columns[0]] = df_madde[df_madde.columns[0]].str.strip().str.lower()
    # Tarih seçenekleri
    date_options = df_madde.columns[1:].tolist()
    formatted_date_options = []
    date_mapping = {}
    for date in date_options:
        try:
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            formatted_date = date_obj.strftime('%Y-%m')
            formatted_date_options.append(formatted_date)
            date_mapping[formatted_date] = date
        except:
            formatted_date_options.append(date)
            date_mapping[date] = date
    formatted_date_options.sort(reverse=True)
    selected_date = request.form.get('date') if request.method == 'POST' else formatted_date_options[0]
    sheet_date = date_mapping[selected_date]
    bar_labels = []
    bar_values = []
    bar_colors = []
    # Maddeler için değerleri al (1103913248)
    for madde in madde_adlari:
        row = df_madde[df_madde.iloc[:,0] == madde]
        if not row.empty:
            try:
                value = float(str(row[sheet_date].values[0]).replace(',', '.'))
                bar_labels.append(madde.title())
                bar_values.append(value)
                bar_colors.append('#118AB2')
            except:
                pass
    # Ana grup değişimini 767776936 ID'li tablodan oku (kırmızı)
    """ana_grup_worksheet = spreadsheet.get_worksheet_by_id(767776936)
    ana_grup_data = ana_grup_worksheet.get_all_values()
    df_ana_grup = pd.DataFrame(ana_grup_data[1:], columns=ana_grup_data[0])"""
    df_ana_grup=pd.read_csv("gruplaraylık.csv",index_col=0)
    df_ana_grup[df_ana_grup.columns[0]] = df_ana_grup[df_ana_grup.columns[0]].str.strip().str.lower()
    ana_grup_row = df_ana_grup[df_ana_grup.iloc[:,0] == selected_group_norm]
    ana_grup_value = None
    if not ana_grup_row.empty:
        try:
            ana_grup_value = float(str(ana_grup_row[sheet_date].values[0]).replace(',', '.'))
        except:
            ana_grup_value = None
    if ana_grup_value is not None:
        bar_labels.append(selected_group.title())
        bar_values.append(ana_grup_value)
        bar_colors.append('#EF476F')
    # Sort bars by value descending (highest first)
    sorted_data = sorted(zip(bar_labels, bar_values, bar_colors), key=lambda x: x[1], reverse=False)
    bar_labels = [x[0] for x in sorted_data]
    bar_values = [x[1] for x in sorted_data]
    bar_colors = [x[2] for x in sorted_data]
    # Barh grafik - harcama_gruplari ile aynı şekilde
    try:
        selected_date_obj = datetime.strptime(selected_date, '%Y-%m')
        turkish_month = get_turkish_month(selected_date_obj.strftime('%Y-%m-%d'))
    except Exception:
        turkish_month = selected_date
    valid_bar_values = [v for v in bar_values if v is not None]
    x_min = min(valid_bar_values + [0]) if valid_bar_values else 0
    x_max = max(valid_bar_values + [0]) if valid_bar_values else 0
    
    # Calculate dynamic margin based on data characteristics
    data_range = x_max - x_min
    max_abs_value = max(abs(x_min), abs(x_max))
    
    # Dynamic margin calculation based on data magnitude and range
    if data_range == 0:
        # If all values are the same, use a fixed margin
        dynamic_margin = max_abs_value * 0.3 + 1.0
    elif max_abs_value < 5:
        # Small values: use larger relative margin for text visibility
        dynamic_margin = max(data_range * 0.4, 2.0)
    elif max_abs_value < 20:
        # Medium values: balanced margin
        dynamic_margin = max(data_range * 0.25, 3.0)
    else:
        # Large values: smaller relative margin but ensure minimum
        dynamic_margin = max(data_range * 0.15, max_abs_value * 0.1)
    
    # Ensure minimum margin for text visibility
    dynamic_margin = max(dynamic_margin, 2.5)
    
    # Calculate range with dynamic margins
    if x_min < 0:
        # For negative values, extend left more to accommodate text
        left_margin = dynamic_margin * 1.5
    else:
        # For positive values, smaller left margin
        left_margin = dynamic_margin * 0.5
    
    if x_max > 0:
        # For positive values, extend right more to accommodate text
        right_margin = dynamic_margin * 1.5
    else:
        # For negative values, smaller right margin
        right_margin = dynamic_margin * 0.5
    
    # Apply margins
    x_min_with_margin = x_min - left_margin
    x_max_with_margin = x_max + right_margin
    
    # Ensure the range includes zero if values span both positive and negative
    if x_min < 0 and x_max > 0:
        xaxis_range = [min(x_min_with_margin, x_min), max(x_max_with_margin, x_max)]
    else:
        xaxis_range = [x_min_with_margin, x_max_with_margin]
    
    # Ensure range is not inverted
    if xaxis_range[0] > xaxis_range[1]:
        xaxis_range = [xaxis_range[1], xaxis_range[0]]
    fig = go.Figure(go.Bar(
        y=bar_labels,
        x=bar_values,
        orientation='h',
        marker_color=bar_colors,
        cliponaxis=False,
        hovertemplate='%{y}: %{x:.2f}<extra></extra>'
    ))
    # Calculate minimal text offset to place text exactly at bar end
    range_span = xaxis_range[1] - xaxis_range[0]
    text_offset = range_span * 0.001  # Reduced to 0.1% for minimal gap
    
    for i, value in enumerate(bar_values):
        # Determine text position based on value sign and magnitude
        if value >= 0:
            # For positive values, place text to the right of the bar
            text_x = value + text_offset
            align_anchor = 'left'
        else:
            # For negative values, place text to the left of the bar
            text_x = value - text_offset
            align_anchor = 'right'
        
        fig.add_annotation(
            x=text_x,
            y=bar_labels[i],
            text=f"<b>{value:.2f}%</b>",
            showarrow=False,
            font=dict(size=15, family="Inter Bold, Inter, sans-serif", color="#2B2D42"),
            align=align_anchor,
            xanchor=align_anchor,
            yanchor='middle'
        )
    fig.update_layout(
        title=dict(
            text=f'{turkish_month} Ayı Madde Değişimleri',
            font=dict(
                size=20,
                family='Inter, sans-serif',
                color='#2B2D42'
            ),
            y=0.999,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        xaxis=dict(
            title='Değişim (%)',
            title_font=dict(
                size=14,
                family='Inter, sans-serif',
                color='#2B2D42'
            ),
            tickfont=dict(
                size=14,
                family='Inter, sans-serif',
                color='#2B2D42'
            ),
            gridcolor='#E9ECEF',
            zerolinecolor='#E9ECEF',
            range=xaxis_range
        ),
        yaxis=dict(
            title='Harcama Grubu',
            title_font=dict(
                size=14,
                family='Inter, sans-serif',
                color='#2B2D42'
            ),
            tickfont=dict(
                size=14,
                family='Inter, sans-serif',
                color='#2B2D42'
            ),
            gridcolor='#E9ECEF'
        ),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=max(min(len(bar_labels) * 70, 3000), 500),
        margin=dict(l=10, r=10, t=40, b=20),
        hovermode='y unified',
        hoverlabel=dict(bgcolor='white', font_size=12, font_family='Inter, sans-serif', font_color='#2B2D42')
    )
    bar_graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('maddeler.html',
        grup_adlari=grup_adlari,
        selected_group=selected_group,
        madde_adlari=madde_adlari,
        date_options=formatted_date_options,
        selected_date=selected_date,
        bar_graphJSON=bar_graphJSON,
        active_page='maddeler'
    )

@app.route('/ozel-kapsamli-gostergeler', methods=['GET', 'POST'])
def ozel_kapsamli_gostergeler():
    # Google Sheets API setup
    """creds = get_google_credentials_2()
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key('14iiu_MQwtMxHTFt6ceyFhkk6v0OL-wuoQS1IGPzSpNE')
    
    # Get endeks data from worksheet
    worksheet = spreadsheet.get_worksheet_by_id(1456874598)
    data = worksheet.get_all_values()
    df = pd.DataFrame(data[1:], columns=data[0])"""

    df=pd.read_csv("özelgöstergeler.csv").rename(columns={"Unnamed: 0":"Tarih"})
    
    # Convert date column to datetime
    df['Tarih'] = pd.to_datetime(df['Tarih'])
    
    # Get indicator names (column names except 'Tarih')
    indicator_names = [col for col in df.columns if col != 'Tarih']
    
    # Get selected indicator from form
    selected_indicator = request.form.get('indicator') if request.method == 'POST' else indicator_names[0]
    
    # Get monthly change data
    """monthly_worksheet = spreadsheet.get_worksheet_by_id(1767722805)
    monthly_data = monthly_worksheet.get_all_values()
    df_monthly = pd.DataFrame(monthly_data[1:], columns=monthly_data[0])"""
    df_monthly=pd.read_csv("özelgöstergeleraylık.csv",index_col=0)
    
    # Get monthly change for selected indicator
    monthly_change = None
    if not df_monthly.empty:
        indicator_row = df_monthly[df_monthly.iloc[:,0].str.strip().str.lower() == selected_indicator.strip().lower()]
        if not indicator_row.empty:
            try:
                monthly_change = float(str(indicator_row.iloc[:,-1].values[0]).replace(',', '.'))
            except:
                monthly_change = None
    
    # Calculate total change
    total_change = None
    if not df.empty and selected_indicator in df.columns:
        values = df[selected_indicator]
        total_change = values.iloc[-1] - 100
    
    # Read TÜİK endeks data from tuikozelgostergelerytd.csv
    tuik_endeks_df = None
    try:
        tuik_endeks_df = pd.read_csv('tuikozelgostergelerytd.csv', index_col=0)
        tuik_endeks_df.index = pd.to_datetime(tuik_endeks_df.index)
        tuik_endeks_df = tuik_endeks_df.sort_index()
    except Exception as e:
        print(f"TÜİK endeks verisi okunamadı: {e}")
    
    # Create line plot
    fig = go.Figure()
    
    if not df.empty and selected_indicator in df.columns:
        values = df[selected_indicator]
        dates = df['Tarih']
        
        # Get Turkish month names for hover text
        turkish_dates = [f"{d.day} {get_turkish_month(d.strftime('%Y-%m-%d'))} {d.year}" for d in dates]
        
        # Get first day of each month for x-axis ticks
        aybasi_tarihler = [d for d in dates if d.day == 1]
        tickvals = aybasi_tarihler
        ticktext = [f"{get_turkish_month(d.strftime('%Y-%m-%d'))} {d.year}" for d in aybasi_tarihler]
        
        # Add Web TÜFE line
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines',
            name=f'Web TÜFE - {selected_indicator}',
            line=dict(color='#EF476F', width=3),
            customdata=turkish_dates,
            hovertemplate='<b>%{customdata}</b><br>' + f'Web TÜFE - {selected_indicator}: ' + '%{y:.2f}<extra></extra>'
        ))
        
        # Add TÜİK endeks line if data is available
        if tuik_endeks_df is not None and selected_indicator in tuik_endeks_df.columns:
            # Filter TÜİK data to match Web TÜFE date range
            tuik_filtered = tuik_endeks_df[tuik_endeks_df.index >= dates.min()]
            tuik_filtered = tuik_filtered[tuik_filtered.index <= dates.max()]
            
            if not tuik_filtered.empty:
                tuik_turkish_dates = [f"{d.day} {get_turkish_month(d.strftime('%Y-%m-%d'))} {d.year}" for d in tuik_filtered.index]
                fig.add_trace(go.Scatter(
                    x=tuik_filtered.index,
                    y=tuik_filtered[selected_indicator],
                    mode='lines',
                    name=f'TÜİK - {selected_indicator}',
                    line=dict(
                        color='#118AB2',
                        width=3,
                        shape='hv'  # Step grafik
                    ),
                    customdata=tuik_turkish_dates,
                    hovertemplate='<b>%{customdata}</b><br>' + f'TÜİK - {selected_indicator}: ' + '%{y:.2f}<extra></extra>'
                ))
        
        fig.update_layout(
            title=dict(
                text=f'{selected_indicator} Endeksi',
                font=dict(
                    size=18,
                    family='Inter, sans-serif',
                    color='#2B2D42'
                ),
                y=0.98
            ),
            xaxis=dict(
                title='Tarih',
                title_font=dict(
                    size=12,
                    family='Inter, sans-serif',
                    color='#2B2D42'
                ),
                tickfont=dict(
                    size=12,
                    family='Inter, sans-serif',
                    color='#2B2D42'
                ),
                gridcolor='#E9ECEF',
                zerolinecolor='#E9ECEF',
                tickvals=tickvals,
                ticktext=ticktext,
                tickangle=0,
                hoverformat='',
            ),
            yaxis=dict(
                title='Endeks',
                title_font=dict(
                    size=12,
                    family='Inter, sans-serif',
                    color='#2B2D42'
                ),
                tickfont=dict(
                    size=12,
                    family='Inter, sans-serif',
                    color='#2B2D42'
                ),
                gridcolor='#E9ECEF'
            ),
            showlegend=True,
            legend=dict(
                font=dict(size=12, family='Inter, sans-serif', color='#2B2D42'),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#E9ECEF',
                borderwidth=1,
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=500,
            margin=dict(l=5, r=5, t=20, b=10),
            hovermode='x unified',
            hoverlabel=dict(
                bgcolor='white',
                font_size=12,
                font_family='Inter, sans-serif',
                namelength=-1
            )
        )

    # Aylık değişim verilerini al
    """monthly_worksheet = spreadsheet.get_worksheet_by_id(1767722805)
    monthly_data = monthly_worksheet.get_all_values()
    df_monthly = pd.DataFrame(monthly_data[1:], columns=monthly_data[0])"""
    df_monthly=pd.read_csv("özelgöstergeleraylık.csv",index_col=0)
    df_monthly[df_monthly.columns[0]] = df_monthly[df_monthly.columns[0]].str.strip().str.lower()
    selected_indicator_norm = selected_indicator.strip().lower()
    monthly_row = df_monthly[df_monthly.iloc[:,0] == selected_indicator_norm]

    # Aylık değişim verilerini hazırla
    monthly_changes = []
    monthly_dates = []
    if not monthly_row.empty:
        for col in df_monthly.columns[1:]:
            try:
                value = float(str(monthly_row[col].values[0]).replace(',', '.'))
                monthly_changes.append(value)
                date_obj = datetime.strptime(col, '%Y-%m-%d')
                monthly_dates.append(f"{get_turkish_month(date_obj.strftime('%Y-%m-%d'))} {date_obj.year}")
            except Exception as e:
                monthly_changes.append(None)
                monthly_dates.append(col)

    # TÜİK verilerini oku
    tuik_changes = []
    
    try:
        tuik_df = pd.read_csv('tüiközelgöstergeler.csv', index_col=0)
        tuik_df.index=pd.to_datetime(tuik_df.index).strftime("%Y-%m")
        # TÜİK verilerini aylık değişim tarihleriyle eşleştir
        for date in monthly_dates:
            print(f"{date} aranıyor")
            try:
                month, year = date.split()
                month_map = {
                    'Ocak': '01', 'Şubat': '02', 'Mart': '03', 'Nisan': '04',
                    'Mayıs': '05', 'Haziran': '06', 'Temmuz': '07', 'Ağustos': '08',
                    'Eylül': '09', 'Ekim': '10', 'Kasım': '11', 'Aralık': '12'
                }
                date_str = f"{year}-{month_map[month]}"  # CSV'deki tarih formatı
                if selected_indicator in tuik_df.columns and date_str in tuik_df.index:
                    tuik_value = tuik_df.loc[date_str, selected_indicator]
                    tuik_changes.append(tuik_value)
                else:
                    tuik_changes.append(None)

                print(tuik_changes)
            except Exception as e:
                print(f"TÜİK verisi eşleştirme hatası: {e}")
                tuik_changes.append(None)
    except Exception as e:
        print("TÜİK verisi okunamadı:", e)
        tuik_changes = [None] * len(monthly_dates)

    # Bar grafik
    bar_fig = go.Figure()
    # Web TÜFE
    bar_fig.add_trace(go.Bar(
        x=monthly_dates,
        y=monthly_changes,
        name='Web TÜFE',
        marker_color='#EF476F',
        text = [f'<b>{v:.2f}</b>' if v is not None else '' for v in monthly_changes],
        textposition='outside',
        textfont=dict(size=14, color='#2B2D42', family='Inter, sans-serif'),
        width=0.35,
        hovertemplate='%{x}<br>Web TÜFE: %{y:.2f}%<extra></extra>'
    ))

    # TÜİK
    bar_fig.add_trace(go.Bar(
        x=monthly_dates,
        y=tuik_changes,
        name='TÜİK',
        marker_color='#118AB2',
        text = [f'<b>{v:.2f}</b>' if v is not None else '' for v in tuik_changes],
        textposition='outside',
        textfont=dict(size=14, color='#118AB2', family='Inter, sans-serif'),
        width=0.35,
        hovertemplate='%{x}<br>TÜİK: %{y:.2f}%<extra></extra>'
    ))

    # Y ekseni aralığını hesapla
    combined_values = monthly_changes + tuik_changes
    valid_values = [v for v in combined_values if v is not None]
    if valid_values:
        y_min = min(valid_values)
        y_max = max(valid_values)
        y_range = y_max - y_min
        y_margin = y_range * 0.2 if y_range != 0 else abs(y_max) * 0.2
        y_min_with_margin = y_min - y_margin
        y_max_with_margin = y_max + y_margin
        if y_min >= 0:
            y_min_with_margin = max(0, y_min - y_margin)
        if y_max <= 0:
            y_max_with_margin = min(0, y_max + y_margin)

    bar_fig.update_layout(
        barmode='group',
        title=dict(
            text=f'{selected_indicator} Aylık Değişim Oranları',
            font=dict(
                size=20,
                family='Inter, sans-serif',
                color='#2B2D42'
            ),
            y=0.95
        ),
        xaxis=dict(
            title='Ay',
            title_font=dict(
                size=14,
                family='Inter, sans-serif',
                color='#2B2D42'
            ),
            tickfont=dict(
                size=12,
                family='Inter, sans-serif',
                color='#2B2D42'
            ),
            gridcolor='#E9ECEF'
        ),
        yaxis=dict(
            title='Değişim (%)',
            title_font=dict(
                size=14,
                family='Inter, sans-serif',
                color='#2B2D42'
            ),
            tickfont=dict(
                size=12,
                family='Inter, sans-serif',
                color='#2B2D42'
            ),
            gridcolor='#E9ECEF',
            range=[y_min_with_margin, y_max_with_margin]
        ),
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400,
        margin=dict(l=10, r=10, t=40, b=20),
        hovermode='x',
        autosize=True
    )

    # Line grafik
    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(
        x=monthly_dates,
        y=monthly_changes,
        mode='lines+markers',
        name='Web TÜFE',
        line=dict(color='#EF476F', width=3),
        marker=dict(size=8, color='#EF476F'),
        hovertemplate='%{x}<br>Web TÜFE: %{y:.2f}%<extra></extra>'
    ))
    line_fig.add_trace(go.Scatter(
        x=monthly_dates,
        y=tuik_changes,
        mode='lines+markers',
        name='TÜİK',
        line=dict(color='#118AB2', width=3),
        marker=dict(size=8, color='#118AB2'),
        hovertemplate='%{x}<br>TÜİK: %{y:.2f}%<extra></extra>'
    ))

    line_fig.update_layout(
        title=dict(
            text=f'{selected_indicator} Aylık Değişim Oranları',
            font=dict(
                size=14,
                family='Inter, sans-serif',
                color='#2B2D42'
            ),
            y=0.95
        ),
        xaxis=dict(
            title='Ay',
            title_font=dict(
                size=14,
                family='Inter, sans-serif',
                color='#2B2D42'
            ),
            tickfont=dict(
                size=12,
                family='Inter, sans-serif',
                color='#2B2D42'
            ),
            tickangle=0
        ),
        yaxis=dict(
            title='Değişim (%)',
            title_font=dict(
                size=14,
                family='Inter, sans-serif',
                color='#2B2D42'
            ),
            tickfont=dict(
                size=12,
                family='Inter, sans-serif',
                color='#2B2D42'
            ),
            gridcolor='#E9ECEF'
            ),
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400,
        margin=dict(l=10, r=10, t=40, b=20),
        hovermode='x',
        autosize=True
    )

    bar_graphJSON = json.dumps(bar_fig, cls=plotly.utils.PlotlyJSONEncoder)
    line_graphJSON = json.dumps(line_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Get the last month from the monthly CSV file
    last_month_from_csv = None
    if not df_monthly.empty:
        # Get the last column (excluding the first column which is the group name)
        last_column = df_monthly.columns[-1]
        try:
            # Parse the date from the column name (format: YYYY-MM-DD)
            date_obj = datetime.strptime(last_column, '%Y-%m-%d')
            last_month_from_csv = get_turkish_month(date_obj.strftime('%Y-%m-%d'))
        except:
            last_month_from_csv = None
    
    return render_template('ozel_kapsamli_gostergeler.html',
    graphJSON=graphJSON,
    indicator_names=indicator_names,
    selected_indicator=selected_indicator,
    total_change=total_change,
    monthly_change=monthly_change,
    active_page='ozel_kapsamli_gostergeler',
    last_date=dates.iloc[-1] if not df.empty else None,
    month_name=last_month_from_csv if last_month_from_csv else (get_turkish_month(dates.iloc[-1].strftime('%Y-%m-%d')) if not df.empty else None),
    bar_graphJSON=bar_graphJSON,
    line_graphJSON=line_graphJSON
)

@app.route('/mevsimsel-duzeltilmis-gostergeler', methods=['GET', 'POST'])
def mevsimsel_duzeltilmis_gostergeler():
    # Read data from ma.xlsx file (Web TÜFE)
    df = pd.read_excel("ma.xlsx")
    
    # Read data from matuik.xlsx file (TÜİK)
    df_tuik = pd.read_excel("matuik.xlsx")
    
    # Get indicator names from the 'Gösterge' column
    indicator_names = df['Gösterge'].dropna().tolist()
    
    # Get selected indicator from form
    selected_indicator = request.form.get('indicator') if request.method == 'POST' else indicator_names[0]
    
    # Get the selected indicator data from Web TÜFE
    indicator_data = df[df['Gösterge'] == selected_indicator]
    
    # Get the selected indicator data from TÜİK (if exists)
    # Special case: if "Web TÜFE" is selected, show "TÜFE" from TÜİK data
    if selected_indicator == "Web TÜFE":
        indicator_data_tuik = df_tuik[df_tuik['Gösterge'] == "TÜFE"]
    else:
        indicator_data_tuik = df_tuik[df_tuik['Gösterge'] == selected_indicator]
    
    # Calculate total change (from first to last month)
    total_change = 0.0  # Default to 0.0 instead of None
    monthly_change = 0.0  # Default to 0.0 instead of None
    last_date = datetime.now()  # Default to current date
    
    if not indicator_data.empty:
        # Get date columns (all columns except 'Unnamed: 0' and 'Gösterge')
        date_columns = [col for col in df.columns if col not in ['Unnamed: 0', 'Gösterge']]
        values = indicator_data[date_columns].iloc[0].values
        if len(values) > 1:
            total_change = values[-1] - values[0]
            # Show the last value itself
            monthly_change = values[-1]
    
    # Create line plot
    fig = go.Figure()
    
    if not indicator_data.empty:
        # Get date columns (all columns except 'Unnamed: 0' and 'Gösterge')
        date_columns = [col for col in df.columns if col not in ['Unnamed: 0', 'Gösterge']]
        values = indicator_data[date_columns].iloc[0].values
        dates = date_columns
        
        # Convert date strings to datetime for proper formatting
        date_objects = []
        turkish_dates = []
        for date_str in dates:
            try:
                # Parse YYYY-MM format
                date_obj = datetime.strptime(date_str, '%Y-%m')
                date_objects.append(date_obj)
                turkish_dates.append(f"{get_turkish_month(date_obj.strftime('%Y-%m'))} {date_obj.year}")
            except:
                date_objects.append(date_str)
                turkish_dates.append(date_str)
        
        # Add Web TÜFE line
        fig.add_trace(go.Scatter(
            x=turkish_dates,
            y=values,
            mode='lines+markers',
            name=f'Web TÜFE - {selected_indicator}',
            line=dict(color='#EF476F', width=3),
            marker=dict(size=8, color='#EF476F'),
            customdata=turkish_dates,
            hovertemplate='<b>%{customdata}</b><br>' + f'Web TÜFE - {selected_indicator}: ' + '%{y:.2f}<extra></extra>'
        ))
    
    # Add TÜİK data if it exists for the selected indicator
    if not indicator_data_tuik.empty:
        # Get date columns from TÜİK data
        date_columns_tuik = [col for col in df_tuik.columns if col not in ['Gösterge']]
        values_tuik = indicator_data_tuik[date_columns_tuik].iloc[0].values
        dates_tuik = date_columns_tuik
        
        # Convert date strings to datetime for proper formatting
        turkish_dates_tuik = []
        for date_str in dates_tuik:
            try:
                # Parse YYYY-MM format
                date_obj = datetime.strptime(date_str, '%Y-%m')
                turkish_dates_tuik.append(f"{get_turkish_month(date_obj.strftime('%Y-%m'))} {date_obj.year}")
            except:
                turkish_dates_tuik.append(date_str)
        
        # Add TÜİK line
        fig.add_trace(go.Scatter(
            x=turkish_dates_tuik,
            y=values_tuik,
            mode='lines+markers',
            name=f'TÜİK - {selected_indicator}',
            line=dict(color='#2B2D42', width=3),
            marker=dict(size=8, color='#2B2D42'),
            customdata=turkish_dates_tuik,
            hovertemplate='<b>%{customdata}</b><br>' + f'TÜİK - {selected_indicator}: ' + '%{y:.2f}<extra></extra>'
        ))
    
    # Update layout for all cases
    fig.update_layout(
        title=dict(
            text=f'Mevsimsel Düzeltilmiş {selected_indicator} Değişimi - Web TÜFE vs TÜİK',
            font=dict(
                size=18,
                family='Inter, sans-serif',
                color='#2B2D42'
            ),
            y=0.98
        ),
        xaxis=dict(
            title='Tarih',
            title_font=dict(
                size=12,
                family='Inter, sans-serif',
                color='#2B2D42'
            ),
            tickfont=dict(
                size=12,
                family='Inter, sans-serif',
                color='#2B2D42'
            ),
            gridcolor='#E9ECEF',
            zerolinecolor='#E9ECEF',
            tickangle=0,
            hoverformat='',
        ),
        yaxis=dict(
            title='Endeks',
            title_font=dict(
                size=12,
                family='Inter, sans-serif',
                color='#2B2D42'
            ),
            tickfont=dict(
                size=12,
                family='Inter, sans-serif',
                color='#2B2D42'
            ),
            gridcolor='#E9ECEF'
        ),
        showlegend=True,
        legend=dict(
            font=dict(size=12, family='Inter, sans-serif', color='#2B2D42'),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#E9ECEF',
            borderwidth=1,
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        margin=dict(l=5, r=5, t=20, b=10),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='white',
            font_size=12,
            font_family='Inter, sans-serif',
            namelength=-1
        )
    )


    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Get the last month from the data
    last_month_from_csv = None
    if not indicator_data.empty:
        # Get date columns (all columns except 'Unnamed: 0' and 'Gösterge')
        date_columns = [col for col in df.columns if col not in ['Unnamed: 0', 'Gösterge']]
        if len(date_columns) > 0:
            last_column = date_columns[-1]
            try:
                date_obj = datetime.strptime(last_column, '%Y-%m')
                last_month_from_csv = get_turkish_month(date_obj.strftime('%Y-%m'))
            except:
                last_month_from_csv = None
    
    return render_template('mevsimsel_duzeltilmis_gostergeler.html',
        graphJSON=graphJSON,
        indicator_names=indicator_names,
        selected_indicator=selected_indicator,
        total_change=total_change,
        monthly_change=monthly_change,
        active_page='mevsimsel_duzeltilmis_gostergeler',
        last_date=last_date,
        month_name=last_month_from_csv
    )

@app.route('/hakkinda')
def hakkinda():
    return render_template('hakkinda.html', active_page='hakkinda')

@app.route('/send-contact', methods=['POST'])
def send_contact():
    try:
        # Form verilerini al
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        
        # Boş alan kontrolü
        if not name or not email or not message:
            return jsonify({'success': False, 'message': 'Lütfen tüm alanları doldurun'}), 400
        
        # Email içeriğini oluştur
        subject = f"Web TÜFE İletişim Formu - {name}"
        
        # HTML formatında email içeriği
        html_content = f"""
        <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
                    <h2 style="color: #4F46E5; border-bottom: 3px solid #06B6D4; padding-bottom: 10px;">
                        Yeni İletişim Formu Mesajı
                    </h2>
                    <div style="background-color: #F3F4F6; padding: 15px; border-radius: 8px; margin: 20px 0;">
                        <p style="margin: 10px 0;"><strong style="color: #4F46E5;">Gönderen:</strong> {name}</p>
                        <p style="margin: 10px 0;"><strong style="color: #4F46E5;">E-posta:</strong> {email}</p>
                    </div>
                    <div style="background-color: #fff; padding: 20px; border-left: 4px solid #06B6D4; margin: 20px 0;">
                        <h3 style="color: #4F46E5; margin-top: 0;">Mesaj:</h3>
                        <p style="white-space: pre-wrap;">{message}</p>
                    </div>
                    <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 12px; color: #666;">
                        <p>Bu mesaj Web TÜFE iletişim formundan gönderilmiştir.</p>
                        <p>Tarih: {datetime.now().strftime('%d.%m.%Y %H:%M')}</p>
                    </div>
                </div>
            </body>
        </html>
        """
        
        # Email mesajını oluştur
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = 'webtufe@gmail.com'  # Bu email SMTP ile gönderen adres olacak
        msg['To'] = 'borakaya8@gmail.com'
        msg['Reply-To'] = email  # Cevap verirken kullanıcının emailine gidecek
        
        # HTML içeriği ekle
        html_part = MIMEText(html_content, 'html', 'utf-8')
        msg.attach(html_part)
        
        # SMTP ile email gönder
        # Gmail SMTP kullanıyorsanız, "Uygulama Şifresi" oluşturmanız gerekir
        # Google Hesabınız > Güvenlik > 2 Adımlı Doğrulama > Uygulama Şifreleri
        
        smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.environ.get('SMTP_PORT', '587'))
        smtp_username = os.environ.get('SMTP_USERNAME', 'borakaya8@gmail.com')
        smtp_password = os.environ.get('SMTP_PASSWORD', '')
        
        if not smtp_password:
            # Eğer SMTP şifresi yoksa, basit bir mail gönder (development için)
            print(f"Email gönderilecek: {name} ({email})")
            print(f"Mesaj: {message}")
            return jsonify({'success': True, 'message': 'Mesajınız alındı (development mode)'}), 200
        
        # SMTP bağlantısı kur ve email gönder
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # TLS şifrelemesini başlat
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
        
        return jsonify({'success': True, 'message': 'Mesajınız başarıyla gönderildi'}), 200
        
    except smtplib.SMTPException as e:
        print(f"SMTP Hatası: {str(e)}")
        return jsonify({'success': False, 'message': 'Email gönderilirken bir hata oluştu'}), 500
    except Exception as e:
        print(f"Genel Hata: {str(e)}")
        return jsonify({'success': False, 'message': 'Bir hata oluştu'}), 500

@app.route('/bultenler', methods=['GET', 'POST'])
def bultenler():
    # List all PDF files in the 'bültenler' directory
    bultenler_dir = os.path.join(os.path.dirname(__file__), 'bültenler')
    pdf_files = [f for f in os.listdir(bultenler_dir) if f.lower().endswith('.pdf')]
    # Extract Turkish month and year from filenames
    aylar = ['Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran', 'Temmuz', 'Ağustos', 'Eylül', 'Ekim', 'Kasım', 'Aralık']
    date_options = []
    file_map = {}
    for fname in pdf_files:
        for ay in aylar:
            if fname.startswith(ay):
                y = fname.replace(ay, '').replace('.pdf', '').replace('_', ' ').strip()
                label = f"{ay} {y}"
                date_options.append(label)
                file_map[label] = fname
                break
    # Tarihleri yıl ve ay'a göre sıralayalım (en yeni en başta)
    def parse_turkish_date(label):
        try:
            ay, yil = label.split()
            ay_map = {a: i+1 for i, a in enumerate(aylar)}
            return int(yil), ay_map.get(ay, 0)
        except:
            return (0, 0)
    date_options.sort(key=parse_turkish_date, reverse=True)
    selected_date = request.form.get('bulten_tarihi') if request.method == 'POST' else (date_options[0] if date_options else None)
    selected_file = file_map[selected_date] if selected_date in file_map else None
    return render_template('bultenler.html', date_options=date_options, selected_date=selected_date, selected_file=selected_file)

@app.route('/bultenler/pdf/<filename>')
def serve_bulten_pdf(filename):
    return render_template('pdf_viewer.html', filename=filename)

@app.route('/bultenler/pdf-direct/<filename>')
def serve_bulten_pdf_direct(filename):
    bultenler_dir = os.path.join(os.path.dirname(__file__), 'bültenler')
    return send_file(os.path.join(bultenler_dir, filename))

@app.route('/download/fiyatlar')
def download_fiyatlar():
    try:
        # Check if Fiyatlar.zip exists in the root directory
        file_path = os.path.join(os.path.dirname(__file__), 'Fiyatlar.zip')
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name='Fiyatlar.zip')
        else:
            flash('Fiyatlar.zip dosyası bulunamadı.', 'error')
            return redirect(url_for('ana_sayfa'))
    except Exception as e:
        flash(f'Dosya indirme hatası: {str(e)}', 'error')
        return redirect(url_for('ana_sayfa'))

@app.route('/metodoloji/pdf')
def serve_metodoloji_pdf():
    import os
    metodoloji_dir = os.path.join(os.path.dirname(__file__), 'metodoloji')
    return send_file(os.path.join(metodoloji_dir, 'Metodoloji.pdf'))

@app.route('/metodoloji')
def metodoloji():
    return render_template('metodoloji.html')

@app.route('/abone', methods=['POST'])
def abone():
    try:
        print("Google Sheets bağlantısı başlatılıyor...")
        
        try:
            creds = get_google_credentials()
            print("Credentials başarıyla yüklendi")
        except Exception as e:
            print(f"Credentials yükleme hatası: {str(e)}")
            flash("Kimlik doğrulama hatası. Lütfen yönetici ile iletişime geçin.", "error")
            return redirect(url_for('index'))

        try:
            client = gspread.authorize(creds)
            print("Google Sheets API'ye başarıyla bağlanıldı")
        except Exception as e:
            print(f"Google Sheets API bağlantı hatası: {str(e)}")
            flash("Google Sheets bağlantı hatası. Lütfen daha sonra tekrar deneyin.", "error")
            return redirect(url_for('index'))
        
        sheet_url = "https://docs.google.com/spreadsheets/d/1Y3SpFSsASfCzrM7iM-j_x5XR5pYv__8etC4ptaA9dio"
        try:
            print(f"Google Sheet açılmaya çalışılıyor: {sheet_url}")
            worksheet = client.open_by_url(sheet_url).sheet1
            print("Google Sheet başarıyla açıldı")
        except (APIError, SpreadsheetNotFound) as e:
            print(f"Google Sheet erişim hatası: {str(e)}")
            flash("Google Sheet erişim hatası. Lütfen daha sonra tekrar deneyin.", "error")
            return redirect(url_for('index'))

        email = request.form.get('email')
        action = request.form.get('action')
        print(f"İşlem: {action}, E-posta: {email}")
        
        if not email or '@' not in email or '.' not in email:
            flash("Lütfen geçerli bir e-posta adresi girin.", "error")
            return redirect(url_for('index'))

        try:
            emails = worksheet.col_values(1)
            print(f"Mevcut e-posta sayısı: {len(emails)}")
            
            if action == "Abone ol":
                if email in emails:
                    print(f"E-posta zaten abone: {email}")
                    flash("Bu e-posta zaten abone.", "info")
                else:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"Yeni abone ekleniyor: {email}")
                    worksheet.append_row([email, now])
                    flash("Aboneliğiniz başarıyla eklendi 🎉", "success")
            elif action == "Çık":
                if email in emails:
                    print(f"Abonelik iptal ediliyor: {email}")
                    cell = worksheet.find(email)
                    worksheet.delete_rows(cell.row)
                    flash("Aboneliğiniz iptal edildi.", "success")
                else:
                    print(f"E-posta abone değil: {email}")
                    flash("Bu e-posta zaten abone değil.", "info")
        except Exception as e:
            print(f"Google Sheets işlem hatası: {str(e)}")
            flash("İşlem sırasında bir hata oluştu. Lütfen daha sonra tekrar deneyin.", "error")
            
    except Exception as e:
        print(f"Genel hata: {str(e)}")
        import traceback
        print("Hata detayı:")
        print(traceback.format_exc())
        flash("Sistem hatası. Lütfen daha sonra tekrar deneyin.", "error")
        
    return redirect(url_for('ana_sayfa'))

@app.route('/korelasyon-analizi', methods=['GET', 'POST'])
def korelasyon_analizi():
    try:
        # Ana gruplar verilerini oku
        df = pd.read_csv("gruplaraylık.csv", index_col=0)
        
        # Tarih sütunlarını al (ilk sütun grup adı olduğu için atla)
        date_columns = df.columns[1:]
        
        # Son 6 ay için varsayılan
        default_months = 6
        
        if request.method == 'POST':
            selected_months = int(request.form.get('months', default_months))
            analysis_type = request.form.get('analysis_type', 'ana_gruplar')
        else:
            selected_months = default_months
            analysis_type = 'ana_gruplar'
        
        # Veri setini seç
        if analysis_type == 'harcama_gruplari':
            df = pd.read_csv("harcama_gruplarıaylık.csv", index_col=0)
        else:
            df = pd.read_csv("gruplaraylık.csv", index_col=0)
        
        # Grup adlarını al
        grup_adlari = df['Grup'].tolist()
        
        # Seçilen ay sayısına göre veriyi filtrele
        date_columns = df.columns[1:]  # İlk sütun 'Grup'
        selected_columns = date_columns[-selected_months:]
        
        # Sadece sayısal değerleri içeren DataFrame oluştur
        numeric_df = df[selected_columns].apply(pd.to_numeric, errors='coerce')
        
        # Korelasyon matrisini hesapla
        correlation_matrix = numeric_df.T.corr()
        
        # Heatmap oluştur
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=grup_adlari,
            y=grup_adlari,
            colorscale=[
                [0, '#EF476F'],      # Negatif - Kırmızı
                [0.5, '#FFFFFF'],    # Sıfır - Beyaz
                [1, '#06D6A0']       # Pozitif - Yeşil
            ],
            zmid=0,
            text=correlation_matrix.values.round(3),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(
                title="Korelasyon<br>Katsayısı",
                titleside="right",
                tickmode="linear",
                tick0=-1,
                dtick=0.5,
                thickness=15,
                len=0.7
            ),
            hovertemplate='%{x} ↔ %{y}<br>Korelasyon: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': f'{analysis_type.replace("_", " ").title()} Korelasyon Analizi<br><sub>Son {selected_months} Ay</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Poppins, sans-serif', 'color': '#1F2937'}
            },
            xaxis_title="",
            yaxis_title="",
            height=700 if analysis_type == 'ana_gruplar' else 1000,
            font=dict(family="Inter, sans-serif", size=11),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis={'side': 'bottom', 'tickangle': -45},
            yaxis={'side': 'left'},
            margin=dict(l=150, r=50, t=100, b=150)
        )
        
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # En yüksek ve en düşük korelasyonları bul (kendisiyle korelasyon hariç)
        correlation_pairs = []
        for i in range(len(grup_adlari)):
            for j in range(i+1, len(grup_adlari)):
                correlation_pairs.append({
                    'grup1': grup_adlari[i],
                    'grup2': grup_adlari[j],
                    'korelasyon': correlation_matrix.iloc[i, j]
                })
        
        # Sırala
        correlation_pairs_sorted = sorted(correlation_pairs, key=lambda x: abs(x['korelasyon']), reverse=True)
        top_correlations = correlation_pairs_sorted[:5]
        
        # Pozitif ve negatif korelasyonları ayır
        positive_corr = sorted([x for x in correlation_pairs if x['korelasyon'] > 0], 
                               key=lambda x: x['korelasyon'], reverse=True)[:5]
        negative_corr = sorted([x for x in correlation_pairs if x['korelasyon'] < 0], 
                               key=lambda x: x['korelasyon'])[:5]
        
        # Mevcut tarih aralığını hesapla
        date_range = f"{selected_columns[0][:7]} - {selected_columns[-1][:7]}"
        
        return render_template('korelasyon_analizi.html',
                             graphJSON=graphJSON,
                             selected_months=selected_months,
                             analysis_type=analysis_type,
                             top_correlations=top_correlations,
                             positive_correlations=positive_corr,
                             negative_correlations=negative_corr,
                             date_range=date_range,
                             active_page='korelasyon')
    
    except Exception as e:
        print(f"Korelasyon analizi hatası: {str(e)}")
        import traceback
        print(traceback.format_exc())
        flash("Korelasyon analizi yüklenirken bir hata oluştu.", "error")
        return redirect(url_for('ana_sayfa'))

# Service Worker route
@app.route('/sw.js')
def service_worker():
    return send_file('static/sw.js', mimetype='application/javascript')

# Get VAPID public key
@app.route('/api/push/vapid-public-key', methods=['GET'])
def get_vapid_public_key():
    if not VAPID_PUBLIC_KEY:
        return jsonify({'error': 'VAPID public key not configured'}), 500
    return jsonify({'publicKey': VAPID_PUBLIC_KEY})

# Subscribe to push notifications
@app.route('/api/push/subscribe', methods=['POST'])
def subscribe_push():
    try:
        subscription_data = request.get_json()
        
        if not subscription_data or 'endpoint' not in subscription_data:
            return jsonify({'error': 'Invalid subscription data'}), 400
        
        endpoint = subscription_data.get('endpoint')
        keys = subscription_data.get('keys', {})
        p256dh = keys.get('p256dh', '')
        auth = keys.get('auth', '')
        user_agent = request.headers.get('User-Agent', '')
        
        # Save subscription to database
        conn = sqlite3.connect('push_subscriptions.db')
        c = conn.cursor()
        try:
            c.execute('''
                INSERT OR REPLACE INTO subscriptions (endpoint, p256dh, auth, user_agent)
                VALUES (?, ?, ?, ?)
            ''', (endpoint, p256dh, auth, user_agent))
            conn.commit()
            conn.close()
            return jsonify({'success': True, 'message': 'Abonelik başarıyla kaydedildi'}), 200
        except sqlite3.Error as e:
            conn.close()
            print(f"Database error: {str(e)}")
            return jsonify({'error': 'Database error'}), 500
    
    except Exception as e:
        print(f"Subscribe error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Unsubscribe from push notifications
@app.route('/api/push/unsubscribe', methods=['POST'])
def unsubscribe_push():
    try:
        subscription_data = request.get_json()
        endpoint = subscription_data.get('endpoint')
        
        if not endpoint:
            return jsonify({'error': 'Endpoint required'}), 400
        
        # Remove subscription from database
        conn = sqlite3.connect('push_subscriptions.db')
        c = conn.cursor()
        c.execute('DELETE FROM subscriptions WHERE endpoint = ?', (endpoint,))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Abonelik iptal edildi'}), 200
    
    except Exception as e:
        print(f"Unsubscribe error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Admin panel for sending push notifications (simple HTML form)
@app.route('/admin/push', methods=['GET', 'POST'])
def admin_push_panel():
    # Simple API key authentication
    admin_api_key = os.environ.get('ADMIN_API_KEY', '')
    
    if request.method == 'GET':
        return render_template('admin_push.html', active_page='admin', has_api_key=bool(admin_api_key))
    
    if request.method == 'POST':
        # Check API key if set
        if admin_api_key:
            provided_key = request.form.get('api_key', '')
            if provided_key != admin_api_key:
                flash('❌ Geçersiz API key!', 'error')
                return redirect(url_for('admin_push_panel'))
        
        # Get notification data from form
        title = request.form.get('title', '').strip()
        body = request.form.get('body', '').strip()
        url = request.form.get('url', '/bultenler').strip()
        
        if not title or not body:
            flash('❌ Başlık ve mesaj gereklidir!', 'error')
            return redirect(url_for('admin_push_panel'))
        
        # Ensure URL starts with / if it's a relative URL
        if url and not url.startswith('/') and not url.startswith('http://') and not url.startswith('https://'):
            url = '/' + url
        
        # Prepare notification data
        notification_data = {
            'title': title,
            'body': body,
            'url': url
        }
        
        # Send notification using internal function
        try:
            # Call the send function directly
            result = send_push_notification_internal(notification_data)
            
            if result.get('success'):
                sent = result.get('sent', 0)
                failed = result.get('failed', 0)
                flash(f"✅ Bildirim gönderildi! {sent} kullanıcıya ulaştı. ({failed} başarısız)", 'success')
            else:
                flash(f"❌ Hata: {result.get('error', 'Bilinmeyen hata')}", 'error')
        except Exception as e:
            flash(f'❌ Hata: {str(e)}', 'error')
        
        return redirect(url_for('admin_push_panel'))

# Internal function to send push notifications
def send_push_notification_internal(notification_data):
    """Internal function to send push notifications"""
    try:
        title = notification_data.get('title', 'Web TÜFE')
        body = notification_data.get('body', 'Yeni bülten yayınlandı!')
        url = notification_data.get('url', '/bultenler')
        icon = notification_data.get('icon', '/static/icon-192x192.png')
        
        if not VAPID_PUBLIC_KEY or not VAPID_PRIVATE_KEY:
            return {'success': False, 'error': 'VAPID keys not configured'}
        
        # Get all subscriptions
        conn = sqlite3.connect('push_subscriptions.db')
        c = conn.cursor()
        c.execute('SELECT endpoint, p256dh, auth FROM subscriptions')
        subscriptions = c.fetchall()
        conn.close()
        
        if not subscriptions:
            return jsonify({'message': 'No subscriptions found', 'sent': 0}), 200
        
        vapid_claims = {
            "sub": VAPID_CLAIM_EMAIL
        }
        
        success_count = 0
        fail_count = 0
        
        for endpoint, p256dh, auth in subscriptions:
            try:
                subscription_info = {
                    "endpoint": endpoint,
                    "keys": {
                        "p256dh": p256dh,
                        "auth": auth
                    }
                }
                
                # Generate unique tag for each notification (prevents browser from replacing previous notification)
                # Use timestamp + endpoint suffix to ensure uniqueness
                unique_tag = f"web-tufe-{int(datetime.now().timestamp() * 1000)}-{endpoint[-10:]}"
                
                payload = json.dumps({
                    "title": title,
                    "body": body,
                    "icon": icon,
                    "url": url,
                    "tag": unique_tag
                })
                
                # Get VAPID private key in base64 URL-safe format (pywebpush expects this)
                vapid_private_key = get_vapid_private_key_for_webpush()
                if not vapid_private_key:
                    raise Exception("VAPID private key not configured or invalid")
                
                webpush(
                    subscription_info=subscription_info,
                    data=payload,
                    vapid_private_key=vapid_private_key,
                    vapid_claims=vapid_claims
                )
                success_count += 1
            except WebPushException as e:
                print(f"WebPush error for endpoint {endpoint[:50]}...: {str(e)}")
                # Remove invalid subscription
                if e.response and e.response.status_code == 410:
                    print(f"Removing invalid subscription (410 Gone): {endpoint[:50]}...")
                    conn = sqlite3.connect('push_subscriptions.db')
                    c = conn.cursor()
                    c.execute('DELETE FROM subscriptions WHERE endpoint = ?', (endpoint,))
                    conn.commit()
                    conn.close()
                fail_count += 1
            except Exception as e:
                print(f"Error sending push to {endpoint[:50]}...: {str(e)}")
                import traceback
                print(traceback.format_exc())
                fail_count += 1
        
        return {
            'success': True,
            'sent': success_count,
            'failed': fail_count,
            'total': len(subscriptions)
        }
    
    except Exception as e:
        print(f"Send push error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {'success': False, 'error': str(e)}

# Send push notification to all subscribers (API endpoint)
@app.route('/api/push/send', methods=['POST'])
def send_push_notification():
    try:
        # Check API key if set (optional security)
        admin_api_key = os.environ.get('ADMIN_API_KEY', '')
        if admin_api_key:
            provided_key = request.headers.get('X-API-Key') or (request.get_json() or {}).get('api_key')
            if provided_key and provided_key != admin_api_key:
                return jsonify({'error': 'Unauthorized'}), 401
        
        notification_data = request.get_json() or {}
        result = send_push_notification_internal(notification_data)
        
        if result.get('success'):
            return jsonify(result), 200
        else:
            return jsonify(result), 500
    
    except Exception as e:
        print(f"Send push error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

# Get subscription count (admin endpoint)
@app.route('/api/push/subscription-count', methods=['GET'])
def get_subscription_count():
    try:
        conn = sqlite3.connect('push_subscriptions.db')
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM subscriptions')
        count = c.fetchone()[0]
        conn.close()
        return jsonify({'count': count}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 