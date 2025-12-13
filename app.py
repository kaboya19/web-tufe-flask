from flask import Flask, render_template, request, send_file, flash, redirect, url_for, jsonify
from flask_caching import Cache
from flask_compress import Compress
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
import io
from gspread.exceptions import APIError, SpreadsheetNotFound
import base64
from oauth2client.service_account import ServiceAccountCredentials
import gspread
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from pywebpush import webpush, WebPushException
import sqlite3
from cryptography.hazmat.primitives import serialization
import base64
import re
from dotenv import load_dotenv

# Load environment variables from .env file (only if not already set)
# This allows Render/production environment variables to take priority
load_dotenv(override=False)

app = Flask(__name__)
app.secret_key = os.urandom(24).hex()  # Güvenli, rastgele bir secret key oluştur

# Compression configuration - Compress responses to reduce bandwidth
Compress(app)

# Global error handlers
@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors gracefully"""
    print(f"❌ Internal Server Error: {error}")
    import traceback
    traceback.print_exc()
    return jsonify({
        'error': 'Internal server error',
        'message': 'Sunucuda bir hata oluştu. Lütfen daha sonra tekrar deneyin.'
    }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Not found',
        'message': 'Aradığınız sayfa bulunamadı.'
    }), 404

@app.errorhandler(503)
def service_unavailable(error):
    """Handle 503 errors (service unavailable)"""
    return jsonify({
        'error': 'Service unavailable',
        'message': 'Servis şu anda kullanılamıyor. Lütfen birkaç dakika sonra tekrar deneyin.'
    }), 503

# Cache configuration - Memory cache for better performance
cache_config = {
    'CACHE_TYPE': 'SimpleCache',  # In-memory cache (works well for single-instance deployments)
    'CACHE_DEFAULT_TIMEOUT': 600,  # 10 minutes default cache timeout (increased for better performance)
    'CACHE_THRESHOLD': 2000  # Maximum number of items in cache (increased)
}
cache = Cache(app, config=cache_config)

# Helper function to cache CSV reads
@cache.memoize(timeout=600)  # 10 minutes cache timeout
def cached_read_csv(filepath, **kwargs):
    """Cached version of pd.read_csv to avoid repeated file I/O"""
    return pd.read_csv(filepath, **kwargs)

# Make get_turkish_month available in templates
@app.template_global()
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

# Push Subscriptions Storage
# Use Google Sheets for persistent storage (survives deploys)
# Fallback to SQLite for local development

def get_push_subscriptions_sheet():
    """Get Google Sheets worksheet for push subscriptions"""
    try:
        # Use the same credentials and spreadsheet as email subscriptions
        # This ensures we have proper permissions
        creds = get_google_credentials()
        client = gspread.authorize(creds)
        
        # Use the same spreadsheet URL as email subscriptions (it has proper permissions)
        sheet_url = "https://docs.google.com/spreadsheets/d/1Y3SpFSsASfCzrM7iM-j_x5XR5pYv__8etC4ptaA9dio"
        try:
            spreadsheet = client.open_by_url(sheet_url)
        except Exception as e:
            print(f"Error opening spreadsheet by URL: {str(e)}")
            # Try by ID as fallback
            spreadsheet_id = "1Y3SpFSsASfCzrM7iM-j_x5XR5pYv__8etC4ptaA9dio"
            spreadsheet = client.open_by_key(spreadsheet_id)
        
        # Try to get the worksheet, create if it doesn't exist
        try:
            worksheet = spreadsheet.worksheet('Push Subscriptions')
            # Check if headers exist, if not add them
            try:
                headers = worksheet.row_values(1)
                if not headers or len(headers) == 0 or headers[0] != 'Endpoint':
                    # Clear and add headers
                    worksheet.clear()
                    worksheet.append_row(['Endpoint', 'P256DH', 'Auth', 'User Agent', 'Created At'])
            except Exception as e:
                print(f"Warning: Could not check headers: {str(e)}")
                # Headers might be empty, add them
                worksheet.append_row(['Endpoint', 'P256DH', 'Auth', 'User Agent', 'Created At'])
        except gspread.exceptions.WorksheetNotFound:
            # Create new worksheet
            try:
                worksheet = spreadsheet.add_worksheet(title='Push Subscriptions', rows=1000, cols=5)
                # Add headers
                worksheet.append_row(['Endpoint', 'P256DH', 'Auth', 'User Agent', 'Created At'])
            except Exception as e:
                print(f"Error creating worksheet: {str(e)}")
                return None
        
        return worksheet
    except Exception as e:
        print(f"Error accessing Google Sheets for push subscriptions: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def init_push_db():
    """Initialize push subscriptions storage - lazy initialization"""
    # Just initialize SQLite for fallback
    # Google Sheets will be initialized on first use
    try:
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
    except Exception as e:
        print(f"Error initializing SQLite: {str(e)}")

# Initialize SQLite on startup (Google Sheets will be initialized on first use)
init_push_db()




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

@cache.memoize(timeout=600)  # Cache for 10 minutes
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

    df=cached_read_csv("gruplaraylık.csv",index_col=0)
    
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

@cache.memoize(timeout=600)  # Cache for 10 minutes
def get_tufe_data():
    # Google Sheets API setup
    """creds = get_google_credentials_2()
    client = gspread.authorize(creds)
    
    spreadsheet = client.open_by_key('14iiu_MQwtMxHTFt6ceyFhkk6v0OL-wuoQS1IGPzSpNE')
    
    worksheet = spreadsheet.worksheet('Web TÜFE')
    
    data = worksheet.get_all_values()"""

    df=cached_read_csv('tüfe.csv').rename(columns={"Unnamed: 0":"Tarih"})
    
    # Convert to DataFrame
    #df = pd.DataFrame(data[1:], columns=data[0])
    
    # Convert date column to datetime
    df['Tarih'] = pd.to_datetime(df['Tarih'])
    
    # Sort by date
    df = df.sort_values('Tarih')
    
    # Convert TÜFE values to float
    
    return df

@cache.memoize(timeout=600)  # Cache for 10 minutes
def get_monthly_change():
    # Google Sheets API setup
    """creds = get_google_credentials_2()
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key('14iiu_MQwtMxHTFt6ceyFhkk6v0OL-wuoQS1IGPzSpNE')
    worksheet = spreadsheet.get_worksheet_by_id(767776936)
    data = worksheet.get_all_values()
    df = pd.DataFrame(data[1:], columns=data[0])"""

    df=cached_read_csv("gruplaraylık.csv",index_col=0)
   
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

@cache.memoize(timeout=600)  # Cache for 10 minutes
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
    tuik_df = cached_read_csv('tüik.csv', index_col=0)
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

@cache.memoize(timeout=600)  # Cache for 10 minutes
def get_ana_gruplar_data():
    # Google Sheets API setup
    """creds = get_google_credentials_2()
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key('14iiu_MQwtMxHTFt6ceyFhkk6v0OL-wuoQS1IGPzSpNE')
    worksheet = spreadsheet.get_worksheet_by_id(564638736)
    data = worksheet.get_all_values()
    df = pd.DataFrame(data[1:], columns=data[0])"""

    df=cached_read_csv("gruplar_int.csv").rename(columns={"Unnamed: 0":"Tarih"})
    df['Tarih'] = pd.to_datetime(df['Tarih'])
    df = df.sort_values('Tarih')
    return df

@cache.memoize(timeout=600)  # Cache for 10 minutes
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

@cache.memoize(timeout=600)  # Cache for 10 minutes
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
            hoverlabel=dict(
                bgcolor='white',
                font_size=12,
                font_family='Inter, sans-serif',
                namelength=-1
            ),
            height=600,
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

# Health check endpoint for Render monitoring
@app.route('/health')
def health_check():
    """Health check endpoint for Render and monitoring services"""
    try:
        # Quick check: try to read a small CSV file to verify file system access
        test_df = cached_read_csv('tüfe.csv', nrows=1)
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'cache_status': 'active'
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 503

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

@cache.memoize(timeout=600)  # Cache for 10 minutes
def get_monthly_group_data_for_date(date_str):
    """Get monthly group data for a specific date"""
    try:
        df = cached_read_csv("gruplaraylık.csv", index_col=0)
        
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

@cache.memoize(timeout=600)  # Cache for 10 minutes
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

@cache.memoize(timeout=600)  # Cache for 10 minutes
def get_top_movers():
    # Daily Data
    top_risers = []
    top_fallers = []
    try:
        endeksler_df = cached_read_csv('endeksler.csv', index_col=0)
        # Calculate pct change
        pct_changes = endeksler_df.pct_change().iloc[-1] * 100
        pct_changes = pct_changes.dropna()
        pct_changes_sorted = pct_changes.sort_values()
        
        top_fallers = [(index, value) for index, value in pct_changes_sorted.head(20).items()]
        top_risers = [(index, value) for index, value in pct_changes_sorted.tail(20).iloc[::-1].items()]
        
    except Exception as e:
        print(f"Error in get_top_movers (daily endeksler): {e}")

    top_harcama_risers = []
    top_harcama_fallers = []
    try:
        harcama_df = cached_read_csv('harcama_grupları.csv', index_col=0)
        # Calculate pct change
        harcama_pct_changes = harcama_df.pct_change().iloc[-1] * 100
        harcama_pct_changes = harcama_pct_changes.dropna()
        harcama_pct_changes_sorted = harcama_pct_changes.sort_values()
        
        top_harcama_fallers = [(index, value) for index, value in harcama_pct_changes_sorted.head(20).items()]
        top_harcama_risers = [(index, value) for index, value in harcama_pct_changes_sorted.tail(20).iloc[::-1].items()]
        
    except Exception as e:
        print(f"Error in get_top_movers (daily harcama): {e}")
        
    # Monthly Data
    monthly_top_risers = []
    monthly_top_fallers = []
    try:
        maddeler_monthly = cached_read_csv('maddeleraylık.csv')
        if 'Madde' in maddeler_monthly.columns:
            maddeler_monthly = maddeler_monthly.set_index('Madde')
            # Select the last column (assuming it is the latest date)
            # The first column in CSV is often index (Unnamed: 0), we set Madde as index, 
            # so we should check columns carefully. 
            # The columns will be: Unnamed: 0, Date1, Date2...
            # We want the last column.
            monthly_changes = maddeler_monthly.iloc[:, -1]
            monthly_changes = monthly_changes.dropna()
            monthly_changes_sorted = monthly_changes.sort_values()
            
            monthly_top_fallers = [(index, value) for index, value in monthly_changes_sorted.head(20).items()]
            monthly_top_risers = [(index, value) for index, value in monthly_changes_sorted.tail(20).iloc[::-1].items()]
    except Exception as e:
        print(f"Error in get_top_movers (monthly maddeler): {e}")

    monthly_top_harcama_risers = []
    monthly_top_harcama_fallers = []
    try:
        harcama_monthly = cached_read_csv('harcama_gruplarıaylık.csv')
        if 'Grup' in harcama_monthly.columns:
            harcama_monthly = harcama_monthly.set_index('Grup')
            monthly_harcama_changes = harcama_monthly.iloc[:, -1]
            monthly_harcama_changes = monthly_harcama_changes.dropna()
            monthly_harcama_changes_sorted = monthly_harcama_changes.sort_values()
            
            monthly_top_harcama_fallers = [(index, value) for index, value in monthly_harcama_changes_sorted.head(20).items()]
            monthly_top_harcama_risers = [(index, value) for index, value in monthly_harcama_changes_sorted.tail(20).iloc[::-1].items()]
    except Exception as e:
        print(f"Error in get_top_movers (monthly harcama): {e}")

    return (top_risers, top_fallers, top_harcama_risers, top_harcama_fallers,
            monthly_top_risers, monthly_top_fallers, monthly_top_harcama_risers, monthly_top_harcama_fallers)

@app.route('/ana-sayfa', methods=['GET', 'POST'])
def ana_sayfa():
    # Get Top Movers
    (top_risers, top_fallers, top_harcama_risers, top_harcama_fallers,
     monthly_top_risers, monthly_top_fallers, monthly_top_harcama_risers, monthly_top_harcama_fallers) = get_top_movers()
    
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
                margin=dict(l=40, r=20, t=30, b=50), hovermode='closest',
                hoverlabel=dict(bgcolor='white', font_size=12, font_family='Inter, sans-serif', font_color='#2B2D42'),
                autosize=True
            )
            graphJSON = json.dumps(single_fig, cls=plotly.utils.PlotlyJSONEncoder)
            # Get monthly change data for data view (from gruplaraylık.csv)
            try:
                monthly_df = pd.read_csv("gruplaraylık.csv", index_col=0)
                # Transpose: groups become columns, dates become rows
                group_names = monthly_df.iloc[:, 0].tolist()
                date_columns = monthly_df.columns[1:].tolist()
                
                # Create transposed dataframe
                transposed_data = []
                for date_col in date_columns:
                    row_data = {'Tarih': date_col}
                    for idx, group_name in enumerate(group_names):
                        value = monthly_df.iloc[idx][date_col]
                        try:
                            value = float(str(value).replace(',', '.'))
                        except:
                            value = None
                        row_data[group_name] = value
                    transposed_data.append(row_data)
                
                time_series_df = pd.DataFrame(transposed_data)
                time_series_df = time_series_df.sort_values('Tarih', ascending=False)  # Most recent first
                time_series_data = time_series_df.to_dict('records')
                time_series_columns = time_series_df.columns.tolist()
            except Exception as e:
                print(f"Error loading monthly data: {e}")
                time_series_data = []
                time_series_columns = []
            
            return render_template('index.html', 
                                 graphJSON=graphJSON, 
                                 active_page='ana_sayfa', 
                                 last_update=last_update,
                                 available_dates=available_dates,
                                 selected_date=selected_date,
                                 sorted_group_data=data_pairs_sorted,
                                 show_contrib=show_contrib,
                                 chart_title=chart_title,
                                 top_risers=top_risers,
                                 top_fallers=top_fallers,
                                 top_harcama_risers=top_harcama_risers,
                                 top_harcama_fallers=top_harcama_fallers,
                                 time_series_data=time_series_data,
                                 time_series_columns=time_series_columns)

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
            margin=dict(l=40, r=20, t=30, b=50), hovermode='closest',
            hoverlabel=dict(bgcolor='white', font_size=12, font_family='Inter, sans-serif', font_color='#2B2D42'),
            autosize=True
        )

        # Export combined
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Get monthly change data for data view (from gruplaraylık.csv)
        try:
            monthly_df = cached_read_csv("gruplaraylık.csv", index_col=0)
            # Transpose: groups become columns, dates become rows
            # First column is group names, rest are dates
            group_names = monthly_df.iloc[:, 0].tolist()
            date_columns = monthly_df.columns[1:].tolist()
            
            # Create transposed dataframe
            transposed_data = []
            for date_col in date_columns:
                row_data = {'Tarih': date_col}
                for idx, group_name in enumerate(group_names):
                    value = monthly_df.iloc[idx][date_col]
                    try:
                        value = float(str(value).replace(',', '.'))
                    except:
                        value = None
                    row_data[group_name] = value
                transposed_data.append(row_data)
            
            time_series_df = pd.DataFrame(transposed_data)
            time_series_df = time_series_df.sort_values('Tarih', ascending=False)  # Most recent first
            time_series_data = time_series_df.to_dict('records')
            time_series_columns = time_series_df.columns.tolist()
        except Exception as e:
            print(f"Error loading monthly data: {e}")
            time_series_data = []
            time_series_columns = []
        
        return render_template('index.html', 
                             graphJSON=graphJSON, 
                             active_page='ana_sayfa', 
                             last_update=last_update,
                             available_dates=available_dates,
                             selected_date=selected_date,
                             sorted_group_data=data_pairs_sorted,
                             show_contrib=show_contrib,
                                 chart_title=chart_title,
                                 top_risers=top_risers,
                                 top_fallers=top_fallers,
                                 top_harcama_risers=top_harcama_risers,
                                 top_harcama_fallers=top_harcama_fallers,
                                 monthly_top_risers=monthly_top_risers,
                                 monthly_top_fallers=monthly_top_fallers,
                                 monthly_top_harcama_risers=monthly_top_harcama_risers,
                                 monthly_top_harcama_fallers=monthly_top_harcama_fallers,
                                 time_series_data=time_series_data,
                                 time_series_columns=time_series_columns)
    except Exception as e:
        flash(f'Bir hata oluştu: {str(e)}', 'error')
        available_dates = get_available_dates()
        # Get monthly change data for data view even on error
        try:
            monthly_df = pd.read_csv("gruplaraylık.csv", index_col=0)
            # Transpose: groups become columns, dates become rows
            group_names = monthly_df.iloc[:, 0].tolist()
            date_columns = monthly_df.columns[1:].tolist()
            
            # Create transposed dataframe
            transposed_data = []
            for date_col in date_columns:
                row_data = {'Tarih': date_col}
                for idx, group_name in enumerate(group_names):
                    value = monthly_df.iloc[idx][date_col]
                    try:
                        value = float(str(value).replace(',', '.'))
                    except:
                        value = None
                    row_data[group_name] = value
                transposed_data.append(row_data)
            
            time_series_df = pd.DataFrame(transposed_data)
            time_series_df = time_series_df.sort_values('Tarih', ascending=False)  # Most recent first
            time_series_data = time_series_df.to_dict('records')
            time_series_columns = time_series_df.columns.tolist()
        except Exception as e:
            print(f"Error loading monthly data: {e}")
            time_series_data = []
            time_series_columns = []
        return render_template('index.html', 
                             available_dates=available_dates,
                             selected_date=None,
                             sorted_group_data=[],
                             show_contrib=True,
                             top_risers=top_risers,
                             top_fallers=top_fallers,
                             top_harcama_risers=top_harcama_risers,
                             top_harcama_fallers=top_harcama_fallers,
                             monthly_top_risers=monthly_top_risers,
                             monthly_top_fallers=monthly_top_fallers,
                             monthly_top_harcama_risers=monthly_top_harcama_risers,
                             monthly_top_harcama_fallers=monthly_top_harcama_fallers,
                             time_series_data=time_series_data,
                             time_series_columns=time_series_columns)

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
            hovermode='x unified',
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
            hovermode='x unified',
            hoverlabel=dict(
                bgcolor='white',
                font_size=12,
                font_family='Inter, sans-serif',
                namelength=-1
            ),
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
        # Get endeksler.csv data for data view
        try:
            endeksler_df = cached_read_csv('endeksler.csv', index_col=0)
            endeksler_df.index = pd.to_datetime(endeksler_df.index)
            endeksler_df = endeksler_df.sort_index(ascending=False)  # Most recent first
            # Reset index to make Tarih a column
            endeksler_df = endeksler_df.reset_index()
            endeksler_df.rename(columns={endeksler_df.columns[0]: 'Tarih'}, inplace=True)
            # Format Tarih column as YYYY-MM-DD string
            endeksler_df['Tarih'] = endeksler_df['Tarih'].dt.strftime('%Y-%m-%d')
            # Convert to dict for template
            endeks_data = endeksler_df.to_dict('records')
            endeks_columns = endeksler_df.columns.tolist()
        except Exception as e:
            print(f"Error loading endeksler data: {e}")
            endeks_data = []
            endeks_columns = []
        
        # Get maddeleraylık.csv data for monthly change data view
        try:
            # Read CSV: first column is index (0,1,2...), second column is 'Madde', rest are dates
            maddeler_monthly_df = cached_read_csv('maddeleraylık.csv', index_col=0)
            # After index_col=0, first column (index) is removed, so columns are: ['Madde', '2025-02-28', ...]
            # Get 'Madde' column values
            madde_names_list = maddeler_monthly_df['Madde'].tolist()
            # Get date columns (all columns except 'Madde')
            date_columns_monthly = [col for col in maddeler_monthly_df.columns if col != 'Madde']
            
            # Create transposed dataframe: dates as rows, maddeler as columns
            transposed_monthly_data = []
            for date_col in date_columns_monthly:
                row_data = {'Tarih': date_col}
                for idx, madde_name in enumerate(madde_names_list):
                    value = maddeler_monthly_df.iloc[idx][date_col]
                    try:
                        value = float(str(value).replace(',', '.'))
                    except:
                        value = None
                    row_data[madde_name] = value
                transposed_monthly_data.append(row_data)
            
            maddeler_monthly_transposed_df = pd.DataFrame(transposed_monthly_data)
            # Format Tarih column as YYYY-MM-DD string
            try:
                maddeler_monthly_transposed_df['Tarih'] = pd.to_datetime(maddeler_monthly_transposed_df['Tarih']).dt.strftime('%Y-%m-%d')
            except:
                pass
            maddeler_monthly_transposed_df = maddeler_monthly_transposed_df.sort_values('Tarih', ascending=False)  # Most recent first
            maddeler_monthly_data = maddeler_monthly_transposed_df.to_dict('records')
            maddeler_monthly_columns = maddeler_monthly_transposed_df.columns.tolist()
        except Exception as e:
            print(f"Error loading maddeleraylık data: {e}")
            import traceback
            traceback.print_exc()
            maddeler_monthly_data = []
            maddeler_monthly_columns = []
        
        return render_template('tufe.html', graphJSON=graphJSON,
            last_date=last_date,
            change_rate=change_rate,
            month_name=month_name,
            monthly_change=monthly_change,
            bar_graphJSON=bar_graphJSON,
            line_graphJSON=line_graphJSON,
            active_page='tufe',
            madde_names=madde_names,
            selected_madde=selected_madde,
            endeks_data=endeks_data,
            endeks_columns=endeks_columns,
            maddeler_monthly_data=maddeler_monthly_data,
            maddeler_monthly_columns=maddeler_monthly_columns
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
            df_monthly=cached_read_csv("maddeleraylık.csv",index_col=0)
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
                hovermode='x unified',
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
                hovermode='x unified',
                hoverlabel=dict(
                    bgcolor='white',
                    font_size=12,
                    font_family='Inter, sans-serif',
                    namelength=-1
                )
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
            
            # Get endeksler.csv data for data view
            try:
                endeksler_df = cached_read_csv('endeksler.csv', index_col=0)
                endeksler_df.index = pd.to_datetime(endeksler_df.index)
                endeksler_df = endeksler_df.sort_index(ascending=False)  # Most recent first
                # Reset index to make Tarih a column
                endeksler_df = endeksler_df.reset_index()
                endeksler_df.rename(columns={endeksler_df.columns[0]: 'Tarih'}, inplace=True)
                # Format Tarih column as YYYY-MM-DD string
                endeksler_df['Tarih'] = endeksler_df['Tarih'].dt.strftime('%Y-%m-%d')
                # Convert to dict for template
                endeks_data = endeksler_df.to_dict('records')
                endeks_columns = endeksler_df.columns.tolist()
            except Exception as e:
                print(f"Error loading endeksler data: {e}")
                endeks_data = []
                endeks_columns = []
            
            # Get maddeleraylık.csv data for monthly change data view
            try:
                maddeler_monthly_df = cached_read_csv('maddeleraylık.csv', index_col=0)
                # After index_col=0, columns are: ['Madde', '2025-02-28', ...]
                # Get 'Madde' column values
                madde_names_list = maddeler_monthly_df['Madde'].tolist()
                # Get date columns (all columns except 'Madde')
                date_columns_monthly = [col for col in maddeler_monthly_df.columns if col != 'Madde']
                
                # Create transposed dataframe
                transposed_monthly_data = []
                for date_col in date_columns_monthly:
                    row_data = {'Tarih': date_col}
                    for idx, madde_name in enumerate(madde_names_list):
                        value = maddeler_monthly_df.iloc[idx][date_col]
                        try:
                            value = float(str(value).replace(',', '.'))
                        except:
                            value = None
                        row_data[madde_name] = value
                    transposed_monthly_data.append(row_data)
                
                maddeler_monthly_transposed_df = pd.DataFrame(transposed_monthly_data)
                # Format Tarih column as YYYY-MM-DD string
                try:
                    maddeler_monthly_transposed_df['Tarih'] = pd.to_datetime(maddeler_monthly_transposed_df['Tarih']).dt.strftime('%Y-%m-%d')
                except:
                    pass
                maddeler_monthly_transposed_df = maddeler_monthly_transposed_df.sort_values('Tarih', ascending=False)  # Most recent first
                maddeler_monthly_data = maddeler_monthly_transposed_df.to_dict('records')
                maddeler_monthly_columns = maddeler_monthly_transposed_df.columns.tolist()
            except Exception as e:
                print(f"Error loading maddeleraylık data: {e}")
                maddeler_monthly_data = []
                maddeler_monthly_columns = []
            
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
                no_data=False,
                endeks_data=endeks_data,
                endeks_columns=endeks_columns,
                maddeler_monthly_data=maddeler_monthly_data,
                maddeler_monthly_columns=maddeler_monthly_columns
            )
        else:
            # Get endeksler.csv data for data view
            try:
                endeksler_df = cached_read_csv('endeksler.csv', index_col=0)
                endeksler_df.index = pd.to_datetime(endeksler_df.index)
                endeksler_df = endeksler_df.sort_index(ascending=False)  # Most recent first
                # Reset index to make Tarih a column
                endeksler_df = endeksler_df.reset_index()
                endeksler_df.rename(columns={endeksler_df.columns[0]: 'Tarih'}, inplace=True)
                # Format Tarih column as YYYY-MM-DD string
                endeksler_df['Tarih'] = endeksler_df['Tarih'].dt.strftime('%Y-%m-%d')
                # Convert to dict for template
                endeks_data = endeksler_df.to_dict('records')
                endeks_columns = endeksler_df.columns.tolist()
            except Exception as e:
                print(f"Error loading endeksler data: {e}")
                endeks_data = []
                endeks_columns = []
            
            # Get maddeleraylık.csv data for monthly change data view even on error
            try:
                maddeler_monthly_df = cached_read_csv('maddeleraylık.csv', index_col=0)
                # After index_col=0, columns are: ['Madde', '2025-02-28', ...]
                # Get 'Madde' column values
                madde_names_list = maddeler_monthly_df['Madde'].tolist()
                # Get date columns (all columns except 'Madde')
                date_columns_monthly = [col for col in maddeler_monthly_df.columns if col != 'Madde']
                
                # Create transposed dataframe
                transposed_monthly_data = []
                for date_col in date_columns_monthly:
                    row_data = {'Tarih': date_col}
                    for idx, madde_name in enumerate(madde_names_list):
                        value = maddeler_monthly_df.iloc[idx][date_col]
                        try:
                            value = float(str(value).replace(',', '.'))
                        except:
                            value = None
                        row_data[madde_name] = value
                    transposed_monthly_data.append(row_data)
                
                maddeler_monthly_transposed_df = pd.DataFrame(transposed_monthly_data)
                # Format Tarih column as YYYY-MM-DD string
                try:
                    maddeler_monthly_transposed_df['Tarih'] = pd.to_datetime(maddeler_monthly_transposed_df['Tarih']).dt.strftime('%Y-%m-%d')
                except:
                    pass
                maddeler_monthly_transposed_df = maddeler_monthly_transposed_df.sort_values('Tarih', ascending=False)  # Most recent first
                maddeler_monthly_data = maddeler_monthly_transposed_df.to_dict('records')
                maddeler_monthly_columns = maddeler_monthly_transposed_df.columns.tolist()
            except Exception as e:
                print(f"Error loading maddeleraylık data: {e}")
                maddeler_monthly_data = []
                maddeler_monthly_columns = []
            
            return render_template('tufe.html',
                graphJSON=None,
                last_date=None,
                change_rate=None,
                month_name=None,
                monthly_change=None,
                bar_graphJSON=None,
                line_graphJSON=None,
                active_page='tufe',
                madde_names=madde_names,
                selected_madde=selected_madde,
                aylik_degisim_graphJSON=None,
                no_data=True,
                endeks_data=endeks_data,
                endeks_columns=endeks_columns,
                maddeler_monthly_data=maddeler_monthly_data,
                maddeler_monthly_columns=maddeler_monthly_columns
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
        name='Web TÜFE',
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
                name='TÜİK',
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
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='white',
            font_size=12,
            font_family='Inter, sans-serif',
            namelength=-1
        )
    )
    line_graphJSON = line_fig.to_json()

    # Get gruplar_int.csv data for index data view
    gruplar_data = []
    gruplar_columns = []
    try:
        gruplar_df = cached_read_csv('gruplar_int.csv', index_col=0)
        gruplar_df.index = pd.to_datetime(gruplar_df.index)
        gruplar_df = gruplar_df.sort_index(ascending=False)
        gruplar_df.index = gruplar_df.index.strftime('%Y-%m-%d')
        gruplar_data = gruplar_df.reset_index().to_dict('records')
        gruplar_columns = ['Tarih'] + [col for col in gruplar_df.columns]
        # Rename index column to Tarih in records
        for row in gruplar_data:
            if 'index' in row:
                row['Tarih'] = row.pop('index')
    except Exception as e:
        print(f"Error loading gruplar_int.csv data: {e}")
        import traceback
        traceback.print_exc()
        gruplar_data = []
        gruplar_columns = []

    # Get gruplaraylık.csv data for monthly change data view
    gruplar_monthly_data = []
    gruplar_monthly_columns = []
    try:
        # Read CSV: first column is index (0,1,2...), second column is 'Grup', rest are dates
        gruplar_monthly_df = cached_read_csv('gruplaraylık.csv', index_col=0)
        # After index_col=0, first column (index) is removed, so columns are: ['Grup', '2025-02-28', ...]
        # Get 'Grup' column values
        grup_names_list = gruplar_monthly_df['Grup'].tolist()
        # Get date columns (all columns except 'Grup')
        date_columns_monthly = [col for col in gruplar_monthly_df.columns if col != 'Grup']
        
        # Create transposed dataframe: dates as rows, gruplar as columns
        transposed_monthly_data = []
        for date_col in date_columns_monthly:
            row_data = {'Tarih': date_col}
            for idx, grup_name in enumerate(grup_names_list):
                value = gruplar_monthly_df.iloc[idx][date_col]
                try:
                    value = float(str(value).replace(',', '.'))
                except:
                    value = None
                row_data[grup_name] = value
            transposed_monthly_data.append(row_data)
        
        gruplar_monthly_transposed_df = pd.DataFrame(transposed_monthly_data)
        # Format Tarih column as YYYY-MM-DD string
        try:
            gruplar_monthly_transposed_df['Tarih'] = pd.to_datetime(gruplar_monthly_transposed_df['Tarih']).dt.strftime('%Y-%m-%d')
        except:
            pass
        gruplar_monthly_transposed_df = gruplar_monthly_transposed_df.sort_values('Tarih', ascending=False)
        gruplar_monthly_data = gruplar_monthly_transposed_df.to_dict('records')
        gruplar_monthly_columns = gruplar_monthly_transposed_df.columns.tolist()
    except Exception as e:
        print(f"Error loading gruplaraylık.csv data: {e}")
        import traceback
        traceback.print_exc()
        gruplar_monthly_data = []
        gruplar_monthly_columns = []

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
        line_graphJSON=line_graphJSON,
        gruplar_data=gruplar_data,
        gruplar_columns=gruplar_columns,
        gruplar_monthly_data=gruplar_monthly_data,
        gruplar_monthly_columns=gruplar_monthly_columns
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
    # Kırılım seviyesi (5: Harcama grubu, 4: Dörtlü, 3: Üçlü)
    breakdown_level = request.form.get('breakdown_level', '5') if request.method == 'POST' else '5'
    print("Seçilen grup:", selected_group)
    print("Seçilen tarih:", selected_date)
    print("Kırılım seviyesi:", breakdown_level)
    
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
    
    # Kırılım seviyesine göre verileri filtrele
    try:
        urunler_detay = pd.read_csv('harcamaürünleri1.csv')
        urunler_detay['Ana Grup'] = urunler_detay['Ana Grup'].str.strip().str.lower()
        urunler_detay['Grup'] = urunler_detay['Grup'].str.strip().str.lower()
        urunler_detay['Üçlü'] = urunler_detay['Üçlü'].str.strip().str.lower()
        urunler_detay['Dörtlü'] = urunler_detay['Dörtlü'].str.strip().str.lower()
        
        # Seçilen ana gruba göre filtrele
        filtered_urunler = urunler_detay[urunler_detay["Ana Grup"] == selected_group_norm]
        
        # Kırılım seviyesine göre liste oluştur
        if breakdown_level == '5':
            # Harcama grupları (mevcut mantık)
            breakdown_items = filtered_urunler["Grup"].unique().tolist()
        elif breakdown_level == '4':
            # Dörtlü endeksler
            breakdown_items = filtered_urunler["Dörtlü"].unique().tolist()
        elif breakdown_level == '3':
            # Üçlü endeksler
            breakdown_items = filtered_urunler["Üçlü"].unique().tolist()
        else:
            breakdown_items = harcama_gruplari
    except Exception as e:
        print("harcamaürünleri1.csv okuma hatası:", e)
        breakdown_items = harcama_gruplari if breakdown_level == '5' else []

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
    # Kırılım seviyesine göre veri kaynağını seç
    if breakdown_level == '5':
        # Harcama grupları (mevcut mantık)
        df_data = df_harcama
        df_data[df_data.columns[0]] = df_data[df_data.columns[0]].str.strip().str.lower()
        data_items = breakdown_items
    elif breakdown_level == '4':
        # Dörtlü endeksler
        df_data = pd.read_csv("dörtlüleraylık.csv", index_col=0)
        df_data[df_data.columns[0]] = df_data[df_data.columns[0]].str.strip().str.lower()
        data_items = breakdown_items
    elif breakdown_level == '3':
        # Üçlü endeksler
        df_data = pd.read_csv("üçlüleraylık.csv", index_col=0)
        df_data[df_data.columns[0]] = df_data[df_data.columns[0]].str.strip().str.lower()
        data_items = breakdown_items
    else:
        df_data = df_harcama
        df_data[df_data.columns[0]] = df_data[df_data.columns[0]].str.strip().str.lower()
        data_items = harcama_gruplari
    
    # Seçilen kırılım seviyesine göre değerleri al
    bar_labels = []
    bar_values = []
    bar_colors = []
    for item in data_items:
        item_norm = item.strip().lower()
        row = df_data[df_data.iloc[:,0] == item_norm]
        if not row.empty:
            try:
                value = float(str(row[sheet_date].values[0]).replace(',', '.'))
                bar_labels.append(item.title())
                bar_values.append(value)
                bar_colors.append('#EF476F' if item_norm == selected_group_norm else '#118AB2')
            except Exception as e:
                print(f'Item {item} için değer alınırken hata:', e)
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
        # For values near zero, place text on opposite side to avoid overlap with group names
        if abs(value) <= 0.001:
            # Near zero: place inside bar
            text_x = value
            align_anchor = 'center'
        elif abs(value) < 0.1:  # Threshold for overlap detection - place on opposite side
            # For values near zero, place on opposite side
            if value >= 0:
                # Positive but small: place on left (opposite)
                text_x = value - text_offset
                align_anchor = 'right'
            else:
                # Negative but small: place on right (opposite)
                text_x = value + text_offset
                align_anchor = 'left'
        else:
            # Normal values: place on normal side
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
            # Select correct CSV by contribution type and breakdown level
            if breakdown_level == '3':
                # Üçlü endeksler
                file_path = 'üçlümanşetkatkılar.csv' if contrib_type == 'ana' else 'üçlükatkılar.csv'
            elif breakdown_level == '4':
                # Dörtlü endeksler
                file_path = 'dörtlümanşetkatkılar.csv' if contrib_type == 'ana' else 'dörtlükatkılar.csv'
            else:
                # Harcama grupları (breakdown_level == '5')
                file_path = 'harcamagrupları_manşetkatkı.csv' if contrib_type == 'ana' else 'harcamagrupları_katkı.csv'
            df_katki = pd.read_csv(file_path, index_col=0)
            # choose date row
            target_date = sheet_date if sheet_date in df_katki.index else (sheet_date[:7] if sheet_date[:7] in df_katki.index else df_katki.index[-1])
            row = df_katki.loc[target_date]
            # normalize headers
            row.index = [str(c).strip().lower() for c in row.index]
            labels = []
            values = []
            # Kırılım seviyesine göre doğru öğe listesini kullan
            items_to_check = breakdown_items if breakdown_level in ['3', '4'] else harcama_gruplari
            for g in items_to_check:
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

            # Ranges function - define before use
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
            
            # Build text labels for left chart - use annotations to place text on opposite side when near zero to avoid overlap
            left_range = compute_range_left(left_values)
            left_range_span = left_range[1] - left_range[0]
            left_text_offset = left_range_span * 0.01
            
            # Prepare text and textposition lists
            left_text = []
            left_textposition = []
            left_annotations_to_add = []
            
            for i, (cat, val) in enumerate(zip(left_categories, left_values)):
                if abs(val) <= 0.001:
                    # Near zero: place inside
                    left_text.append(f"<b>{val:+.2f}%</b>")
                    left_textposition.append('inside')
                elif abs(val) < 0.1:  # Threshold for overlap detection - place on opposite side
                    # For values near zero, place on opposite side using annotation
                    if val >= 0:
                        # Positive but small: place on left (opposite)
                        text_x = val - left_text_offset
                        xanchor = 'right'
                    else:
                        # Negative but small: place on right (opposite)
                        text_x = val + left_text_offset
                        xanchor = 'left'
                    left_annotations_to_add.append({
                        'x': text_x, 'y': cat,
                        'text': f"<b>{val:+.2f}%</b>",
                        'xanchor': xanchor, 'yanchor': 'middle'
                    })
                    # Don't show text on bar for these values
                    left_text.append('')
                    left_textposition.append('outside')
                else:
                    # Normal values: show text outside
                    left_text.append(f"<b>{val:+.2f}%</b>")
                    left_textposition.append('outside')
            
            comb.add_trace(go.Bar(
                y=left_categories, x=left_values, orientation='h',
                marker=dict(color=left_colors, line=dict(width=0)),
                name='Aylık değişim',
                text=left_text,
                textposition=left_textposition, textfont=dict(size=15, family='Inter, sans-serif', color='#2B2D42'),
                cliponaxis=False,
                hovertemplate='%{y}: %{x:+.2f}%<extra></extra>'
            ), row=1, col=1)
            
            # Add annotations for left chart - place text on opposite side for values near zero
            for ann in left_annotations_to_add:
                comb.add_annotation(
                    x=ann['x'], y=ann['y'],
                    text=ann['text'],
                    showarrow=False,
                    font=dict(size=15, family='Inter, sans-serif', color='#2B2D42'),
                    xanchor=ann['xanchor'], yanchor=ann['yanchor'],
                    row=1, col=1
                )

            # Build text labels safely; place zero values inside bar, leave empty when value is None (ana grup satırı)
            right_range_vals = [v for v in right_values if v is not None]
            if right_range_vals:
                right_range_min = min(right_range_vals)
                right_range_max = max(right_range_vals)
                right_range_span = right_range_max - right_range_min if right_range_max != right_range_min else abs(right_range_max) * 0.1
                right_text_offset = right_range_span * 0.01
            else:
                right_range_span = 1
                right_text_offset = 0.01
            
            # Prepare text and textposition lists for right chart
            right_text = []
            right_textposition = []
            right_annotations_to_add = []
            
            for i, (cat, val) in enumerate(zip(left_categories, right_values)):
                if val is None:
                    # Ana grup satırı: no text
                    right_text.append('')
                    right_textposition.append('inside')
                elif abs(val) <= 0.001:
                    # Near zero: place inside
                    right_text.append(f"<b>{val:+.2f}</b>")
                    right_textposition.append('inside')
                elif abs(val) < 0.01:  # Threshold for overlap detection - place on opposite side
                    # For values near zero, place on opposite side using annotation
                    if val >= 0:
                        # Positive but small: place on left (opposite)
                        text_x = val - right_text_offset
                        xanchor = 'right'
                    else:
                        # Negative but small: place on right (opposite)
                        text_x = val + right_text_offset
                        xanchor = 'left'
                    right_annotations_to_add.append({
                        'x': text_x, 'y': cat,
                        'text': f"<b>{val:+.2f}</b>",
                        'xanchor': xanchor, 'yanchor': 'middle'
                    })
                    # Don't show text on bar for these values
                    right_text.append('')
                    right_textposition.append('outside')
                else:
                    # Normal values: show text outside
                    right_text.append(f"<b>{val:+.2f}</b>")
                    right_textposition.append('outside')
            
            comb.add_trace(go.Bar(
                y=left_categories, x=right_values, orientation='h',
                marker=dict(color='#118AB2', line=dict(width=0)),
                name='Katkı',
                text=right_text,
                textposition=right_textposition, textfont=dict(size=15, family='Inter, sans-serif', color='#2B2D42'),
                cliponaxis=False,
                hovertemplate='%{y}: %{x:+.2f} puan<extra></extra>'
            ), row=1, col=2)
            
            # Add annotations for right chart - place text on opposite side for values near zero
            for ann in right_annotations_to_add:
                comb.add_annotation(
                    x=ann['x'], y=ann['y'],
                    text=ann['text'],
                    showarrow=False,
                    font=dict(size=15, family='Inter, sans-serif', color='#2B2D42'),
                    xanchor=ann['xanchor'], yanchor=ann['yanchor'],
                    row=1, col=2
                )

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
    # Kırılım seviyesine göre seçilebilir öğeleri belirle
    if breakdown_level == '5':
        harcama_grubu_adlari = harcama_gruplari if harcama_gruplari else []
    elif breakdown_level == '4':
        harcama_grubu_adlari = breakdown_items if breakdown_items else []
    elif breakdown_level == '3':
        harcama_grubu_adlari = breakdown_items if breakdown_items else []
    else:
        harcama_grubu_adlari = harcama_gruplari if harcama_gruplari else []
    harcama_grubu_endeks_graphJSON = None
    harcama_grubu_total_change = None
    harcama_grubu_monthly_change = None

    # Endeks grafiği için harcama grubu seçildiyse veriyi oku ve çiz
    if selected_harcama_grubu:
        toplam_baslik=""
        son_ay=""
        try:
            # Kırılım seviyesine göre endeks veri kaynağını seç
            if breakdown_level == '5':
                df_endeks=cached_read_csv("harcama_grupları.csv").rename(columns={"Unnamed: 0":"Tarih"})
            elif breakdown_level == '4':
                df_endeks=pd.read_csv("dörtlüler.csv").rename(columns={"Unnamed: 0":"Tarih"})
            elif breakdown_level == '3':
                df_endeks=pd.read_csv("üçlüler.csv").rename(columns={"Unnamed: 0":"Tarih"})
            else:
                df_endeks=cached_read_csv("harcama_grupları.csv").rename(columns={"Unnamed: 0":"Tarih"})
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
                
                # --- Fix: Son ay değişimi ve ay ismi kırılım seviyesine göre alınacak ---
                # Kırılım seviyesine göre aylık değişim veri kaynağını seç
                if breakdown_level == '5':
                    df_harcama_aylik=pd.read_csv("harcama_gruplarıaylık.csv",index_col=0)
                elif breakdown_level == '4':
                    df_harcama_aylik=pd.read_csv("dörtlüleraylık.csv",index_col=0)
                elif breakdown_level == '3':
                    df_harcama_aylik=pd.read_csv("üçlüleraylık.csv",index_col=0)
                else:
                    df_harcama_aylik=pd.read_csv("harcama_gruplarıaylık.csv",index_col=0)
                df_harcama_aylik[df_harcama_aylik.columns[0]] = df_harcama_aylik[df_harcama_aylik.columns[0]].str.strip().str.lower()
                row = df_harcama_aylik[df_harcama_aylik.iloc[:,0] == selected_norm]
                harcama_grubu_monthly_change = None
                son_ay = None
                if not row.empty:
                    last_col = df_harcama_aylik.columns[-1]
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
                    name='Web TÜFE',
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
                            name='TÜİK',
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
                    # Kırılım seviyesine göre aylık değişim veri kaynağını seç
                    if breakdown_level == '5':
                        df_harcama_monthly = pd.read_csv("harcama_gruplarıaylık.csv", index_col=0)
                    elif breakdown_level == '4':
                        df_harcama_monthly = pd.read_csv("dörtlüleraylık.csv", index_col=0)
                    elif breakdown_level == '3':
                        df_harcama_monthly = pd.read_csv("üçlüleraylık.csv", index_col=0)
                    else:
                        df_harcama_monthly = pd.read_csv("harcama_gruplarıaylık.csv", index_col=0)
                    df_harcama_monthly[df_harcama_monthly.columns[0]] = df_harcama_monthly[df_harcama_monthly.columns[0]].str.strip().str.lower()
                    row = df_harcama_monthly[df_harcama_monthly.iloc[:,0] == selected_norm]
                    
                    if not row.empty:
                        # Get all monthly changes
                        monthly_changes = []
                        monthly_dates = []
                        for col in df_harcama_monthly.columns[1:]:
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
                                hovermode='x unified',
                                hoverlabel=dict(
                                    bgcolor='white',
                                    font_size=12,
                                    font_family='Inter, sans-serif',
                                    namelength=-1
                                )
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

    # Get endeks data for index data view (only if harcama grubu selected)
    harcama_endeks_data = []
    harcama_endeks_columns = []
    if selected_harcama_grubu:
        try:
            # Kırılım seviyesine göre endeks veri kaynağını seç
            if breakdown_level == '5':
                harcama_endeks_df = cached_read_csv('harcama_grupları.csv').rename(columns={"Unnamed: 0":"Tarih"})
            elif breakdown_level == '4':
                harcama_endeks_df = pd.read_csv('dörtlüler.csv').rename(columns={"Unnamed: 0":"Tarih"})
            elif breakdown_level == '3':
                harcama_endeks_df = pd.read_csv('üçlüler.csv').rename(columns={"Unnamed: 0":"Tarih"})
            else:
                harcama_endeks_df = cached_read_csv('harcama_grupları.csv').rename(columns={"Unnamed: 0":"Tarih"})
            harcama_endeks_df['Tarih'] = pd.to_datetime(harcama_endeks_df['Tarih'])
            harcama_endeks_df = harcama_endeks_df.sort_values('Tarih', ascending=False)
            harcama_endeks_df['Tarih'] = harcama_endeks_df['Tarih'].dt.strftime('%Y-%m-%d')
            harcama_endeks_data = harcama_endeks_df.to_dict('records')
            harcama_endeks_columns = harcama_endeks_df.columns.tolist()
        except Exception as e:
            print(f"Error loading harcama_grupları.csv data: {e}")
            import traceback
            traceback.print_exc()
            harcama_endeks_data = []
            harcama_endeks_columns = []

    # Get harcama_gruplarıaylık.csv data for monthly change data view (always prepare, not just when harcama grubu selected)
    harcama_monthly_data = []
    harcama_monthly_columns = []
    try:
        # Read CSV: first column is index (0,1,2...), second column is 'Grup', rest are dates
        harcama_monthly_df = cached_read_csv('harcama_gruplarıaylık.csv', index_col=0)
        # After index_col=0, first column (index) is removed, so columns are: ['Grup', '2025-02-28', ...]
        # Get 'Grup' column values
        harcama_grup_names_list = harcama_monthly_df['Grup'].tolist()
        # Get date columns (all columns except 'Grup')
        date_columns_harcama = [col for col in harcama_monthly_df.columns if col != 'Grup']
        
        # Create transposed dataframe: dates as rows, harcama grupları as columns
        transposed_harcama_data = []
        for date_col in date_columns_harcama:
            row_data = {'Tarih': date_col}
            for idx, harcama_grup_name in enumerate(harcama_grup_names_list):
                value = harcama_monthly_df.iloc[idx][date_col]
                try:
                    value = float(str(value).replace(',', '.'))
                except:
                    value = None
                row_data[harcama_grup_name] = value
            transposed_harcama_data.append(row_data)
        
        harcama_monthly_transposed_df = pd.DataFrame(transposed_harcama_data)
        # Format Tarih column as YYYY-MM-DD string
        try:
            harcama_monthly_transposed_df['Tarih'] = pd.to_datetime(harcama_monthly_transposed_df['Tarih']).dt.strftime('%Y-%m-%d')
        except:
            pass
        harcama_monthly_transposed_df = harcama_monthly_transposed_df.sort_values('Tarih', ascending=False)
        harcama_monthly_data = harcama_monthly_transposed_df.to_dict('records')
        harcama_monthly_columns = harcama_monthly_transposed_df.columns.tolist()
    except Exception as e:
        print(f"Error loading harcama_gruplarıaylık.csv data: {e}")
        import traceback
        traceback.print_exc()
        harcama_monthly_data = []
        harcama_monthly_columns = []

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
        combined_graphJSON=combined_graphJSON,
        harcama_endeks_data=harcama_endeks_data,
        harcama_endeks_columns=harcama_endeks_columns,
        harcama_monthly_data=harcama_monthly_data,
        harcama_monthly_columns=harcama_monthly_columns,
        breakdown_level=breakdown_level
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
    df_madde=cached_read_csv("maddeleraylık.csv",index_col=0)
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

    df=cached_read_csv("özelgöstergeler.csv").rename(columns={"Unnamed: 0":"Tarih"})
    
    # Convert date column to datetime
    df['Tarih'] = pd.to_datetime(df['Tarih'])
    
    # Get indicator names (column names except 'Tarih')
    indicator_names = [col for col in df.columns if col != 'Tarih']
    
    # Handle selected indicator and view type
    if request.method == 'POST':
        selected_indicator = request.form.get('indicator', indicator_names[0])
        view_type = request.form.get('view_type', 'graph')
        selected_current_month = request.form.get('current_month')
        selected_previous_month = request.form.get('previous_month')
    else:
        selected_indicator = indicator_names[0]
        view_type = request.args.get('view_type', 'graph')
        selected_current_month = request.args.get('current_month')
        selected_previous_month = request.args.get('previous_month')
    
    if not view_type:
        view_type = 'graph'
    
    # Get monthly change data
    """monthly_worksheet = spreadsheet.get_worksheet_by_id(1767722805)
    monthly_data = monthly_worksheet.get_all_values()
    df_monthly = pd.DataFrame(monthly_data[1:], columns=monthly_data[0])"""
    df_monthly_raw = cached_read_csv("özelgöstergeleraylık.csv", index_col=0)
    df_monthly_norm = df_monthly_raw.copy()
    first_column_name = df_monthly_norm.columns[0] if not df_monthly_norm.empty else None
    month_columns = list(df_monthly_norm.columns[1:]) if not df_monthly_norm.empty else []
    
    if first_column_name and not df_monthly_norm.empty:
        df_monthly_norm[first_column_name] = (
            df_monthly_norm[first_column_name]
            .astype(str)
            .str.strip()
            .str.lower()
        )
    
    selected_indicator_norm = selected_indicator.strip().lower()
    
    # Get monthly change for selected indicator
    monthly_change = None
    if not df_monthly_norm.empty and first_column_name:
        indicator_row = df_monthly_norm[df_monthly_norm.iloc[:, 0] == selected_indicator_norm]
        if not indicator_row.empty:
            try:
                monthly_change = float(str(indicator_row.iloc[:, -1].values[0]).replace(',', '.'))
            except Exception:
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
            name='Web TÜFE',
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
                    name='TÜİK',
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
            margin=dict(l=20, r=20, t=80, b=20),
            hovermode='x unified',
            autosize=True,
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
    monthly_row = pd.DataFrame()
    if not df_monthly_norm.empty and first_column_name:
        monthly_row = df_monthly_norm[df_monthly_norm.iloc[:, 0] == selected_indicator_norm]

    # Aylık değişim verilerini hazırla
    monthly_changes = []
    monthly_dates = []
    if not monthly_row.empty:
        for col in df_monthly_norm.columns[1:]:
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
        margin=dict(l=20, r=20, t=80, b=20),
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
            tickangle=0,
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
            gridcolor='#E9ECEF'
            ),
        showlegend=True,
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
        ),
        autosize=True
    )

    bar_graphJSON = json.dumps(bar_fig, cls=plotly.utils.PlotlyJSONEncoder)
    line_graphJSON = json.dumps(line_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Get the last month from the monthly CSV file and prepare table data
    last_month_from_csv = None
    previous_month_label = None
    current_month_label = None
    table_rows = []
    
    def format_month_label(column_name):
        try:
            date_obj = datetime.strptime(column_name, '%Y-%m-%d')
            month_name = get_turkish_month(date_obj.strftime('%Y-%m-%d'))
            short_year = date_obj.strftime('%y')
            return f"{month_name} {short_year}"
        except Exception:
            return column_name
    
    def parse_numeric(value):
        try:
            return float(str(value).replace(',', '.'))
        except Exception:
            return None
    
    def get_ytd_value(series_name):
        try:
            if series_name in df.columns:
                return float(df[series_name].iloc[-1]) - 100
            if tuik_endeks_df is not None and series_name in tuik_endeks_df.columns:
                return float(tuik_endeks_df.iloc[-1][series_name]) - 100
        except Exception:
            return None
        return None
    
    if not df_monthly_raw.empty:
        date_columns = month_columns
        month_options = [{"value": col, "label": format_month_label(col)} for col in date_columns]
        
        default_current_col = date_columns[-1] if date_columns else None
        default_previous_col = date_columns[-2] if len(date_columns) >= 2 else None
        
        selected_current_col = selected_current_month if selected_current_month in date_columns else default_current_col
        selected_previous_col = selected_previous_month if selected_previous_month in date_columns else default_previous_col
        
        if selected_current_col is None and date_columns:
            selected_current_col = date_columns[-1]
        if selected_previous_col is None:
            selected_previous_col = default_previous_col if default_previous_col else selected_current_col
        
        if selected_current_col:
            current_month_label = format_month_label(selected_current_col)
        if selected_previous_col:
            previous_month_label = format_month_label(selected_previous_col)
        
        if default_current_col:
            try:
                date_obj = datetime.strptime(default_current_col, '%Y-%m-%d')
                last_month_from_csv = get_turkish_month(date_obj.strftime('%Y-%m-%d'))
            except Exception:
                last_month_from_csv = None
        
        if first_column_name and selected_current_col:
            table_values = {}
            for _, row in df_monthly_raw.iterrows():
                group_raw = row[first_column_name]
                if pd.isna(group_raw):
                    continue
                group_name = str(group_raw).strip()
                if not group_name or group_name.lower() == 'nan':
                    continue
                
                previous_value = parse_numeric(row.get(selected_previous_col)) if selected_previous_col else None
                current_value = parse_numeric(row.get(selected_current_col))
                ytd_value = get_ytd_value(group_name)
                
                table_values[group_name] = {
                    'previous': previous_value,
                    'current': current_value,
                    'ytd': ytd_value
                }
        
            table_layout = [
                {"label": "Web TÜFE", "source": "Web TÜFE", "indent_px": 0, "is_total": True},
                {"label": "1. Mallar", "source": "Mallar", "indent_px": 0, "is_section": True},
                {"label": "Enerji", "source": "Enerji", "indent_px": 24},
                {"label": "Gıda ve Alkolsüz İçecekler", "source": "Gıda ve alkolsüz içecekler", "indent_px": 24},
                {"label": "İşlenmemiş Gıda", "source": "İşlenmemiş gıda", "indent_px": 40},
                {"label": "Taze Meyve-Sebze", "source": "Taze meyve ve sebze", "indent_px": 56},
                {"label": "Diğer İşlenmemiş Gıda", "source": "Diğer işlenmemiş gıda", "indent_px": 56},
                {"label": "İşlenmiş Gıda", "source": "İşlenmiş Gıda", "indent_px": 40},
                {"label": "Ekmek ve Tahıllar", "source": "Ekmek ve tahıllar", "indent_px": 56},
                {"label": "Diğer İşlenmiş", "source": "Diğer işlenmiş gıda", "indent_px": 56},
                {"label": "Enerji ve Gıda Dışı Mallar", "source": "Enerji ve gıda dışı mallar", "indent_px": 24},
                {"label": "Temel Mallar", "source": "Temel mallar", "indent_px": 40},
                {"label": "Dayanıklı Mallar (Altın Hariç)", "source": "Dayanıklı Mallar (altın hariç)", "indent_px": 56},
                {"label": "Giyim ve Ayakkabı", "source": "Giyim ve ayakkabı", "indent_px": 56},
                {"label": "Diğer Temel Mallar", "source": "Diğer Temel Mallar", "indent_px": 56},
                {"label": "Alkollü İçecekler, Tütün ve Altın", "source": "Alkollü içecekler, tütün ve altın", "indent_px": 40},
                {"label": "2. Hizmetler", "source": "Hizmet", "indent_px": 0, "is_section": True},
                {"label": "Kira", "source": "Kira", "indent_px": 24},
                {"label": "Lokanta ve Oteller", "source": "Lokanta ve oteller", "indent_px": 24},
                {"label": "Ulaştırma Hizmetleri", "source": "Ulaştırma hizmetleri", "indent_px": 24},
                {"label": "Haberleşme Hizmetleri", "source": "Haberleşme hizmetleri", "indent_px": 24},
                {"label": "Diğer Hizmetler", "source": "Diğer hizmetler", "indent_px": 24},
                {"label": "3. Temel Göstergeler", "source": None, "indent_px": 0, "is_section": True, "is_header_only": True},
                {"label": "B- İşlenmemiş Gıda, Enerji, Alkollü İçecekler, Tütün ve Altın Hariç TÜFE", "source": "TÜFE B", "indent_px": 24},
                {"label": "C- Gıda ve Alkolsüz İçecekler, Enerji, Alkollü İçecekler, Tütün ve Altın Hariç TÜFE", "source": "TÜFE C", "indent_px": 24},
                {"label": "D- İşlenmemiş Gıda, Alkollü İçecekler, Tütün ve Altın Hariç TÜFE", "source": "TÜFE D", "indent_px": 24},
                {"label": "E- Alkollü İçecekler, Tütün ve Altın Hariç TÜFE", "source": "TÜFE E", "indent_px": 24},
                {"label": "F- Yönetilen Yönlendirilen Fiyatlar Hariç TÜFE", "source": "TÜFE F", "indent_px": 24},
            ]
            
            for item in table_layout:
                data = table_values.get(item.get("source"), {}) if item.get("source") else {}
                table_rows.append({
                    "label": item["label"],
                    "previous": data.get("previous"),
                    "current": data.get("current"),
                    "ytd": data.get("ytd"),
                    "indent_px": item.get("indent_px", 0),
                    "is_section": item.get("is_section", False),
                    "is_total": item.get("is_total", False),
                    "is_header_only": item.get("is_header_only", False)
                })
        selected_current_col_for_template = selected_current_col
        selected_previous_col_for_template = selected_previous_col
    else:
        month_options = []
        selected_current_col_for_template = None
        selected_previous_col_for_template = None
    
    # Prepare endeks data for data view (from özelgöstergeler.csv)
    ozel_endeks_data = None
    ozel_endeks_columns = None
    if not df.empty:
        try:
            # Sort by date descending
            ozel_endeks_df = df.copy()
            ozel_endeks_df = ozel_endeks_df.sort_values('Tarih', ascending=False)
            # Format dates
            ozel_endeks_df['Tarih'] = ozel_endeks_df['Tarih'].dt.strftime('%Y-%m-%d')
            ozel_endeks_columns = ['Tarih'] + [col for col in ozel_endeks_df.columns if col != 'Tarih']
            ozel_endeks_data = ozel_endeks_df.to_dict('records')
        except Exception as e:
            print(f"Özel göstergeler endeks veri hazırlama hatası: {e}")
            ozel_endeks_data = None
            ozel_endeks_columns = None
    
    # Prepare monthly data for monthly change graph data view (from özelgöstergeleraylık.csv)
    ozel_monthly_data = None
    ozel_monthly_columns = None
    if not df_monthly_raw.empty:
        try:
            # Transpose: dates as rows, indicators as columns
            ozel_monthly_df = df_monthly_raw.set_index(df_monthly_raw.columns[0]).T
            ozel_monthly_df.index.name = 'Tarih'
            ozel_monthly_df = ozel_monthly_df.reset_index()
            # Sort by date descending
            ozel_monthly_df['Tarih'] = pd.to_datetime(ozel_monthly_df['Tarih'])
            ozel_monthly_df = ozel_monthly_df.sort_values('Tarih', ascending=False)
            # Format dates
            ozel_monthly_df['Tarih'] = ozel_monthly_df['Tarih'].dt.strftime('%Y-%m-%d')
            ozel_monthly_columns = ['Tarih'] + list(ozel_monthly_df.columns[1:])
            ozel_monthly_data = ozel_monthly_df.to_dict('records')
        except Exception as e:
            print(f"Özel göstergeler aylık veri hazırlama hatası: {e}")
            ozel_monthly_data = None
            ozel_monthly_columns = None
    
    return render_template('ozel_kapsamli_gostergeler.html',
    graphJSON=graphJSON,
    indicator_names=indicator_names,
    selected_indicator=selected_indicator,
    total_change=total_change,
    monthly_change=monthly_change,
    active_page='ozel_kapsamli_gostergeler',
    last_date=dates.iloc[-1] if not df.empty else None,
    month_name=last_month_from_csv if last_month_from_csv else (get_turkish_month(dates.iloc[-1].strftime('%Y-%m-%d')) if not df.empty else None),
    view_type=view_type,
    table_rows=table_rows,
    previous_month_label=previous_month_label,
    current_month_label=current_month_label,
    selected_previous_month=selected_previous_col_for_template,
    selected_current_month=selected_current_col_for_template,
    month_options=month_options,
    bar_graphJSON=bar_graphJSON,
    line_graphJSON=line_graphJSON,
    ozel_endeks_data=ozel_endeks_data,
    ozel_endeks_columns=ozel_endeks_columns,
    ozel_monthly_data=ozel_monthly_data,
    ozel_monthly_columns=ozel_monthly_columns
)

@app.route('/download/ozel-kapsamli/csv')
def download_ozel_kapsamli_csv():
    try:
        # Endeks verileri (özelgöstergeler.csv)
        df = pd.read_csv("özelgöstergeler.csv").rename(columns={"Unnamed: 0":"Tarih"})
        df['Tarih'] = pd.to_datetime(df['Tarih'])
        df = df.sort_values('Tarih', ascending=False)
        df['Tarih'] = df['Tarih'].dt.strftime('%Y-%m-%d')
        
        output = io.StringIO()
        df.to_csv(output, index=False, encoding='utf-8-sig')
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8-sig')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='ozel_kapsamli_gostergeler_endeks.csv'
        )
    except Exception as e:
        flash(f'CSV indirme hatası: {str(e)}', 'error')
        return redirect(url_for('ozel_kapsamli_gostergeler'))

@app.route('/download/ozel-kapsamli/xlsx')
def download_ozel_kapsamli_xlsx():
    try:
        # Endeks verileri (özelgöstergeler.csv)
        df = pd.read_csv("özelgöstergeler.csv").rename(columns={"Unnamed: 0":"Tarih"})
        df['Tarih'] = pd.to_datetime(df['Tarih'])
        df = df.sort_values('Tarih', ascending=False)
        df['Tarih'] = df['Tarih'].dt.strftime('%Y-%m-%d')
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Özel Kapsamlı Göstergeler Endeks')
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='ozel_kapsamli_gostergeler_endeks.xlsx'
        )
    except Exception as e:
        flash(f'Excel indirme hatası: {str(e)}', 'error')
        return redirect(url_for('ozel_kapsamli_gostergeler'))

@app.route('/download/ozel-kapsamli-aylik/csv')
def download_ozel_kapsamli_aylik_csv():
    try:
        df_monthly_raw = cached_read_csv("özelgöstergeleraylık.csv", index_col=0)
        
        # Transpose: dates as rows, indicators as columns
        ozel_monthly_df = df_monthly_raw.set_index(df_monthly_raw.columns[0]).T
        ozel_monthly_df.index.name = 'Tarih'
        ozel_monthly_df = ozel_monthly_df.reset_index()
        # Sort by date descending
        ozel_monthly_df['Tarih'] = pd.to_datetime(ozel_monthly_df['Tarih'])
        ozel_monthly_df = ozel_monthly_df.sort_values('Tarih', ascending=False)
        # Format dates
        ozel_monthly_df['Tarih'] = ozel_monthly_df['Tarih'].dt.strftime('%Y-%m-%d')
        
        output = io.StringIO()
        ozel_monthly_df.to_csv(output, index=False, encoding='utf-8-sig')
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8-sig')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='ozel_kapsamli_gostergeler_aylik.csv'
        )
    except Exception as e:
        flash(f'CSV indirme hatası: {str(e)}', 'error')
        return redirect(url_for('ozel_kapsamli_gostergeler'))

@app.route('/download/ozel-kapsamli-aylik/xlsx')
def download_ozel_kapsamli_aylik_xlsx():
    try:
        df_monthly_raw = cached_read_csv("özelgöstergeleraylık.csv", index_col=0)
        
        # Transpose: dates as rows, indicators as columns
        ozel_monthly_df = df_monthly_raw.set_index(df_monthly_raw.columns[0]).T
        ozel_monthly_df.index.name = 'Tarih'
        ozel_monthly_df = ozel_monthly_df.reset_index()
        # Sort by date descending
        ozel_monthly_df['Tarih'] = pd.to_datetime(ozel_monthly_df['Tarih'])
        ozel_monthly_df = ozel_monthly_df.sort_values('Tarih', ascending=False)
        # Format dates
        ozel_monthly_df['Tarih'] = ozel_monthly_df['Tarih'].dt.strftime('%Y-%m-%d')
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            ozel_monthly_df.to_excel(writer, index=False, sheet_name='Özel Kapsamlı Göstergeler Aylık')
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='ozel_kapsamli_gostergeler_aylik.xlsx'
        )
    except Exception as e:
        flash(f'Excel indirme hatası: {str(e)}', 'error')
        return redirect(url_for('ozel_kapsamli_gostergeler'))

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
        margin=dict(l=20, r=20, t=80, b=20),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='white',
            font_size=12,
            font_family='Inter, sans-serif',
            namelength=-1
        ),
        autosize=True
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
    
    # Prepare data for data view (from ma.xlsx)
    ma_data = None
    ma_columns = None
    if not df.empty:
        try:
            # Drop 'Unnamed: 0' column if it exists (index column)
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])
            # Transpose: dates as rows, indicators as columns
            ma_df = df.set_index('Gösterge').T
            ma_df.index.name = 'Tarih'
            ma_df = ma_df.reset_index()
            # Sort by date descending
            ma_df['Tarih'] = pd.to_datetime(ma_df['Tarih'], format='%Y-%m', errors='coerce')
            ma_df = ma_df.sort_values('Tarih', ascending=False)
            # Format dates as YYYY-MM
            ma_df['Tarih'] = ma_df['Tarih'].dt.strftime('%Y-%m')
            # Replace NaN with None for JSON serialization
            ma_df = ma_df.fillna('')
            ma_columns = ['Tarih'] + [col for col in ma_df.columns if col != 'Tarih']
            ma_data = ma_df.to_dict('records')
        except Exception as e:
            print(f"Mevsimsel düzeltilmiş göstergeler veri hazırlama hatası: {e}")
            ma_data = None
            ma_columns = None
    
    return render_template('mevsimsel_duzeltilmis_gostergeler.html',
        graphJSON=graphJSON,
        indicator_names=indicator_names,
        selected_indicator=selected_indicator,
        total_change=total_change,
        monthly_change=monthly_change,
        active_page='mevsimsel_duzeltilmis_gostergeler',
        last_date=last_date,
        month_name=last_month_from_csv,
        ma_data=ma_data,
        ma_columns=ma_columns
    )

@app.route('/download/mevsimsel/csv')
def download_mevsimsel_csv():
    try:
        df = pd.read_excel("ma.xlsx")
        
        # Drop 'Unnamed: 0' column if it exists (index column)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        
        # Transpose: dates as rows, indicators as columns
        ma_df = df.set_index('Gösterge').T
        ma_df.index.name = 'Tarih'
        ma_df = ma_df.reset_index()
        # Sort by date descending
        ma_df['Tarih'] = pd.to_datetime(ma_df['Tarih'], format='%Y-%m', errors='coerce')
        ma_df = ma_df.sort_values('Tarih', ascending=False)
        # Format dates as YYYY-MM
        ma_df['Tarih'] = ma_df['Tarih'].dt.strftime('%Y-%m')
        
        output = io.StringIO()
        ma_df.to_csv(output, index=False, encoding='utf-8-sig')
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8-sig')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='mevsimsel_duzeltilmis_gostergeler.csv'
        )
    except Exception as e:
        flash(f'CSV indirme hatası: {str(e)}', 'error')
        return redirect(url_for('mevsimsel_duzeltilmis_gostergeler'))

@app.route('/download/mevsimsel/xlsx')
def download_mevsimsel_xlsx():
    try:
        df = pd.read_excel("ma.xlsx")
        
        # Drop 'Unnamed: 0' column if it exists (index column)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        
        # Transpose: dates as rows, indicators as columns
        ma_df = df.set_index('Gösterge').T
        ma_df.index.name = 'Tarih'
        ma_df = ma_df.reset_index()
        # Sort by date descending
        ma_df['Tarih'] = pd.to_datetime(ma_df['Tarih'], format='%Y-%m', errors='coerce')
        ma_df = ma_df.sort_values('Tarih', ascending=False)
        # Format dates as YYYY-MM
        ma_df['Tarih'] = ma_df['Tarih'].dt.strftime('%Y-%m')
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            ma_df.to_excel(writer, index=False, sheet_name='Mevsimsel Düzeltilmiş Göstergeler')
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='mevsimsel_duzeltilmis_gostergeler.xlsx'
        )
    except Exception as e:
        flash(f'Excel indirme hatası: {str(e)}', 'error')
        return redirect(url_for('mevsimsel_duzeltilmis_gostergeler'))

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
    
    # Get current year for Web TÜFE reports
    current_year = datetime.now().year
    
    for fname in pdf_files:
        processed = False
        # First, try to match files starting with Turkish month names
        for ay in aylar:
            if fname.startswith(ay):
                y = fname.replace(ay, '').replace('.pdf', '').replace('_', ' ').strip()
                label = f"{ay} {y}"
                date_options.append(label)
                file_map[label] = fname
                processed = True
                break
        
        # If not processed, try to match "Web TÜFE" files
        if not processed and fname.startswith('Web TÜFE'):
            # Extract month from filename like "Web TÜFE Kasım Öncü.pdf"
            for ay in aylar:
                if ay in fname:
                    # Try to extract year from filename, default to current year
                    y = str(current_year)
                    # Look for year pattern in filename (e.g., "2025")
                    year_match = re.search(r'20\d{2}', fname)
                    if year_match:
                        y = year_match.group()
                    # Enflasyon görünümü raporları sadece direkt link ile açılmalı, listede gösterme
                    processed = True
                    break
    
    # Tarihleri yıl ve ay'a göre sıralayalım (en yeni en başta)
    def parse_turkish_date(label):
        try:
            # Handle labels with "(Enflasyon Görünümü)" suffix
            label_clean = label.replace(' (Enflasyon Görünümü)', '')
            parts = label_clean.split()
            if len(parts) >= 2:
                ay = parts[0]
                yil = parts[1]
                ay_map = {a: i+1 for i, a in enumerate(aylar)}
                return int(yil), ay_map.get(ay, 0)
            return (0, 0)
        except:
            return (0, 0)
    date_options.sort(key=parse_turkish_date, reverse=True)
    selected_date = request.form.get('bulten_tarihi') if request.method == 'POST' else (date_options[0] if date_options else None)
    selected_file = file_map[selected_date] if selected_date in file_map else None
    return render_template('bultenler.html', date_options=date_options, selected_date=selected_date, selected_file=selected_file, active_page='bultenler')

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
    return render_template('metodoloji.html', active_page='metodoloji')

@app.route('/download/ana-gruplar/csv')
def download_ana_gruplar_csv():
    try:
        # Get monthly change data (from gruplaraylık.csv)
        monthly_df = pd.read_csv("gruplaraylık.csv", index_col=0)
        # Transpose: groups become columns, dates become rows
        group_names = monthly_df.iloc[:, 0].tolist()
        date_columns = monthly_df.columns[1:].tolist()
        
        # Create transposed dataframe
        transposed_data = []
        for date_col in date_columns:
            row_data = {'Tarih': date_col}
            for idx, group_name in enumerate(group_names):
                value = monthly_df.iloc[idx][date_col]
                try:
                    value = float(str(value).replace(',', '.'))
                except:
                    value = None
                row_data[group_name] = value
            transposed_data.append(row_data)
        
        df = pd.DataFrame(transposed_data)
        df = df.sort_values('Tarih', ascending=False)  # Most recent first
        
        # Create a temporary CSV file
        from io import BytesIO
        output = BytesIO()
        df.to_csv(output, index=False, encoding='utf-8-sig')
        output.seek(0)
        return send_file(output, mimetype='text/csv', as_attachment=True, download_name='ana_gruplar_aylik_degisim.csv')
    except Exception as e:
        flash(f'CSV indirme hatası: {str(e)}', 'error')
        return redirect(url_for('ana_sayfa'))

@app.route('/download/ana-gruplar/xlsx')
def download_ana_gruplar_xlsx():
    try:
        # Get monthly change data (from gruplaraylık.csv)
        monthly_df = pd.read_csv("gruplaraylık.csv", index_col=0)
        # Transpose: groups become columns, dates become rows
        group_names = monthly_df.iloc[:, 0].tolist()
        date_columns = monthly_df.columns[1:].tolist()
        
        # Create transposed dataframe
        transposed_data = []
        for date_col in date_columns:
            row_data = {'Tarih': date_col}
            for idx, group_name in enumerate(group_names):
                value = monthly_df.iloc[idx][date_col]
                try:
                    value = float(str(value).replace(',', '.'))
                except:
                    value = None
                row_data[group_name] = value
            transposed_data.append(row_data)
        
        df = pd.DataFrame(transposed_data)
        df = df.sort_values('Tarih', ascending=False)  # Most recent first
        
        # Create a temporary Excel file
        from io import BytesIO
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Aylık Değişimler')
        output.seek(0)
        return send_file(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 
                        as_attachment=True, download_name='ana_gruplar_aylik_degisim.xlsx')
    except Exception as e:
        flash(f'Excel indirme hatası: {str(e)}', 'error')
        return redirect(url_for('ana_sayfa'))

@app.route('/download/gruplar-endeksler/csv')
def download_gruplar_endeksler_csv():
    try:
        gruplar_df = cached_read_csv('gruplar_int.csv', index_col=0)
        gruplar_df.index = pd.to_datetime(gruplar_df.index)
        gruplar_df = gruplar_df.sort_index(ascending=False)
        gruplar_df.index = gruplar_df.index.strftime('%Y-%m-%d')
        gruplar_df = gruplar_df.reset_index()
        gruplar_df.rename(columns={'index': 'Tarih'}, inplace=True)
        
        output = io.StringIO()
        gruplar_df.to_csv(output, index=False, encoding='utf-8-sig')
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8-sig')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='gruplar_endeksler.csv'
        )
    except Exception as e:
        flash(f'CSV indirme hatası: {str(e)}', 'error')
        return redirect(url_for('ana_gruplar'))

@app.route('/download/gruplar-endeksler/xlsx')
def download_gruplar_endeksler_xlsx():
    try:
        gruplar_df = cached_read_csv('gruplar_int.csv', index_col=0)
        gruplar_df.index = pd.to_datetime(gruplar_df.index)
        gruplar_df = gruplar_df.sort_index(ascending=False)
        gruplar_df.index = gruplar_df.index.strftime('%Y-%m-%d')
        gruplar_df = gruplar_df.reset_index()
        gruplar_df.rename(columns={'index': 'Tarih'}, inplace=True)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            gruplar_df.to_excel(writer, index=False, sheet_name='Gruplar Endeksler')
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='gruplar_endeksler.xlsx'
        )
    except Exception as e:
        flash(f'Excel indirme hatası: {str(e)}', 'error')
        return redirect(url_for('ana_gruplar'))

@app.route('/download/gruplar-aylik/csv')
def download_gruplar_aylik_csv():
    try:
        gruplar_monthly_df = cached_read_csv('gruplaraylık.csv', index_col=0)
        grup_names_list = gruplar_monthly_df['Grup'].tolist()
        date_columns_monthly = [col for col in gruplar_monthly_df.columns if col != 'Grup']
        
        transposed_monthly_data = []
        for date_col in date_columns_monthly:
            row_data = {'Tarih': date_col}
            for idx, grup_name in enumerate(grup_names_list):
                value = gruplar_monthly_df.iloc[idx][date_col]
                try:
                    value = float(str(value).replace(',', '.'))
                except:
                    value = None
                row_data[grup_name] = value
            transposed_monthly_data.append(row_data)
        
        df = pd.DataFrame(transposed_monthly_data)
        df['Tarih'] = pd.to_datetime(df['Tarih']).dt.strftime('%Y-%m-%d')
        df = df.sort_values('Tarih', ascending=False)
        
        output = io.StringIO()
        df.to_csv(output, index=False, encoding='utf-8-sig')
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8-sig')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='gruplar_aylik_degisim.csv'
        )
    except Exception as e:
        flash(f'CSV indirme hatası: {str(e)}', 'error')
        return redirect(url_for('ana_gruplar'))

@app.route('/download/harcama-endeksler/csv')
def download_harcama_endeksler_csv():
    try:
        harcama_endeks_df = cached_read_csv('harcama_grupları.csv').rename(columns={"Unnamed: 0":"Tarih"})
        harcama_endeks_df['Tarih'] = pd.to_datetime(harcama_endeks_df['Tarih'])
        harcama_endeks_df = harcama_endeks_df.sort_values('Tarih', ascending=False)
        harcama_endeks_df['Tarih'] = harcama_endeks_df['Tarih'].dt.strftime('%Y-%m-%d')
        
        output = io.StringIO()
        harcama_endeks_df.to_csv(output, index=False, encoding='utf-8-sig')
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8-sig')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='harcama_endeksler.csv'
        )
    except Exception as e:
        flash(f'CSV indirme hatası: {str(e)}', 'error')
        return redirect(url_for('harcama_gruplari'))

@app.route('/download/harcama-endeksler/xlsx')
def download_harcama_endeksler_xlsx():
    try:
        harcama_endeks_df = cached_read_csv('harcama_grupları.csv').rename(columns={"Unnamed: 0":"Tarih"})
        harcama_endeks_df['Tarih'] = pd.to_datetime(harcama_endeks_df['Tarih'])
        harcama_endeks_df = harcama_endeks_df.sort_values('Tarih', ascending=False)
        harcama_endeks_df['Tarih'] = harcama_endeks_df['Tarih'].dt.strftime('%Y-%m-%d')
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            harcama_endeks_df.to_excel(writer, index=False, sheet_name='Harcama Grupları Endeksler')
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='harcama_endeksler.xlsx'
        )
    except Exception as e:
        flash(f'Excel indirme hatası: {str(e)}', 'error')
        return redirect(url_for('harcama_gruplari'))

@app.route('/download/harcama-aylik/csv')
def download_harcama_aylik_csv():
    try:
        harcama_monthly_df = cached_read_csv('harcama_gruplarıaylık.csv', index_col=0)
        harcama_grup_names_list = harcama_monthly_df['Grup'].tolist()
        date_columns_harcama = [col for col in harcama_monthly_df.columns if col != 'Grup']
        
        transposed_harcama_data = []
        for date_col in date_columns_harcama:
            row_data = {'Tarih': date_col}
            for idx, harcama_grup_name in enumerate(harcama_grup_names_list):
                value = harcama_monthly_df.iloc[idx][date_col]
                try:
                    value = float(str(value).replace(',', '.'))
                except:
                    value = None
                row_data[harcama_grup_name] = value
            transposed_harcama_data.append(row_data)
        
        df = pd.DataFrame(transposed_harcama_data)
        df['Tarih'] = pd.to_datetime(df['Tarih']).dt.strftime('%Y-%m-%d')
        df = df.sort_values('Tarih', ascending=False)
        
        output = io.StringIO()
        df.to_csv(output, index=False, encoding='utf-8-sig')
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8-sig')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='harcama_aylik_degisim.csv'
        )
    except Exception as e:
        flash(f'CSV indirme hatası: {str(e)}', 'error')
        return redirect(url_for('harcama_gruplari'))

@app.route('/download/harcama-aylik/xlsx')
def download_harcama_aylik_xlsx():
    try:
        harcama_monthly_df = cached_read_csv('harcama_gruplarıaylık.csv', index_col=0)
        harcama_grup_names_list = harcama_monthly_df['Grup'].tolist()
        date_columns_harcama = [col for col in harcama_monthly_df.columns if col != 'Grup']
        
        transposed_harcama_data = []
        for date_col in date_columns_harcama:
            row_data = {'Tarih': date_col}
            for idx, harcama_grup_name in enumerate(harcama_grup_names_list):
                value = harcama_monthly_df.iloc[idx][date_col]
                try:
                    value = float(str(value).replace(',', '.'))
                except:
                    value = None
                row_data[harcama_grup_name] = value
            transposed_harcama_data.append(row_data)
        
        df = pd.DataFrame(transposed_harcama_data)
        df['Tarih'] = pd.to_datetime(df['Tarih']).dt.strftime('%Y-%m-%d')
        df = df.sort_values('Tarih', ascending=False)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Harcama Grupları Aylık Değişim')
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='harcama_aylik_degisim.xlsx'
        )
    except Exception as e:
        flash(f'Excel indirme hatası: {str(e)}', 'error')
        return redirect(url_for('harcama_gruplari'))

@app.route('/download/gruplar-aylik/xlsx')
def download_gruplar_aylik_xlsx():
    try:
        gruplar_monthly_df = cached_read_csv('gruplaraylık.csv', index_col=0)
        grup_names_list = gruplar_monthly_df['Grup'].tolist()
        date_columns_monthly = [col for col in gruplar_monthly_df.columns if col != 'Grup']
        
        transposed_monthly_data = []
        for date_col in date_columns_monthly:
            row_data = {'Tarih': date_col}
            for idx, grup_name in enumerate(grup_names_list):
                value = gruplar_monthly_df.iloc[idx][date_col]
                try:
                    value = float(str(value).replace(',', '.'))
                except:
                    value = None
                row_data[grup_name] = value
            transposed_monthly_data.append(row_data)
        
        df = pd.DataFrame(transposed_monthly_data)
        df['Tarih'] = pd.to_datetime(df['Tarih']).dt.strftime('%Y-%m-%d')
        df = df.sort_values('Tarih', ascending=False)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Gruplar Aylık Değişim')
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='gruplar_aylik_degisim.xlsx'
        )
    except Exception as e:
        flash(f'Excel indirme hatası: {str(e)}', 'error')
        return redirect(url_for('ana_gruplar'))

@app.route('/download/tufe-endeksler/csv')
def download_tufe_endeksler_csv():
    try:
        # Get endeksler.csv data
        endeksler_df = cached_read_csv('endeksler.csv', index_col=0)
        endeksler_df.index = pd.to_datetime(endeksler_df.index)
        endeksler_df = endeksler_df.sort_index(ascending=False)  # Most recent first
        # Reset index to make Tarih a column
        endeksler_df = endeksler_df.reset_index()
        endeksler_df.rename(columns={endeksler_df.columns[0]: 'Tarih'}, inplace=True)
        # Format Tarih column as YYYY-MM-DD string
        endeksler_df['Tarih'] = endeksler_df['Tarih'].dt.strftime('%Y-%m-%d')
        
        # Create a temporary CSV file
        from io import BytesIO
        output = BytesIO()
        endeksler_df.to_csv(output, index=False, encoding='utf-8-sig')
        output.seek(0)
        return send_file(output, mimetype='text/csv', as_attachment=True, download_name='tufe_endeksler.csv')
    except Exception as e:
        flash(f'CSV indirme hatası: {str(e)}', 'error')
        return redirect(url_for('tufe'))

@app.route('/download/tufe-endeksler/xlsx')
def download_tufe_endeksler_xlsx():
    try:
        # Get endeksler.csv data
        endeksler_df = cached_read_csv('endeksler.csv', index_col=0)
        endeksler_df.index = pd.to_datetime(endeksler_df.index)
        endeksler_df = endeksler_df.sort_index(ascending=False)  # Most recent first
        # Reset index to make Tarih a column
        endeksler_df = endeksler_df.reset_index()
        endeksler_df.rename(columns={endeksler_df.columns[0]: 'Tarih'}, inplace=True)
        # Format Tarih column as YYYY-MM-DD string
        endeksler_df['Tarih'] = endeksler_df['Tarih'].dt.strftime('%Y-%m-%d')
        
        # Create a temporary Excel file
        from io import BytesIO
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            endeksler_df.to_excel(writer, index=False, sheet_name='Endeksler')
        output.seek(0)
        return send_file(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 
                        as_attachment=True, download_name='tufe_endeksler.xlsx')
    except Exception as e:
        flash(f'Excel indirme hatası: {str(e)}', 'error')
        return redirect(url_for('tufe'))

@app.route('/download/maddeler-aylik/csv')
def download_maddeler_aylik_csv():
    try:
        # Get maddeleraylık.csv data
        maddeler_monthly_df = pd.read_csv('maddeleraylık.csv', index_col=0)
        # After index_col=0, columns are: ['Madde', '2025-02-28', ...]
        # Get 'Madde' column values
        madde_names_list = maddeler_monthly_df['Madde'].tolist()
        # Get date columns (all columns except 'Madde')
        date_columns_monthly = [col for col in maddeler_monthly_df.columns if col != 'Madde']
        
        # Create transposed dataframe
        transposed_monthly_data = []
        for date_col in date_columns_monthly:
            row_data = {'Tarih': date_col}
            for idx, madde_name in enumerate(madde_names_list):
                value = maddeler_monthly_df.iloc[idx][date_col]
                try:
                    value = float(str(value).replace(',', '.'))
                except:
                    value = None
                row_data[madde_name] = value
            transposed_monthly_data.append(row_data)
        
        df = pd.DataFrame(transposed_monthly_data)
        # Format Tarih column as YYYY-MM-DD string
        try:
            df['Tarih'] = pd.to_datetime(df['Tarih']).dt.strftime('%Y-%m-%d')
        except:
            pass
        df = df.sort_values('Tarih', ascending=False)  # Most recent first
        
        # Create a temporary CSV file
        from io import BytesIO
        output = BytesIO()
        df.to_csv(output, index=False, encoding='utf-8-sig')
        output.seek(0)
        return send_file(output, mimetype='text/csv', as_attachment=True, download_name='maddeler_aylik_degisim.csv')
    except Exception as e:
        flash(f'CSV indirme hatası: {str(e)}', 'error')
        return redirect(url_for('tufe'))

@app.route('/download/maddeler-aylik/xlsx')
def download_maddeler_aylik_xlsx():
    try:
        # Get maddeleraylık.csv data
        maddeler_monthly_df = pd.read_csv('maddeleraylık.csv', index_col=0)
        # After index_col=0, columns are: ['Madde', '2025-02-28', ...]
        # Get 'Madde' column values
        madde_names_list = maddeler_monthly_df['Madde'].tolist()
        # Get date columns (all columns except 'Madde')
        date_columns_monthly = [col for col in maddeler_monthly_df.columns if col != 'Madde']
        
        # Create transposed dataframe
        transposed_monthly_data = []
        for date_col in date_columns_monthly:
            row_data = {'Tarih': date_col}
            for idx, madde_name in enumerate(madde_names_list):
                value = maddeler_monthly_df.iloc[idx][date_col]
                try:
                    value = float(str(value).replace(',', '.'))
                except:
                    value = None
                row_data[madde_name] = value
            transposed_monthly_data.append(row_data)
        
        df = pd.DataFrame(transposed_monthly_data)
        # Format Tarih column as YYYY-MM-DD string
        try:
            df['Tarih'] = pd.to_datetime(df['Tarih']).dt.strftime('%Y-%m-%d')
        except:
            pass
        df = df.sort_values('Tarih', ascending=False)  # Most recent first
        
        # Create a temporary Excel file
        from io import BytesIO
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Aylık Değişimler')
        output.seek(0)
        return send_file(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 
                        as_attachment=True, download_name='maddeler_aylik_degisim.xlsx')
    except Exception as e:
        flash(f'Excel indirme hatası: {str(e)}', 'error')
        return redirect(url_for('tufe'))

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

# Manifest route
@app.route('/manifest.json')
def manifest():
    return send_file('static/manifest.json', mimetype='application/manifest+json')


# Get VAPID public key
@app.route('/api/push/vapid-public-key', methods=['GET'])
def get_vapid_public_key():
    if not VAPID_PUBLIC_KEY:
        return jsonify({'error': 'VAPID public key not configured'}), 500
    return jsonify({'publicKey': VAPID_PUBLIC_KEY})

# Helper functions for push subscriptions storage
def save_subscription_to_storage(endpoint, p256dh, auth, user_agent):
    """Save subscription to Google Sheets or SQLite"""
    sheet = get_push_subscriptions_sheet()
    
    if sheet:
        # Use Google Sheets
        try:
            # Check if endpoint already exists
            cell = sheet.find(endpoint)
            if cell:
                # Update existing row
                row_num = cell.row
                sheet.update(f'A{row_num}:E{row_num}', [[endpoint, p256dh, auth, user_agent, datetime.now().isoformat()]])
            else:
                # Add new row
                sheet.append_row([endpoint, p256dh, auth, user_agent, datetime.now().isoformat()])
            return True
        except Exception as find_error:
            # If find() raises an exception (like CellNotFound in some versions), just append
            try:
                sheet.append_row([endpoint, p256dh, auth, user_agent, datetime.now().isoformat()])
                return True
            except Exception as append_error:
                print(f"Error appending row: {str(append_error)}")
                return False
    
    # Fallback to SQLite
    try:
        conn = sqlite3.connect('push_subscriptions.db')
        c = conn.cursor()
        c.execute('''
            INSERT OR REPLACE INTO subscriptions (endpoint, p256dh, auth, user_agent)
            VALUES (?, ?, ?, ?)
        ''', (endpoint, p256dh, auth, user_agent))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving to SQLite: {str(e)}")
        return False

def get_all_subscriptions():
    """Get all subscriptions from Google Sheets or SQLite"""
    sheet = get_push_subscriptions_sheet()
    
    if sheet:
        # Use Google Sheets
        try:
            # Get all values (skip header row)
            all_values = sheet.get_all_values()
            if len(all_values) <= 1:
                return []  # Only header or empty
            
            subscriptions = []
            for row in all_values[1:]:  # Skip header row
                if len(row) >= 3 and row[0]:  # Ensure we have at least endpoint, p256dh, auth
                    subscriptions.append((row[0], row[1], row[2]))
            
            return subscriptions
        except Exception as e:
            print(f"Error reading from Google Sheets: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return []
    
    # Fallback to SQLite
    try:
        conn = sqlite3.connect('push_subscriptions.db')
        c = conn.cursor()
        c.execute('SELECT endpoint, p256dh, auth FROM subscriptions')
        subscriptions = c.fetchall()
        conn.close()
        return subscriptions
    except Exception as e:
        print(f"Error reading from SQLite: {str(e)}")
        return []

def delete_subscription_from_storage(endpoint):
    """Delete subscription from Google Sheets or SQLite"""
    sheet = get_push_subscriptions_sheet()
    
    if sheet:
        # Use Google Sheets
        try:
            cell = sheet.find(endpoint)
            if cell:
                sheet.delete_rows(cell.row)
            # If cell not found, it's already deleted, so return True
            return True
        except Exception as e:
            # If find() raises exception, assume it's already deleted
            error_str = str(e).lower()
            if 'not found' in error_str or 'cellnotfound' in error_str:
                return True  # Already deleted
            print(f"Error deleting from Google Sheets: {str(e)}")
            return False
    
    # Fallback to SQLite
    try:
        conn = sqlite3.connect('push_subscriptions.db')
        c = conn.cursor()
        c.execute('DELETE FROM subscriptions WHERE endpoint = ?', (endpoint,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error deleting from SQLite: {str(e)}")
        return False

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
        
        # Save subscription to storage (Google Sheets or SQLite)
        if save_subscription_to_storage(endpoint, p256dh, auth, user_agent):
            return jsonify({'success': True, 'message': 'Abonelik başarıyla kaydedildi'}), 200
        else:
            return jsonify({'error': 'Failed to save subscription'}), 500
    
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
        
        # Remove subscription from storage
        if delete_subscription_from_storage(endpoint):
            return jsonify({'success': True, 'message': 'Abonelik iptal edildi'}), 200
        else:
            return jsonify({'error': 'Failed to delete subscription'}), 500
    
    except Exception as e:
        print(f"Unsubscribe error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Admin panel for sending emails with BCC
@app.route('/admin', methods=['GET', 'POST'])
def admin_email_panel():
    # Simple API key authentication
    admin_api_key = os.environ.get('ADMIN_API_KEY', '')
    BCC_EMAIL = 'bora.587@hotmail.com'  # BCC email address
    
    if request.method == 'GET':
        return render_template('admin.html', active_page='admin', has_api_key=bool(admin_api_key))
    
    if request.method == 'POST':
        # Check API key if set
        if admin_api_key:
            provided_key = request.form.get('api_key', '')
            if provided_key != admin_api_key:
                flash('❌ Geçersiz API key!', 'error')
                return redirect(url_for('admin_email_panel'))
        
        # Get email data from form
        to_email = request.form.get('to_email', '').strip()
        subject = request.form.get('subject', '').strip()
        body = request.form.get('body', '').strip()
        
        if not to_email or not subject or not body:
            flash('❌ Alıcı, konu ve mesaj gereklidir!', 'error')
            return redirect(url_for('admin_email_panel'))
        
        # Email validation
        if '@' not in to_email or '.' not in to_email:
            flash('❌ Geçerli bir e-posta adresi giriniz!', 'error')
            return redirect(url_for('admin_email_panel'))
        
        try:
            # Load logo and convert to base64 for email embedding
            logo_base64 = None
            try:
                logo_path = os.path.join(os.path.dirname(__file__), 'static', 'logo.png')
                if os.path.exists(logo_path):
                    with open(logo_path, 'rb') as logo_file:
                        logo_data = logo_file.read()
                        logo_base64 = base64.b64encode(logo_data).decode('utf-8')
            except Exception as e:
                print(f"Logo yüklenemedi: {str(e)}")
            
            # Create email message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = os.environ.get('SMTP_USERNAME', 'webtufe@gmail.com')
            msg['To'] = to_email
            msg['Bcc'] = BCC_EMAIL  # BCC olarak bora.587@hotmail.com ekleniyor
            
            # Create signature HTML
            signature_html = ""
            if logo_base64:
                signature_html = f"""
                <div style="margin-top: 40px; padding-top: 30px; border-top: 2px solid #e0e0e0;">
                    <table cellpadding="0" cellspacing="0" border="0" style="width: 100%;">
                        <tr>
                            <td style="vertical-align: top; padding-right: 20px;">
                                <img src="data:image/png;base64,{logo_base64}" alt="Web-TÜFE Logo" style="max-width: 120px; height: auto; display: block;">
                            </td>
                            <td style="vertical-align: top;">
                                <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
                                    <p style="margin: 0 0 8px 0; font-size: 18px; font-weight: 600; color: #800020; line-height: 1.4;">
                                        Bora Kaya
                                    </p>
                                    <p style="margin: 0; font-size: 14px; color: #4F46E5; line-height: 1.4;">
                                        Web Tüketici Fiyat Endeksi Kurucusu
                                    </p>
                                </div>
                            </td>
                        </tr>
                    </table>
                </div>
                """
            else:
                signature_html = """
                <div style="margin-top: 40px; padding-top: 30px; border-top: 2px solid #e0e0e0;">
                    <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
                        <p style="margin: 0 0 8px 0; font-size: 18px; font-weight: 600; color: #800020; line-height: 1.4;">
                            Bora Kaya
                        </p>
                        <p style="margin: 0; font-size: 14px; color: #4F46E5; line-height: 1.4;">
                            Web Tüketici Fiyat Endeksi Kurucusu
                        </p>
                    </div>
                </div>
                """
            
            # Create HTML content
            html_content = f"""
            <html>
                <body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; background-color: #ffffff;">
                    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                        <h2 style="color: #4F46E5; margin-top: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">{subject}</h2>
                    </div>
                    <div style="background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0;">
                        <div style="white-space: pre-wrap; margin-bottom: 20px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #333; font-size: 15px;">{body}</div>
                    </div>
                    {signature_html}
                    <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 11px; color: #999; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
                        <p style="margin: 5px 0;">Bu e-posta Web TÜFE admin panelinden gönderilmiştir.</p>
                        <p style="margin: 5px 0;">Tarih: {datetime.now().strftime('%d.%m.%Y %H:%M')}</p>
                    </div>
                </body>
            </html>
            """
            
            # Add HTML content
            html_part = MIMEText(html_content, 'html', 'utf-8')
            msg.attach(html_part)
            
            # SMTP configuration
            smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
            smtp_port = int(os.environ.get('SMTP_PORT', '587'))
            smtp_username = os.environ.get('SMTP_USERNAME', 'borakaya8@gmail.com')
            smtp_password = os.environ.get('SMTP_PASSWORD', '')
            
            # Check if SMTP password is configured
            if not smtp_password:
                error_msg = '❌ SMTP şifresi ayarlanmamış! Lütfen SMTP_PASSWORD environment variable\'ını ayarlayın.'
                print(error_msg)
                flash(error_msg, 'error')
                return redirect(url_for('admin_email_panel'))
            
            if not smtp_username:
                error_msg = '❌ SMTP kullanıcı adı ayarlanmamış! Lütfen SMTP_USERNAME environment variable\'ını ayarlayın.'
                print(error_msg)
                flash(error_msg, 'error')
                return redirect(url_for('admin_email_panel'))
            
            # Send email via SMTP
            print(f"SMTP bağlantısı kuruluyor: {smtp_server}:{smtp_port}")
            print(f"Kullanıcı: {smtp_username}")
            print(f"Alıcı: {to_email}")
            print(f"BCC: {BCC_EMAIL}")
            
            with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server:
                server.starttls()
                server.login(smtp_username, smtp_password)
                server.send_message(msg)
            
            print(f"✅ E-posta başarıyla gönderildi: {to_email} (BCC: {BCC_EMAIL})")
            flash(f'✅ E-posta başarıyla gönderildi! (BCC: {BCC_EMAIL})', 'success')
            
        except smtplib.SMTPAuthenticationError as e:
            error_msg = f'❌ SMTP kimlik doğrulama hatası! Kullanıcı adı veya şifre hatalı olabilir. Gmail kullanıyorsanız "Uygulama Şifresi" kullanmayı deneyin. Hata: {str(e)}'
            print(f"SMTP Kimlik Doğrulama Hatası: {str(e)}")
            flash(error_msg, 'error')
        except smtplib.SMTPConnectError as e:
            error_msg = f'❌ SMTP sunucusuna bağlanılamadı! Sunucu adresini ve portu kontrol edin. Hata: {str(e)}'
            print(f"SMTP Bağlantı Hatası: {str(e)}")
            flash(error_msg, 'error')
        except smtplib.SMTPException as e:
            error_msg = f'❌ E-posta gönderilirken SMTP hatası oluştu: {str(e)}'
            print(f"SMTP Hatası: {str(e)}")
            flash(error_msg, 'error')
        except Exception as e:
            error_msg = f'❌ Beklenmeyen bir hata oluştu: {str(e)}'
            print(f"Genel Hata: {str(e)}")
            import traceback
            traceback.print_exc()
            flash(error_msg, 'error')
        
        return redirect(url_for('admin_email_panel'))

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
                error_summary = result.get('error_summary', {})
                
                # Check for VAPID mismatch errors
                vapid_mismatch_found = False
                error_details_all = result.get('error_details', {})
                for endpoint_type, errors in error_details_all.items():
                    for error in errors:
                        if error.get('vapid_mismatch', False):
                            vapid_mismatch_found = True
                            break
                    if vapid_mismatch_found:
                        break
                
                message = f"✅ Bildirim gönderildi! {sent} kullanıcıya ulaştı."
                if failed > 0:
                    message += f" ({failed} başarısız)"
                    if error_summary:
                        error_types = ", ".join([f"{k}: {v}" for k, v in error_summary.items()])
                        message += f" - Hata tipleri: {error_types}"
                
                if vapid_mismatch_found:
                    message += " ⚠️ Bazı abonelikler eski VAPID key'leri ile oluşturulmuş - kullanıcıların yeniden abone olması gerekiyor!"
                
                # Check for WNS 401 errors specifically
                wns_401_errors = 0
                error_details_all = result.get('error_details', {})
                for endpoint_type, errors in error_details_all.items():
                    if 'WNS' in endpoint_type:
                        for error in errors:
                            if error.get('status') == 401:
                                wns_401_errors += 1
                
                if wns_401_errors > 0:
                    message += f" ⚠️ WNS (Edge) endpoint'leri için {wns_401_errors} hata - bu pywebpush ve WNS uyumsuzluğu olabilir (bilinen sorun)."
                
                flash(message, 'success' if sent > 0 else 'error')
            else:
                flash(f"❌ Hata: {result.get('error', 'Bilinmeyen hata')}", 'error')
        except Exception as e:
            flash(f'❌ Hata: {str(e)}', 'error')
        
        return redirect(url_for('admin_push_panel'))

def detect_endpoint_type(endpoint):
    """Detect the type of push notification endpoint"""
    if not endpoint:
        return "unknown"
    endpoint_lower = endpoint.lower()
    if "fcm.googleapis.com" in endpoint_lower or "googleapis.com" in endpoint_lower:
        return "FCM (Chrome)"
    elif "notify.windows.com" in endpoint_lower or "wns" in endpoint_lower:
        return "WNS (Windows/Edge)"
    elif "updates.push.services.mozilla.com" in endpoint_lower:
        return "Mozilla (Firefox)"
    elif "push.apple.com" in endpoint_lower or "api.push.apple.com" in endpoint_lower:
        return "APNs (Safari)"
    else:
        return "Unknown"

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
        
        # Get all subscriptions from storage
        subscriptions = get_all_subscriptions()
        
        if not subscriptions:
            return {'success': True, 'message': 'No subscriptions found', 'sent': 0, 'failed': 0, 'total': 0}
        
        # VAPID claims - ensure email is properly formatted with mailto: prefix
        # This is required by the Web Push Protocol specification
        vapid_sub = VAPID_CLAIM_EMAIL
        if vapid_sub and not vapid_sub.startswith('mailto:'):
            vapid_sub = f'mailto:{vapid_sub}'
        elif not vapid_sub:
            vapid_sub = 'mailto:webtufe@example.com'
        
        vapid_claims = {
            "sub": vapid_sub
        }
        
        # Get VAPID private key in base64 URL-safe format (pywebpush expects this)
        vapid_private_key = get_vapid_private_key_for_webpush()
        if not vapid_private_key:
            return {'success': False, 'error': 'VAPID private key not configured or invalid'}
        
        success_count = 0
        fail_count = 0
        error_details = {}
        
        # Separate WNS and non-WNS endpoints to avoid potential pywebpush state issues
        # Process WNS endpoints first, then others, to isolate any potential issues
        wns_subscriptions = []
        other_subscriptions = []
        
        for endpoint, p256dh, auth in subscriptions:
            is_wns = "notify.windows.com" in endpoint.lower() or "wns" in endpoint.lower()
            if is_wns:
                wns_subscriptions.append((endpoint, p256dh, auth))
            else:
                other_subscriptions.append((endpoint, p256dh, auth))
        
        # Process all subscriptions: WNS first, then others
        all_subscriptions_ordered = wns_subscriptions + other_subscriptions
        
        for endpoint, p256dh, auth in all_subscriptions_ordered:
            # Wrap each endpoint processing in its own try-except to ensure loop continues
            try:
                endpoint_type = detect_endpoint_type(endpoint)
                max_retries = 2
                retry_count = 0
                success = False
                
                # Create fresh VAPID claims for each endpoint to avoid any state issues
                # This ensures each endpoint gets a clean, isolated set of parameters
                endpoint_vapid_claims = {
                    "sub": vapid_sub
                }
                
                while retry_count <= max_retries and not success:
                    try:
                        # Create fresh subscription info for each attempt
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
                        
                        # Prepare webpush parameters - create fresh dict for each endpoint
                        # Use endpoint-specific VAPID claims to ensure isolation
                        webpush_params = {
                            "subscription_info": subscription_info,
                            "data": payload,
                            "vapid_private_key": vapid_private_key,  # This should be safe to reuse
                            "vapid_claims": endpoint_vapid_claims,  # Use endpoint-specific claims
                            "ttl": 86400,  # 24 hours TTL - some endpoints require this
                        }
                        
                        # For WNS endpoints, ensure proper configuration
                        # Note: WNS endpoints use web push protocol but may have stricter VAPID requirements
                        is_wns_endpoint = "notify.windows.com" in endpoint.lower() or "wns" in endpoint.lower()
                        if is_wns_endpoint:
                            # Ensure TTL is set (required by some endpoints)
                            webpush_params["ttl"] = 86400
                            # WNS may require specific VAPID claim format
                            # The vapid_claims with 'mailto:' prefix should be correct
                            # Note: Some WNS endpoints may have issues with VAPID authentication
                            # This is a known issue with certain Edge browser versions
                        
                        # Send push notification
                        # Each webpush call is isolated with its own parameters
                        # Add a small delay between endpoints to avoid potential connection pooling issues
                        if retry_count > 0 or endpoint != all_subscriptions_ordered[0][0]:
                            import time
                            time.sleep(0.1)  # Small delay to ensure clean state between endpoints
                        
                        webpush(**webpush_params)
                        
                        success_count += 1
                        success = True
                        
                    except WebPushException as e:
                        # Try to get status code from response object
                        status_code = None
                        if e.response:
                            try:
                                status_code = e.response.status_code
                            except:
                                pass
                        
                        # If status_code is None, try to extract from error message
                        error_msg = str(e)
                        if status_code is None:
                            # Extract status code from error message (e.g., "401 Unauthorized", "Push failed: 401")
                            import re
                            status_match = re.search(r'\b(\d{3})\b', error_msg)
                            if status_match:
                                status_code = int(status_match.group(1))
                        
                        response_text = ""
                        response_headers = {}
                        request_headers_sent = {}
                        
                        # Get response details if available - enhanced for WNS
                        if e.response:
                            try:
                                # Try to get response text - multiple methods
                                response_text = ""
                                if hasattr(e.response, 'text'):
                                    try:
                                        response_text = e.response.text[:500] if e.response.text else ""
                                    except:
                                        pass
                                
                                if not response_text and hasattr(e.response, 'content'):
                                    try:
                                        content = e.response.content
                                        if isinstance(content, bytes):
                                            response_text = content[:500].decode('utf-8', errors='ignore')
                                        else:
                                            response_text = str(content)[:500]
                                    except:
                                        pass
                                
                                if not response_text and hasattr(e.response, 'body'):
                                    try:
                                        response_text = str(e.response.body)[:500]
                                    except:
                                        pass
                                
                                # Try to get response headers - enhanced methods
                                if hasattr(e.response, 'headers'):
                                    try:
                                        headers_obj = e.response.headers
                                        # Try different header object types
                                        if hasattr(headers_obj, 'get'):
                                            # Case-insensitive dict or similar
                                            response_headers = dict(headers_obj)
                                        elif hasattr(headers_obj, 'items'):
                                            # Dict-like object
                                            try:
                                                response_headers = {str(k): str(v) for k, v in headers_obj.items()}
                                            except:
                                                response_headers = {}
                                        elif isinstance(headers_obj, dict):
                                            response_headers = dict(headers_obj)
                                        else:
                                            # Try to convert to dict
                                            try:
                                                response_headers = dict(headers_obj)
                                            except:
                                                response_headers = {}
                                    except Exception as header_error:
                                        print(f"   ⚠️  Could not parse response headers: {header_error}")
                                        response_headers = {}
                                
                                # Try to get request headers that were sent
                                if hasattr(e.response, 'request'):
                                    try:
                                        req = e.response.request
                                        if hasattr(req, 'headers'):
                                            req_headers = req.headers
                                            if hasattr(req_headers, 'items'):
                                                request_headers_sent = {str(k): str(v) for k, v in req_headers.items()}
                                            elif isinstance(req_headers, dict):
                                                request_headers_sent = dict(req_headers)
                                    except:
                                        pass
                                            
                                # Also try to get status code from response if not already set
                                if status_code is None:
                                    if hasattr(e.response, 'status_code'):
                                        try:
                                            status_code = e.response.status_code
                                        except:
                                            pass
                                    elif hasattr(e.response, 'status'):
                                        try:
                                            status_code = e.response.status
                                        except:
                                            pass
                                            
                            except Exception as resp_error:
                                print(f"   ⚠️  Warning: Could not parse response details: {resp_error}")
                                import traceback
                                print(f"   Traceback: {traceback.format_exc()}")
                        
                        # Also try to get info from exception itself
                        if not status_code:
                            # Try to extract from exception message or args
                            if hasattr(e, 'args') and e.args:
                                for arg in e.args:
                                    if isinstance(arg, str) and '401' in arg:
                                        status_code = 401
                                    elif isinstance(arg, str) and '403' in arg:
                                        status_code = 403
                        
                        # For 401 errors, try to get more details
                        if status_code == 401 or "401" in error_msg or "unauthorized" in error_msg.lower():
                            # Ensure status_code is set to 401
                            if status_code != 401:
                                status_code = 401
                            # Check if it's a VAPID authentication issue
                            # Log detailed error for debugging
                            print(f"⚠️ 401 Unauthorized for {endpoint_type} endpoint {endpoint[:50]}...")
                            print(f"   Error: {error_msg}")
                            
                            # Special handling for WNS endpoints
                            if "notify.windows.com" in endpoint.lower() or "wns" in endpoint.lower():
                                print(f"   ℹ️  WNS endpoint detected - this may be a VAPID authentication issue")
                                print(f"   ℹ️  WNS requires valid VAPID keys with proper 'mailto:' claim format")
                                print(f"   ℹ️  VAPID claim used: {endpoint_vapid_claims.get('sub', 'Not set')}")
                            
                            # Log response body
                            if response_text:
                                print(f"   Response body: {response_text}")
                            else:
                                print(f"   Response body: (empty or not available)")
                            
                            # Log all response headers for WNS endpoints, or relevant ones for others
                            if response_headers:
                                if "notify.windows.com" in endpoint.lower() or "wns" in endpoint.lower():
                                    # For WNS, log all headers for debugging
                                    print(f"   📋 All response headers: {response_headers}")
                                else:
                                    # For other endpoints, log only relevant headers
                                    relevant_headers = {k: v for k, v in response_headers.items() 
                                                      if k.lower() in ['www-authenticate', 'x-wns-notificationstatus', 
                                                                      'x-wns-msg-id', 'x-wns-debug-trace', 'retry-after',
                                                                      'content-type', 'content-length']}
                                    if relevant_headers:
                                        print(f"   📋 Response headers: {relevant_headers}")
                                
                                # Check for WNS-specific error indicators
                                headers_lower = {k.lower(): v for k, v in response_headers.items()}
                                if 'www-authenticate' in headers_lower:
                                    print(f"   ⚠️  WWW-Authenticate header: {headers_lower.get('www-authenticate', 'Not found')}")
                                if 'x-wns-notificationstatus' in headers_lower:
                                    print(f"   ℹ️  X-WNS-NotificationStatus: {headers_lower.get('x-wns-notificationstatus', 'Not found')}")
                                if 'x-wns-debug-trace' in headers_lower:
                                    print(f"   ℹ️  X-WNS-Debug-Trace: {headers_lower.get('x-wns-debug-trace', 'Not found')}")
                            else:
                                print(f"   📋 Response headers: (not available - this might indicate a pywebpush parsing issue)")
                            
                            # Log request headers that were sent (especially for WNS)
                            if request_headers_sent and ("notify.windows.com" in endpoint.lower() or "wns" in endpoint.lower()):
                                print(f"   📤 Request headers sent:")
                                # Log Authorization header (masked) and other important headers
                                for key, value in request_headers_sent.items():
                                    if key.lower() == 'authorization':
                                        # Mask the token part
                                        if 'Bearer' in value or 'WebPush' in value:
                                            parts = value.split(' ', 1)
                                            if len(parts) > 1:
                                                token = parts[1]
                                                masked = token[:20] + '...' + token[-10:] if len(token) > 30 else '***'
                                                print(f"      {key}: {parts[0]} {masked}")
                                            else:
                                                print(f"      {key}: {value[:50]}...")
                                        else:
                                            print(f"      {key}: {value[:50]}...")
                                    elif key.lower() in ['content-type', 'content-encoding', 'ttl', 'content-length']:
                                        print(f"      {key}: {value}")
                                
                            # Log additional debugging info for WNS
                            if "notify.windows.com" in endpoint.lower() or "wns" in endpoint.lower():
                                print(f"   🔍 WNS Debugging info:")
                                print(f"      - Endpoint: {endpoint[:80]}")
                                print(f"      - Has p256dh key: {'Yes' if p256dh else 'No'} ({len(p256dh) if p256dh else 0} chars)")
                                print(f"      - Has auth key: {'Yes' if auth else 'No'} ({len(auth) if auth else 0} chars)")
                                print(f"      - VAPID private key format: {'Set' if vapid_private_key else 'Not set'}")
                                if vapid_private_key:
                                    print(f"      - VAPID private key length: {len(vapid_private_key)} chars")
                                    print(f"      - VAPID private key starts with: {vapid_private_key[:20]}...")
                                print(f"      - VAPID claim (sub): {endpoint_vapid_claims.get('sub', 'Not set')}")
                                print(f"      - Payload size: {len(payload)} bytes")
                                print(f"      - TTL: {webpush_params.get('ttl', 'Not set')}")
                                
                                # Check if this might be a VAPID key mismatch issue or pywebpush bug
                                print(f"   💡 Possible causes for WNS 401 error:")
                                print(f"      1. VAPID keys don't match the subscription (user subscribed with different keys)")
                                print(f"      2. pywebpush library issue with WNS endpoints (known issue)")
                                print(f"      3. WNS endpoint doesn't fully support VAPID authentication")
                                print(f"      4. VAPID key format issue (should be base64 URL-safe)")
                                print(f"      5. Edge browser version compatibility issue")
                                print(f"   💡 Solutions to try:")
                                print(f"      a) User should unsubscribe and re-subscribe with current VAPID keys")
                                print(f"      b) Check if pywebpush needs to be updated")
                                print(f"      c) Verify VAPID keys are correctly formatted")
                                print(f"      d) Test with a different Edge browser version")
                                
                                # Additional debugging: Check if VAPID public key matches
                                print(f"   🔍 VAPID Key Verification:")
                                print(f"      - VAPID Public Key (first 30 chars): {VAPID_PUBLIC_KEY[:30] if VAPID_PUBLIC_KEY else 'Not set'}...")
                                print(f"      - VAPID Private Key length: {len(vapid_private_key) if vapid_private_key else 0} chars")
                                print(f"      - VAPID Claim: {endpoint_vapid_claims.get('sub', 'Not set')}")
                                
                                # Check if this is a known pywebpush WNS issue
                                print(f"   ⚠️  NOTE: This might be a known issue with pywebpush and WNS endpoints")
                                print(f"      Some users report that WNS endpoints don't work properly with pywebpush")
                                print(f"      even with correct VAPID keys. This could be a library limitation.")
                            
                            # For WNS endpoints with 401, this might be a VAPID configuration issue
                            # Possible causes:
                            # 1. VAPID keys are invalid or incorrectly formatted
                            # 2. VAPID claim (sub) is not properly formatted with 'mailto:' prefix
                            # 3. WNS endpoint requires additional authentication (though this shouldn't happen with web push)
                            # 4. The subscription endpoint itself might be invalid (though we keep it as requested)
                            # We keep the subscription as requested by user, but log the issue
                            if endpoint_type not in error_details:
                                error_details[endpoint_type] = []
                            error_details[endpoint_type].append({
                                'endpoint': endpoint[:50],
                                'status': 401,
                                'error': error_msg,
                                'response_body': response_text if response_text else None,
                                'vapid_claim': endpoint_vapid_claims.get('sub', 'Not set')
                            })
                            fail_count += 1
                            # Don't retry 401 errors - it's an authentication issue
                            break
                        
                        # For 403 (Forbidden) errors - VAPID credentials don't match
                        # This happens when VAPID keys were changed after subscriptions were created
                        elif status_code == 403 or "403" in error_msg or "forbidden" in error_msg.lower():
                            # Ensure status_code is set to 403
                            if status_code != 403:
                                status_code = 403
                            
                            print(f"⚠️ 403 Forbidden for {endpoint_type} endpoint {endpoint[:50]}...")
                            print(f"   Error: {error_msg}")
                            
                            # Check if it's a VAPID credentials mismatch
                            if "vapid credentials" in error_msg.lower() or "credentials used to create" in error_msg.lower():
                                print(f"   ⚠️  VAPID credentials mismatch detected!")
                                print(f"   ℹ️  This subscription was created with different VAPID keys")
                                print(f"   ℹ️  Current VAPID claim: {endpoint_vapid_claims.get('sub', 'Not set')}")
                                print(f"   ⚠️  This subscription is invalid and needs to be re-subscribed")
                                print(f"   💡 Solution: User needs to unsubscribe and re-subscribe with new VAPID keys")
                            
                            if response_text:
                                print(f"   Response body: {response_text}")
                            
                            if endpoint_type not in error_details:
                                error_details[endpoint_type] = []
                            error_details[endpoint_type].append({
                                'endpoint': endpoint[:50],
                                'status': 403,
                                'error': error_msg,
                                'response_body': response_text if response_text else None,
                                'vapid_mismatch': True
                            })
                            fail_count += 1
                            # Don't retry 403 errors - it's a credentials mismatch issue
                            break
                        
                        # For 410 (Gone) errors, subscription is invalid but we keep it as requested
                        elif status_code == 410:
                            print(f"⚠️ 410 Gone for {endpoint_type} endpoint {endpoint[:50]}... (subscription kept)")
                            if endpoint_type not in error_details:
                                error_details[endpoint_type] = []
                            error_details[endpoint_type].append({
                                'endpoint': endpoint[:50],
                                'status': status_code,
                                'error': 'Subscription expired or invalid'
                            })
                            fail_count += 1
                            break
                        
                        # For other errors (not 401, not 403, not 410), check if we should retry
                        # Don't retry 401/403 errors (already handled above)
                        elif status_code not in [401, 403] and retry_count < max_retries:
                            retry_count += 1
                            print(f"⚠️ Retry {retry_count}/{max_retries} for {endpoint_type} endpoint {endpoint[:50]}... (Status: {status_code})")
                            import time
                            time.sleep(0.5)  # Small delay before retry
                            continue
                        else:
                            # Max retries reached or non-retryable error
                            if status_code == 401:
                                # This shouldn't happen as we break above, but just in case
                                print(f"❌ 401 Unauthorized for {endpoint_type} endpoint {endpoint[:50]}...: {error_msg}")
                            else:
                                print(f"❌ Failed after {max_retries} retries for {endpoint_type} endpoint {endpoint[:50]}...: {error_msg} (Status: {status_code})")
                            
                            if endpoint_type not in error_details:
                                error_details[endpoint_type] = []
                            error_details[endpoint_type].append({
                                'endpoint': endpoint[:50],
                                'status': status_code if status_code else 'Unknown',
                                'error': error_msg,
                                'response_body': response_text if response_text else None
                            })
                            fail_count += 1
                            break
                            
                    except Exception as e:
                        # For non-WebPush exceptions, log and continue
                        error_msg = str(e)
                        print(f"❌ Exception for {endpoint_type} endpoint {endpoint[:50]}...: {error_msg}")
                        if endpoint_type not in error_details:
                            error_details[endpoint_type] = []
                        error_details[endpoint_type].append({
                            'endpoint': endpoint[:50],
                            'status': 'Exception',
                            'error': error_msg
                        })
                        fail_count += 1
                        break
                    
            except Exception as outer_exception:
                # Catch any exception that might occur outside the while loop
                # This ensures the for loop continues even if there's an unexpected error
                error_msg = str(outer_exception)
                endpoint_type = detect_endpoint_type(endpoint) if 'endpoint' in locals() else "Unknown"
                print(f"❌ Outer exception for {endpoint_type} endpoint {endpoint[:50] if 'endpoint' in locals() else 'unknown'}...: {error_msg}")
                import traceback
                print(f"   Traceback: {traceback.format_exc()}")
                if endpoint_type not in error_details:
                    error_details[endpoint_type] = []
                error_details[endpoint_type].append({
                    'endpoint': endpoint[:50] if 'endpoint' in locals() else 'unknown',
                    'status': 'OuterException',
                    'error': error_msg
                })
                fail_count += 1
                # Continue to next endpoint in for loop
                continue
        
        # Prepare result
        result = {
            'success': True,
            'sent': success_count,
            'failed': fail_count,
            'total': len(subscriptions),
            'error_summary': {k: len(v) for k, v in error_details.items()} if error_details else {},
            'error_details': error_details  # Include full error details for admin panel
        }
        
        # Log summary
        print(f"\n📊 Push notification summary:")
        print(f"   ✅ Sent: {success_count}")
        print(f"   ❌ Failed: {fail_count}")
        print(f"   Total: {len(subscriptions)}")
        
        # Check for VAPID credentials mismatch
        vapid_mismatch_count = 0
        for endpoint_type, errors in error_details.items():
            for error in errors:
                if error.get('vapid_mismatch', False):
                    vapid_mismatch_count += 1
        
        if vapid_mismatch_count > 0:
            print(f"\n⚠️  IMPORTANT: {vapid_mismatch_count} subscription(s) have VAPID credentials mismatch!")
            print(f"   💡 These subscriptions were created with OLD VAPID keys")
            print(f"   💡 Users need to unsubscribe and re-subscribe with NEW VAPID keys")
            print(f"   💡 This happens when VAPID keys are changed after subscriptions are created")
        
        if error_details:
            print(f"\n   ⚠️  Errors by type:")
            for endpoint_type, errors in error_details.items():
                vapid_mismatches = sum(1 for e in errors if e.get('vapid_mismatch', False))
                other_errors = len(errors) - vapid_mismatches
                error_desc = []
                if vapid_mismatches > 0:
                    error_desc.append(f"{vapid_mismatches} VAPID mismatch")
                if other_errors > 0:
                    error_desc.append(f"{other_errors} other")
                desc = " + ".join(error_desc) if error_desc else str(len(errors))
                print(f"      {endpoint_type}: {desc} errors")
        
        return result
    
    except Exception as e:
        print(f"❌ Send push error: {str(e)}")
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
        subscriptions = get_all_subscriptions()
        count = len(subscriptions)
        return jsonify({'count': count}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/aclik-siniri', methods=['GET', 'POST'])
def aclik_siniri():
    try:
        # Load açlıksınırı.csv
        df_raw = cached_read_csv('açlıksınırı.csv', index_col=0)
        # Clean column names (remove quotes and extra spaces)
        df_raw.columns = df_raw.columns.str.strip().str.strip('"').str.strip("'")
        df_raw.index = pd.to_datetime(df_raw.index)
        df_raw = df_raw.sort_index()
        
        # Create a copy for default calculations (before any user modifications)
        df_default = df_raw.copy()
        
        # Default quantities for each item
        default_quantities = {
            'Süt': 50,
            'Yoğurt': 18,
            'Beyaz Peynir': 4,
            'Dana Eti': 10,
            'Tavuk Eti': 10,
            'Balık': 4,
            'Mercimek': 2,
            'Kuru Fasulye': 3,
            'Nohut': 2,
            'Yumurta': 150,
            'Kuruyemiş (Fındık,Ceviz,Ayçekirdeği)': 1,
            'Ekmek': 45,
            'Pirinç': 7,
            'Makarna': 6,
            'Bulgur': 4,
            'Un': 4,
            'Patates': 10,
            'Soğan': 7,
            'Domates': 10,
            'Biber': 6,
            'Havuç': 5,
            'Mevsim Sebzeleri': 12,
            'Meyve': 35,
            'Ayçiçek Yağı': 4,
            'Margarin': 0.5,
            'Şeker': 3,
            'Reçel': 1,
            'Bal': 0.5,
            'Tuz': 0.7,
            'Çay': 0.7,
            'Salça': 2,
            'Baharat': 0.3
        }
        
        # Units for each item
        units = {
            'Süt': 'L',
            'Yoğurt': 'kg',
            'Beyaz Peynir': 'kg',
            'Dana Eti': 'kg',
            'Tavuk Eti': 'kg',
            'Balık': 'kg',
            'Mercimek': 'kg',
            'Kuru Fasulye': 'kg',
            'Nohut': 'kg',
            'Yumurta': 'Adet',
            'Kuruyemiş (Fındık,Ceviz,Ayçekirdeği)': 'kg',
            'Ekmek': 'kg',
            'Pirinç': 'kg',
            'Makarna': 'kg',
            'Bulgur': 'kg',
            'Un': 'kg',
            'Patates': 'kg',
            'Soğan': 'kg',
            'Domates': 'kg',
            'Biber': 'kg',
            'Havuç': 'kg',
            'Mevsim Sebzeleri': 'kg',
            'Meyve': 'kg',
            'Ayçiçek Yağı': 'L',
            'Margarin': 'kg',
            'Şeker': 'kg',
            'Reçel': 'kg',
            'Bal': 'kg',
            'Tuz': 'kg',
            'Çay': 'kg',
            'Salça': 'kg',
            'Baharat': 'kg'
        }
        
        # Get quantities from POST request if available
        quantities = default_quantities.copy()
        if request.method == 'POST':
            for item in default_quantities.keys():
                qty = request.form.get(f'qty_{item}')
                if qty:
                    try:
                        quantities[item] = float(qty)
                    except:
                        pass
        
        # Check if user is using custom basket (different from default)
        is_custom_basket = False
        for item in default_quantities.keys():
            if abs(quantities.get(item, 0) - default_quantities.get(item, 0)) > 0.001:
                is_custom_basket = True
                break
        
        # For monthly/yearly change calculations:
        # - If using default basket: use CSV's existing "Açlık Sınırı" column
        # - If using custom basket: calculate with user's quantities
        if is_custom_basket:
            # User modified basket - calculate with their quantities
            calculated_values_default = []
            for date in df_default.index:
                total = 0
                for item, qty in quantities.items():
                    if item in df_default.columns:
                        total += df_default.loc[date, item] * qty
                calculated_values_default.append(total)
            df_default['Açlık Sınırı'] = calculated_values_default
        else:
            # Using default basket - use CSV's existing "Açlık Sınırı" column if available
            if 'Açlık Sınırı' not in df_default.columns:
                calculated_values_default = []
                for date in df_default.index:
                    total = 0
                    for item, qty in default_quantities.items():
                        if item in df_default.columns:
                            total += df_default.loc[date, item] * qty
                    calculated_values_default.append(total)
                df_default['Açlık Sınırı'] = calculated_values_default
            # If CSV already has "Açlık Sınırı" column, use it directly (don't recalculate)
        
        # Create df for display (with user-modified quantities if any)
        df = df_raw.copy()
        # Calculate Açlık Sınırı based on quantities (for display)
        if 'Açlık Sınırı' in df.columns:
            # Use the formula to calculate
            calculated_values = []
            for date in df.index:
                total = 0
                for item, qty in quantities.items():
                    if item in df.columns:
                        total += df.loc[date, item] * qty
                calculated_values.append(total)
            
            # Update the Açlık Sınırı column with calculated values
            df['Açlık Sınırı'] = calculated_values
        
        # Create main line plot for Açlık Sınırı
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Açlık Sınırı'],
            mode='lines',
            name='Açlık Sınırı',
            line=dict(
                color='#EF476F',
                width=3
            ),
            hovertemplate='<b>%{customdata[0]}</b><br>Açlık Sınırı: %{customdata[1]:,.2f} TL<extra></extra>',
            customdata=[[f"{date.strftime('%Y-%m-%d')}", y] for date, y in zip(df.index, df['Açlık Sınırı'])]
        ))
        
        fig.update_layout(
            title=dict(
                text='Açlık Sınırı',
                font=dict(
                    size=24,
                    family='Inter, sans-serif',
                    color='#2B2D42'
                ),
                y=0.95
            ),
            xaxis=dict(
                title='Tarih',
                title_font=dict(size=14, family='Inter, sans-serif', color='#2B2D42'),
                tickfont=dict(size=12, family='Inter, sans-serif', color='#2B2D42'),
                gridcolor='#E9ECEF',
                zerolinecolor='#E9ECEF',
                tickformat='%Y-%m',
                dtick='M1',  # Her ay için bir tick
                tickangle=45
            ),
            yaxis=dict(
                title='Tutar (TL)',
                title_font=dict(size=14, family='Inter, sans-serif', color='#2B2D42'),
                tickfont=dict(size=12, family='Inter, sans-serif', color='#2B2D42'),
                gridcolor='#E9ECEF'
            ),
            showlegend=True,
            legend=dict(
                font=dict(size=12, family='Inter, sans-serif', color='#2B2D42'),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#E9ECEF',
                borderwidth=1
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
                align='left',
                namelength=-1
            ),
            hoverdistance=10
        )
        
        main_graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Calculate monthly change (aylık değişim)
        # 2024-12 tarihi ile başlayacak
        # Normal aylar: Ayın 24'ünde hesaplanır (ilk 24 günün ortalaması vs önceki ay ilk 24 günün ortalaması)
        # Son ay (eğer son gün < 24 ise): Son güne kadar olan ortalamayı önceki ayın aynı dönem ortalamasıyla kıyasla
        # NOTE: Monthly change is always calculated with default quantities, not user-modified quantities
        monthly_change_data = []
        monthly_change_dates = []
        
        df_monthly = df_default[df_default.index >= '2024-12-01'].copy()
        if not df_monthly.empty:
            # Son ayı bul
            last_date = df_monthly.index.max()
            last_year = last_date.year
            last_month = last_date.month
            last_day = last_date.day
            
            for (year, month), group in df_monthly.groupby([df_monthly.index.year, df_monthly.index.month]):
                # Önceki ay bilgisi
                if month == 1:
                    prev_year = year - 1
                    prev_month = 12
                else:
                    prev_year = year
                    prev_month = month - 1
                
                # Son ay mı ve son gün 24'ten küçük mü?
                is_last_month = (year == last_year and month == last_month)
                is_before_24 = is_last_month and last_day < 24
                
                if is_before_24:
                    # Son ay ve 24'ten önce: Son güne kadar olan ortalamayı hesapla
                    # Bu ayın ilk last_day gününün ortalaması
                    current_month_data = group[group.index.day <= last_day]
                    if len(current_month_data) > 0:
                        current_mean = current_month_data['Açlık Sınırı'].mean()
                        
                        # Önceki ayın ilk last_day gününün ortalaması
                        prev_month_data = df_default[(df_default.index.year == prev_year) & 
                                            (df_default.index.month == prev_month) & 
                                            (df_default.index.day <= last_day)]
                        if len(prev_month_data) > 0:
                            prev_mean = prev_month_data['Açlık Sınırı'].mean()
                            
                            if prev_mean > 0:
                                change = ((current_mean / prev_mean) - 1) * 100
                                monthly_change_data.append(change)
                                monthly_change_dates.append(last_date)
                else:
                    # Normal aylar veya son ay ama 24'ü geçmiş: Sadece 24'te hesapla
                    # Ayın 24'ünü bul
                    date_24 = pd.Timestamp(year, month, 24)
                    if date_24 in group.index:
                        # Bu ayın ilk 24 gününün ortalaması (1'den 24'e kadar, 24 dahil)
                        current_month_data = group[group.index.day <= 24]
                        if len(current_month_data) > 0:
                            current_mean = current_month_data['Açlık Sınırı'].mean()
                            
                            # Önceki ayın ilk 24 gününün ortalaması (1'den 24'e kadar, 24 dahil)
                            prev_month_data = df_default[(df_default.index.year == prev_year) & 
                                                (df_default.index.month == prev_month) & 
                                                (df_default.index.day <= 24)]
                            if len(prev_month_data) > 0:
                                prev_mean = prev_month_data['Açlık Sınırı'].mean()
                                
                                if prev_mean > 0:
                                    change = ((current_mean / prev_mean) - 1) * 100
                                    monthly_change_data.append(change)
                                    monthly_change_dates.append(date_24)
                    
                    # Son ay ve 24'ü geçmişse, 24'ten sonraki günler için de 24'teki değeri göster
                    if is_last_month and last_day >= 24:
                        # 24'teki değeri al
                        if date_24 in group.index:
                            current_month_data = group[group.index.day <= 24]
                            if len(current_month_data) > 0:
                                current_mean = current_month_data['Açlık Sınırı'].mean()
                                prev_month_data = df_default[(df_default.index.year == prev_year) & 
                                                    (df_default.index.month == prev_month) & 
                                                    (df_default.index.day <= 24)]
                                if len(prev_month_data) > 0:
                                    prev_mean = prev_month_data['Açlık Sınırı'].mean()
                                    if prev_mean > 0:
                                        change = ((current_mean / prev_mean) - 1) * 100
                                        # 24'ten sonraki tüm günler için aynı değeri ekle
                                        for date_after_24 in group[group.index.day > 24].index:
                                            monthly_change_data.append(change)
                                            monthly_change_dates.append(date_after_24)
        
        # Calculate yearly change (yıllık değişim)
        # NOTE: Yearly change is always calculated with default quantities, not user-modified quantities
        def gunluk_yillik_enflasyon(df, col="Açlık Sınırı"):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            df["Yıl"] = df.index.year
            df["Ay"] = df.index.month
            df["Gün"] = df.index.day

            results = []

            for current_date in df.index:
                yıl, ay, gün = current_date.year, current_date.month, current_date.day

                # (1) Bu ayın başından bugüne kadar olan ortalama (mevcut ortalama)
                current_mask = (df["Yıl"] == yıl) & (df["Ay"] == ay) & (df["Gün"] <= gün)
                current_mean = df.loc[current_mask, col].mean()

                # Geçen yılın aynı dönemi
                prev_mask = (df["Yıl"] == yıl - 1) & (df["Ay"] == ay) & (df["Gün"] <= gün)
                prev_mean = df.loc[prev_mask, col].mean()

                if np.isnan(prev_mean) or prev_mean == 0:
                    results.append(np.nan)
                    continue

                # (1) Gerçekleşen ortalama ile yıllık değişim
                real_change = (current_mean / prev_mean - 1) * 100

                # (2) Eğer endeks bugünden 24'e kadar sabit kalsaydı tahmini yıllık değişim
                if gün < 24:
                    # 24'e kadar sabit kalırsa, o ayın 1-24 arası ortalamasını tahmin et
                    future_days = 24 - gün
                    sabit_endeks = df.loc[current_date, col]
                    toplam_gün = gün + future_days
                    tahmini_ort = (current_mean * gün + sabit_endeks * future_days) / toplam_gün

                    prev_24_mask = (df["Yıl"] == yıl - 1) & (df["Ay"] == ay) & (df["Gün"] <= 24)
                    prev_24_mean = df.loc[prev_24_mask, col].mean()

                    if np.isnan(prev_24_mean) or prev_24_mean == 0:
                        tahmini_change = np.nan
                    else:
                        tahmini_change = (tahmini_ort / prev_24_mean - 1) * 100

                    # iki yöntemin ortalaması
                    enflasyon = (real_change + tahmini_change) / 2
                else:
                    # 24'ü ve sonrası: sadece gerçek (nihai)
                    enflasyon = real_change

                results.append(enflasyon)

            df["Yıllık Enflasyon"] = results

            # (3) Ayın 24'ünden sonrası sabit kalır
            sabit_df = []
            for (yıl, ay), group in df.groupby(["Yıl", "Ay"]):
                if 24 in group["Gün"].values:
                    sabit_deger = group.loc[group["Gün"] == 24, "Yıllık Enflasyon"].values[0]
                    group.loc[group["Gün"] > 24, "Yıllık Enflasyon"] = sabit_deger
                sabit_df.append(group)

            df_final = pd.concat(sabit_df).sort_index()
            return df_final[["Yıllık Enflasyon"]]
        
        # Calculate yearly change (using default quantities)
        yearly_df = gunluk_yillik_enflasyon(df_default, "Açlık Sınırı").dropna()
        yearly_df = yearly_df.sort_index()
        
        # Create monthly change chart
        monthly_fig = go.Figure()
        if monthly_change_data:
            monthly_fig.add_trace(go.Scatter(
                x=monthly_change_dates,
                y=monthly_change_data,
                mode='lines',
                name='Aylık Değişim',
                line=dict(
                    color='#06B6D4',
                    width=3
                ),
                hovertemplate='<b>%{customdata[0]}</b><br>Aylık Değişim: %{y:+.2f}%<extra></extra>',
                customdata=[[f"{date.strftime('%Y-%m')}"] for date in monthly_change_dates]
            ))
        
        monthly_fig.update_layout(
            title=dict(
                text='Aylık Değişim (%)',
                font=dict(
                    size=24,
                    family='Inter, sans-serif',
                    color='#2B2D42'
                ),
                y=0.95
            ),
            xaxis=dict(
                title='Tarih',
                title_font=dict(size=14, family='Inter, sans-serif', color='#2B2D42'),
                tickfont=dict(size=12, family='Inter, sans-serif', color='#2B2D42'),
                gridcolor='#E9ECEF',
                zerolinecolor='#E9ECEF',
                tickformat='%Y-%m',
                dtick='M1',
                tickangle=45
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
                borderwidth=1
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
                align='left',
                namelength=-1
            ),
            hoverdistance=100
        )
        
        monthly_graphJSON = json.dumps(monthly_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Create yearly change chart
        yearly_fig = go.Figure()
        if not yearly_df.empty:
            yearly_fig.add_trace(go.Scatter(
                x=yearly_df.index,
                y=yearly_df['Yıllık Enflasyon'],
                mode='lines',
                name='Yıllık Değişim',
                line=dict(
                    color='#F97316',
                    width=3
                ),
                hovertemplate='<b>%{customdata[0]}</b><br>Yıllık Değişim: %{y:+.2f}%<extra></extra>',
                customdata=[[f"{date.strftime('%Y-%m-%d')}"] for date in yearly_df.index]
            ))
        
        yearly_fig.update_layout(
            title=dict(
                text='Yıllık Değişim (%)',
                font=dict(
                    size=24,
                    family='Inter, sans-serif',
                    color='#2B2D42'
                ),
                y=0.95
            ),
            xaxis=dict(
                title='Tarih',
                title_font=dict(size=14, family='Inter, sans-serif', color='#2B2D42'),
                tickfont=dict(size=12, family='Inter, sans-serif', color='#2B2D42'),
                gridcolor='#E9ECEF',
                zerolinecolor='#E9ECEF',
                tickformat='%Y-%m-%d',
                tickangle=45
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
                borderwidth=1
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
                align='left',
                namelength=-1
            ),
            hoverdistance=100
        )
        
        yearly_graphJSON = json.dumps(yearly_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Prepare item data for popup charts
        item_names = [col for col in df.columns if col != 'Açlık Sınırı']
        item_charts = {}
        
        for item in item_names:
            item_fig = go.Figure()
            item_fig.add_trace(go.Scatter(
                x=df.index,
                y=df[item],
                mode='lines',
                name=item,
                line=dict(color='#6366F1', width=2),
                hovertemplate='<b>%{customdata[0]}</b><br>' + item + ': %{y:,.2f} TL<extra></extra>',
                customdata=[[f"{date.strftime('%Y-%m-%d')}"] for date in df.index]
            ))
            item_fig.update_layout(
                title=dict(text=item, font=dict(size=14, family='Inter, sans-serif')),
                xaxis=dict(
                    title='Tarih', 
                    title_font=dict(size=10),
                    tickfont=dict(size=9),
                    tickformat='%Y-%m',
                    dtick='M1',  # Her ay için bir tick
                    gridcolor='#E9ECEF',
                    zerolinecolor='#E9ECEF',
                    tickangle=45
                ),
                yaxis=dict(
                    title='Fiyat (TL)', 
                    title_font=dict(size=10),
                    tickfont=dict(size=9),
                    gridcolor='#E9ECEF'
                ),
                height=250,
                margin=dict(l=40, r=20, t=40, b=40),
                plot_bgcolor='white',
                paper_bgcolor='white',
                hovermode='x unified',
                hoverlabel=dict(
                    bgcolor='white',
                    font_size=11,
                    font_family='Inter, sans-serif',
                    align='left',
                    namelength=-1
                ),
                hoverdistance=10
            )
            item_charts[item] = json.dumps(item_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Get latest values for display
        latest_date = df.index[-1]
        latest_values = {}
        for item in item_names:
            if item in df.columns:
                latest_values[item] = df.loc[latest_date, item]
        
        latest_aclik_siniri = df.loc[latest_date, 'Açlık Sınırı']
        
        return render_template('aclik_siniri.html',
                               is_custom_basket=is_custom_basket,
                             main_graphJSON=main_graphJSON,
                             monthly_graphJSON=monthly_graphJSON,
                             yearly_graphJSON=yearly_graphJSON,
                             item_names=item_names,
                             item_charts=item_charts,
                             quantities=quantities,
                             default_quantities=default_quantities,
                             units=units,
                             latest_values=latest_values,
                             latest_aclik_siniri=latest_aclik_siniri,
                             latest_date=latest_date,
                             active_page='aclik_siniri')
    
    except Exception as e:
        import traceback
        print(f"Error in aclik_siniri route: {str(e)}")
        print(traceback.format_exc())
        flash(f'Hata: {str(e)}', 'error')
        return render_template('aclik_siniri.html',
                               is_custom_basket=False,
                             main_graphJSON=None,
                             monthly_graphJSON=None,
                             yearly_graphJSON=None,
                             item_names=[],
                             item_charts={},
                             quantities={},
                             default_quantities={},
                             units={},
                             latest_values={},
                             latest_aclik_siniri=0,
                             latest_date=None,
                             active_page='aclik_siniri')

if __name__ == '__main__':
    app.run(debug=True) 