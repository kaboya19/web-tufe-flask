from flask import Flask, render_template, request, send_file, flash, redirect, url_for
import pandas as pd
import json
import plotly
import plotly.graph_objects as go
import plotly.utils
from datetime import datetime
import csv
from dateutil.parser import parse
import os
from gspread.exceptions import APIError, SpreadsheetNotFound
import base64
from oauth2client.service_account import ServiceAccountCredentials
import gspread

app = Flask(__name__)
app.secret_key = os.urandom(24).hex()  # Güvenli, rastgele bir secret key oluştur

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

@app.route('/ana-sayfa')
def ana_sayfa():
    try:
        # Get data from Google Sheets
        data_pairs, month_name = get_google_sheets_data()
        categories = [pair[0] for pair in data_pairs]
        values = [pair[1] for pair in data_pairs]
        
        # Get last update date
        last_update = get_last_update_date()
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        # Add bars for each category
        for i, (category, value) in enumerate(data_pairs):
            color = '#EF476F' if category == 'Web TÜFE' else '#118AB2'  # Modern color scheme
            fig.add_trace(go.Bar(
                y=[category],
                x=[value],
                orientation='h',
                marker_color=color,
                name=category,
                marker=dict(
                    line=dict(width=0)
                ),
                text = [f'<b>{value:+.2f}%</b>'],
                textposition='outside',
                textfont=dict(
                    size=15,
                    color='#2B2D42',
                    family='Inter, sans-serif'
                ),
                hovertemplate=f'{category}: %{{x:+.2f}}%<extra></extra>'
            ))
        
        valid_values = [v for _, v in data_pairs if v is not None]

        if valid_values:
            x_min = min(valid_values)
            x_max = max(valid_values)

            # Marj hesapla
            x_range = x_max - x_min
            x_margin = x_range * 0.2 if x_range != 0 else abs(x_max) * 0.2

            x_min_with_margin = x_min - x_margin
            x_max_with_margin = x_max + x_margin

            # Sıfıra yaklaşma kontrolü
            if x_min >= 0:
                x_min_with_margin = max(0, x_min - x_margin)
            if x_max <= 0:
                x_max_with_margin = min(0, x_max + x_margin)

        # Update layout with modern theme
        fig.update_layout(
            title=dict(
                text=f'Web TÜFE {month_name} Ayı Ana Grup Artış Oranları',
                font=dict(
                    size=24,
                    family='Inter, sans-serif',
                    color='#2B2D42'
                ),
                y=0.95
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
                range=[x_min_with_margin, x_max_with_margin]
            ),
            yaxis=dict(
                title='Grup',
                title_font=dict(
                    size=14,
                    family='Inter, sans-serif',
                    color='#2B2D42'
                ),
                tickfont=dict(
                    size=14,
                    family='Arial Black, sans-serif',
                    color='#2B2D42'
                ),
                gridcolor='#E9ECEF'
            ),
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=600,
            margin=dict(l=20, r=20, t=80, b=20),
            hovermode='y unified',
            hoverlabel=dict(
                bgcolor='white',
                font_size=12,
                font_family='Inter, sans-serif',
                font_color='#2B2D42'
            )
        )
        
        # Convert plot to JSON
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return render_template('index.html', graphJSON=graphJSON, active_page='ana_sayfa', last_update=last_update)
    except Exception as e:
        flash(f'Bir hata oluştu: {str(e)}', 'error')
        return render_template('index.html')

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
    df_madde=pd.read_csv("endeksler_int.csv").rename(columns={"Unnamed: 0":"Tarih"})
    madde_names = df_madde.columns[1:].tolist()  # Get column names as madde names
    
    selected_madde = request.form.get('madde') if request.method == 'POST' else 'TÜFE'
    
    if selected_madde == 'TÜFE':
        # Filter dates to show only first day of each month
        df['month'] = df['Tarih'].dt.to_period('M')
        first_days = df.groupby('month').first()
        
        # Create line plot
        fig = go.Figure()
        
        # Add TÜFE line
        fig.add_trace(go.Scatter(
            x=df['Tarih'],
            y=df['Web TÜFE'],
            mode='lines',
            name='TÜFE',
            line=dict(
                color='#EF476F',
                width=3
            ),
            hovertemplate='%{customdata[0]}<br>TÜFE: %{customdata[1]:+.2f}%' + '<extra></extra>',
            customdata=[[f"{date.strftime('%d')} {get_turkish_month(date.strftime('%Y-%m-%d'))} {date.strftime('%Y')}", y-100] for date, y in zip(df['Tarih'], df['Web TÜFE'])]
        ))
        
        # Update layout with modern theme
        fig.update_layout(
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
            selected_date_obj = datetime.strptime(selected_date, '%Y-%m')
            turkish_month = get_turkish_month(selected_date_obj.strftime('%Y-%m-%d'))
        except Exception:
            turkish_month = "selected_date"
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
                hovertemplate='%{x}<br>Değişim: %{y:.2f}%<extra></extra>'
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
                height=400,
                margin=dict(l=10, r=10, t=40, b=20),
                hovermode='x'
            )
            
            bar_graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            # Aylık değişim line grafiği
            line_fig = go.Figure()
            line_fig.add_trace(go.Scatter(
                x=monthly_dates,
                y=monthly_changes,
                mode='lines+markers',
                name=selected_madde,
                line=dict(color='#EF476F', width=3),
                marker=dict(size=8, color='#EF476F'),
                hovertemplate='%{x}<br>Değişim: %{y:.2f}%<extra></extra>'
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
                    gridcolor='#E9ECEF'
                ),
                showlegend=True,
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=400,
                margin=dict(l=10, r=10, t=40, b=20),
                hovermode='x'
            )
            line_graphJSON = line_fig.to_json()
            
            return render_template('tufe.html',
                graphJSON=graphJSON,
                last_date=endeks_dates[-1] if not endeks_dates.empty else None,
                change_rate=total_change,
                month_name=get_turkish_month(endeks_dates[-1].strftime('%Y-%m-%d')) if not endeks_dates.empty else None,
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
    grup_adlari = [col for col in df.columns if col != 'Tarih']
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
    fig = go.Figure()
    customdata = [f"{d.day} {get_turkish_month(d.strftime('%Y-%m-%d'))} {d.year}" for d in tarih]

    fig.add_trace(go.Scatter(
        x=tarih,
        y=values,
        mode='lines',
        name=selected_group,
        line=dict(color='#EF476F', width=3),
        customdata=customdata,
        hovertemplate='<b>%{customdata}</b><br>' + f'{selected_group}: ' + '%{y:.2f}<extra></extra>'
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
        margin=dict(l=10, r=10, t=40, b=20),
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
    xaxis_range = [x_min * 1.4 if x_min < 0 else x_min * 1.4, x_max * 1.4 if x_max > 0 else x_max * 1.4]
    fig = go.Figure(go.Bar(
        y=bar_labels,
        x=bar_values,
        orientation='h',
        marker_color=bar_colors,
        cliponaxis=False,
        hovertemplate='%{y}: %{x:.2f}<extra></extra>'
    ))
    offset = (x_max - x_min) * 0.01 if bar_values else 0.2
    for i, value in enumerate(bar_values):
        fig.add_annotation(
            x=value + offset if value>=0 else value-offset*12.5,
            y=bar_labels[i],
            text=f"<b>{value:.2f}%</b>",
            showarrow=False,
            font=dict(size=15, family="Inter, sans-serif", color="#2B2D42"),
            align='left',
            xanchor='left',
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
                son_ay = get_turkish_month(last_date.strftime('%Y-%m-%d')) + f" {last_date.year}"
                harcama_grubu_total_change = values.iloc[-1] - values.iloc[0]
                # --- Fix: Son ay değişimi 1927818004 ID'li tablodan alınacak ---
                """worksheet_harcama = spreadsheet.get_worksheet_by_id(1927818004)
                data_harcama = worksheet_harcama.get_all_values()
                df_harcama = pd.DataFrame(data_harcama[1:], columns=data_harcama[0])"""
                df_harcama=pd.read_csv("harcama_gruplarıaylık.csv",index_col=0)
                df_harcama[df_harcama.columns[0]] = df_harcama[df_harcama.columns[0]].str.strip().str.lower()
                row = df_harcama[df_harcama.iloc[:,0] == selected_norm]
                harcama_grubu_monthly_change = None
                if not row.empty:
                    last_col = df_harcama.columns[-1]
                    try:
                        harcama_grubu_monthly_change = float(str(row[last_col].values[0]).replace(',', '.'))
                    except:
                        harcama_grubu_monthly_change = None
                # --- End Fix ---
                fig_endeks = go.Figure()
                fig_endeks.add_trace(go.Scatter(
                    x=dates,
                    y=values,
                    mode='lines',
                    name=selected_harcama_grubu.title(),
                    line=dict(color='#EF476F', width=3),
                    marker=dict(size=8, color='#EF476F'),
                    hovertemplate='%{x|%d.%m.%Y}<br>Endeks: %{y:.2f}<extra></extra>'
                ))
                fig_endeks.update_layout(
                    title=dict(
                        text=f'{selected_harcama_grubu.title()} Endeksi',
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
                        tickvals=tickvals,
                        ticktext=ticktext
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
                    showlegend=False,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    height=400,
                    margin=dict(l=10, r=10, t=40, b=20),
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
                                margin=dict(l=10, r=10, t=40, b=20),
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
                                margin=dict(l=10, r=10, t=40, b=20),
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
        monthly_line_graphJSON=monthly_line_graphJSON if selected_harcama_grubu else None
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
    x_min = min(valid_bar_values + [0])
    x_max = max(valid_bar_values + [0])
    xaxis_range = [x_min * 1.4 if x_min < 0 else x_min * 1.4, x_max * 1.4 if x_max > 0 else x_max * 1.4]
    fig = go.Figure(go.Bar(
        y=bar_labels,
        x=bar_values,
        orientation='h',
        marker_color=bar_colors,
        cliponaxis=False,
        hovertemplate='%{y}: %{x:.2f}<extra></extra>'
    ))
    offset = (x_max - x_min) * 0.01 if bar_values else 0.2
    for i, value in enumerate(bar_values):
        fig.add_annotation(
            x=value + offset/2 if value>=0 else value-offset*7,
            y=bar_labels[i],
            text=f"<b>{value:.2f}%</b>",
            showarrow=False,
            font=dict(size=15, family="Inter Bold, Inter, sans-serif", color="#2B2D42"),
            align='left',
            xanchor='left',
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
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines',
            name=selected_indicator,
            line=dict(color='#EF476F', width=3),
            customdata=turkish_dates,
            hovertemplate='<b>%{customdata}</b><br>' + f'{selected_indicator}: ' + '%{y:.2f}<extra></extra>'
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
            showlegend=False,
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
        hovermode='x'
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
            gridcolor='#E9ECEF'
        ),
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400,
        margin=dict(l=10, r=10, t=40, b=20),
        hovermode='x'
    )

    bar_graphJSON = json.dumps(bar_fig, cls=plotly.utils.PlotlyJSONEncoder)
    line_graphJSON = json.dumps(line_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('ozel_kapsamli_gostergeler.html',
    graphJSON=graphJSON,
    indicator_names=indicator_names,
    selected_indicator=selected_indicator,
    total_change=total_change,
    monthly_change=monthly_change,
    active_page='ozel_kapsamli_gostergeler',
    last_date=dates.iloc[-1] if not df.empty else None,
    month_name=get_turkish_month(dates.iloc[-1].strftime('%Y-%m-%d')) if not df.empty else None,
    bar_graphJSON=bar_graphJSON,
    line_graphJSON=line_graphJSON
)

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

if __name__ == '__main__':
    app.run(debug=True) 