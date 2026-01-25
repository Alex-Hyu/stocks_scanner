"""
股票波段期权筛选系统 - 最终版
整合：ETF板块资金流（参考） + 个股技术筛选 + SpotGamma交叉验证

运行方式: streamlit run app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Google Sheets集成
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSHEETS_AVAILABLE = True
except ImportError:
    GSHEETS_AVAILABLE = False

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="股票波段期权筛选系统",
    page_icon="🎯",
    layout="wide"
)

# ============================================================
# Squeeze追踪配置
# ============================================================
TRACKING_FILE = "./squeeze_tracking.json"
SQUEEZE_THRESHOLD = 5.0  # 5%涨幅算squeeze确认

# Google Sheets配置
GSHEETS_CREDENTIALS_FILE = "./google_credentials.json"  # 你的API凭证文件
GSHEETS_SPREADSHEET_NAME = "SpotGamma_Tracking"  # Google Sheets文档名称
GSHEETS_WORKSHEET_NAME = "tracking_data"  # 工作表名称

# ============================================================
# 常量定义
# ============================================================

SECTOR_ETFS = {
    'XLK': '科技',
    'SMH': '半导体',
    'XLF': '金融',
    'XLE': '能源',
    'XLV': '医疗健康',
    'XBI': '生物科技',
    'XLI': '工业',
    'XLY': '可选消费',
    'XLP': '必需消费',
    'XLU': '公用事业',
    'XLRE': '房地产',
    'XLB': '材料',
    'XLC': '通信服务',
    'IWM': '小盘股',
}

# 板块关键词映射（用于匹配股票所属板块）
SECTOR_KEYWORDS = {
    '科技': ['Technology', 'Software', 'Internet', 'Electronics', 'Computer'],
    '半导体': ['Semiconductor', 'Chip'],
    '金融': ['Financial', 'Bank', 'Insurance', 'Investment', 'Capital'],
    '能源': ['Energy', 'Oil', 'Gas', 'Petroleum', 'Solar', 'Wind'],
    '医疗健康': ['Healthcare', 'Pharmaceutical', 'Medical', 'Drug'],
    '生物科技': ['Biotechnology', 'Biotech', 'Genomics'],
    '工业': ['Industrial', 'Manufacturing', 'Aerospace', 'Defense', 'Machinery'],
    '可选消费': ['Consumer Cyclical', 'Retail', 'Auto', 'Restaurant', 'Apparel', 'Luxury'],
    '必需消费': ['Consumer Defensive', 'Food', 'Beverage', 'Household', 'Grocery'],
    '公用事业': ['Utilities', 'Electric', 'Water', 'Gas Utilities'],
    '房地产': ['Real Estate', 'REIT', 'Property'],
    '材料': ['Materials', 'Chemical', 'Mining', 'Steel', 'Metals'],
    '通信服务': ['Communication', 'Telecom', 'Media', 'Entertainment', 'Advertising'],
}

# Nasdaq 100 成分股 (2024)
NASDAQ_100 = [
    'AAPL', 'ABNB', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'AMAT', 'AMD', 'AMGN',
    'AMZN', 'ANSS', 'APP', 'ARM', 'ASML', 'AVGO', 'AZN', 'BIIB', 'BKNG', 'BKR',
    'CCEP', 'CDNS', 'CDW', 'CEG', 'CHTR', 'CMCSA', 'COST', 'CPRT', 'CRWD', 'CSCO',
    'CSGP', 'CSX', 'CTAS', 'CTSH', 'DASH', 'DDOG', 'DLTR', 'DXCM', 'EA', 'EXC',
    'FANG', 'FAST', 'FTNT', 'GEHC', 'GFS', 'GILD', 'GOOG', 'GOOGL', 'HON', 'IDXX',
    'ILMN', 'INTC', 'INTU', 'ISRG', 'KDP', 'KHC', 'KLAC', 'LIN', 'LRCX', 'LULU',
    'MAR', 'MCHP', 'MDB', 'MDLZ', 'MELI', 'META', 'MNST', 'MRNA', 'MRVL', 'MSFT',
    'MU', 'NFLX', 'NVDA', 'NXPI', 'ODFL', 'ON', 'ORLY', 'PANW', 'PAYX', 'PCAR',
    'PDD', 'PEP', 'PYPL', 'QCOM', 'REGN', 'ROP', 'ROST', 'SBUX', 'SMCI', 'SNPS',
    'SPLK', 'TEAM', 'TMUS', 'TSLA', 'TTD', 'TTWO', 'TXN', 'VRSK', 'VRTX', 'WBD',
    'WDAY', 'XEL', 'ZS'
]

# S&P 500 成分股 (2024)
SP_500 = [
    'A', 'AAL', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI',
    'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG',
    'AKAM', 'ALB', 'ALGN', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN',
    'AMP', 'AMT', 'AMZN', 'ANET', 'ANSS', 'AON', 'AOS', 'APA', 'APD', 'APH',
    'APTV', 'ARE', 'ATO', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXON', 'AXP', 'AZO',
    'BA', 'BAC', 'BALL', 'BAX', 'BBWI', 'BBY', 'BDX', 'BEN', 'BF.B', 'BG',
    'BIIB', 'BIO', 'BK', 'BKNG', 'BKR', 'BLDR', 'BLK', 'BMY', 'BR', 'BRK.B',
    'BRO', 'BSX', 'BWA', 'BX', 'BXP', 'C', 'CAG', 'CAH', 'CARR', 'CAT',
    'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CDW', 'CE', 'CEG', 'CF',
    'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMCSA', 'CME',
    'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COO', 'COP', 'COR', 'COST',
    'CPAY', 'CPB', 'CPRT', 'CPT', 'CRL', 'CRM', 'CRWD', 'CSCO', 'CSGP', 'CSX',
    'CTAS', 'CTLT', 'CTRA', 'CTSH', 'CTVA', 'CVS', 'CVX', 'CZR', 'D', 'DAL',
    'DAY', 'DD', 'DE', 'DECK', 'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS',
    'DLR', 'DLTR', 'DOC', 'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA',
    'DVN', 'DXCM', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EG', 'EIX', 'EL',
    'ELV', 'EMN', 'EMR', 'ENPH', 'EOG', 'EPAM', 'EQIX', 'EQR', 'EQT', 'ES',
    'ESS', 'ETN', 'ETR', 'ETSY', 'EVRG', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR',
    'F', 'FANG', 'FAST', 'FCX', 'FDS', 'FDX', 'FE', 'FFIV', 'FI', 'FICO',
    'FIS', 'FITB', 'FMC', 'FOX', 'FOXA', 'FRT', 'FSLR', 'FTNT', 'FTV', 'GD',
    'GDDY', 'GE', 'GEHC', 'GEN', 'GEV', 'GILD', 'GIS', 'GL', 'GLW', 'GM',
    'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS',
    'HBAN', 'HCA', 'HD', 'HES', 'HIG', 'HII', 'HLT', 'HOLX', 'HON', 'HPE',
    'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUBB', 'HUM', 'HWM', 'IBM', 'ICE',
    'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY', 'INTC', 'INTU', 'INVH', 'IP', 'IPG',
    'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JBL',
    'JCI', 'JKHY', 'JNJ', 'JNPR', 'JPM', 'K', 'KDP', 'KEY', 'KEYS', 'KHC',
    'KIM', 'KLAC', 'KMB', 'KMI', 'KMX', 'KO', 'KR', 'KVUE', 'L', 'LDOS',
    'LEN', 'LH', 'LHX', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNT', 'LOW', 'LRCX',
    'LULU', 'LUV', 'LVS', 'LW', 'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS',
    'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'META', 'MGM', 'MHK',
    'MKC', 'MKTX', 'MLM', 'MMC', 'MMM', 'MNST', 'MO', 'MOH', 'MOS', 'MPC',
    'MPWR', 'MRK', 'MRNA', 'MRO', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTCH',
    'MTD', 'MU', 'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE',
    'NOC', 'NOW', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR', 'NWS',
    'NWSA', 'NXPI', 'O', 'ODFL', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS',
    'OXY', 'PANW', 'PARA', 'PAYC', 'PAYX', 'PCAR', 'PCG', 'PEG', 'PEP', 'PFE',
    'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PLD', 'PLTR', 'PM', 'PNC',
    'PNR', 'PNW', 'PODD', 'POOL', 'PPG', 'PPL', 'PRU', 'PSA', 'PSX', 'PTC',
    'PWR', 'PYPL', 'QCOM', 'QRVO', 'RCL', 'REG', 'REGN', 'RF', 'RJF', 'RL',
    'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'RVTY', 'SBAC', 'SBUX',
    'SCHW', 'SHW', 'SJM', 'SLB', 'SMCI', 'SNA', 'SNPS', 'SO', 'SOLV', 'SPG',
    'SPGI', 'SRE', 'STE', 'STLD', 'STT', 'STX', 'STZ', 'SW', 'SWK', 'SWKS',
    'SYF', 'SYK', 'SYY', 'T', 'TAP', 'TDG', 'TDY', 'TECH', 'TEL', 'TER',
    'TFC', 'TFX', 'TGT', 'TJX', 'TMO', 'TMUS', 'TPR', 'TRGP', 'TRMB', 'TROW',
    'TRV', 'TSCO', 'TSLA', 'TSN', 'TT', 'TTWO', 'TXN', 'TXT', 'TYL', 'UAL',
    'UBER', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNP', 'UPS', 'URI', 'USB', 'V',
    'VFC', 'VICI', 'VLO', 'VLTO', 'VMC', 'VRSK', 'VRSN', 'VRTX', 'VST', 'VTR',
    'VTRS', 'VZ', 'WAB', 'WAT', 'WBA', 'WBD', 'WDC', 'WEC', 'WELL', 'WFC',
    'WM', 'WMB', 'WMT', 'WRB', 'WST', 'WTW', 'WY', 'WYNN', 'XEL', 'XOM',
    'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZTS'
]

def get_stock_pool(pool_name: str) -> list:
    """获取股票池"""
    if pool_name == "Nasdaq 100":
        return NASDAQ_100
    elif pool_name == "S&P 500":
        return SP_500
    elif pool_name == "Nasdaq 100 + S&P 500":
        return list(set(NASDAQ_100 + SP_500))
    else:
        return []


# ============================================================
# Squeeze追踪模块
# ============================================================

# ===== Google Sheets 集成函数 =====

def get_gsheets_client():
    """获取Google Sheets客户端"""
    if not GSHEETS_AVAILABLE:
        return None
    
    try:
        # 尝试从Streamlit secrets读取凭证
        if hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
            creds = Credentials.from_service_account_info(
                st.secrets['gcp_service_account'],
                scopes=[
                    'https://www.googleapis.com/auth/spreadsheets',
                    'https://www.googleapis.com/auth/drive'
                ]
            )
        # 尝试从本地文件读取凭证
        elif os.path.exists(GSHEETS_CREDENTIALS_FILE):
            creds = Credentials.from_service_account_file(
                GSHEETS_CREDENTIALS_FILE,
                scopes=[
                    'https://www.googleapis.com/auth/spreadsheets',
                    'https://www.googleapis.com/auth/drive'
                ]
            )
        else:
            return None
        
        return gspread.authorize(creds)
    except Exception as e:
        st.warning(f"Google Sheets连接失败: {e}")
        return None

def load_tracking_from_gsheets():
    """从Google Sheets加载追踪数据"""
    client = get_gsheets_client()
    if not client:
        return None
    
    try:
        # 尝试打开文档
        try:
            spreadsheet = client.open(GSHEETS_SPREADSHEET_NAME)
        except gspread.exceptions.SpreadsheetNotFound:
            st.warning(f"⚠️ 找不到Google Sheets文档 '{GSHEETS_SPREADSHEET_NAME}'。请先创建此文档并共享给Service Account。")
            return None
        
        # 尝试获取工作表，如果不存在则创建
        try:
            worksheet = spreadsheet.worksheet(GSHEETS_WORKSHEET_NAME)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=GSHEETS_WORKSHEET_NAME, rows=1000, cols=30)
            # 创建表头
            headers = ['symbol', 'data_json']
            worksheet.append_row(headers)
            return {}
        
        # 读取所有数据
        all_values = worksheet.get_all_values()
        
        # 如果只有表头或空表
        if len(all_values) <= 1:
            return {}
        
        # 解析数据（跳过表头）
        tracking_data = {}
        headers = all_values[0]
        
        # 找到symbol和data_json列的索引
        try:
            symbol_idx = headers.index('symbol')
            data_idx = headers.index('data_json')
        except ValueError:
            # 表头不正确，重新初始化
            worksheet.clear()
            worksheet.append_row(['symbol', 'data_json'])
            return {}
        
        for row in all_values[1:]:
            if len(row) > max(symbol_idx, data_idx):
                symbol = row[symbol_idx]
                data_json = row[data_idx]
                if symbol and data_json:
                    try:
                        tracking_data[symbol] = json.loads(data_json)
                    except json.JSONDecodeError:
                        pass
        
        return tracking_data
        
    except gspread.exceptions.APIError as e:
        st.warning(f"Google Sheets API错误: {e}")
        return None
    except Exception as e:
        st.warning(f"从Google Sheets加载数据失败: {type(e).__name__}: {e}")
        return None

def save_tracking_to_gsheets(tracking_data):
    """保存追踪数据到Google Sheets"""
    client = get_gsheets_client()
    if not client:
        return False
    
    try:
        spreadsheet = client.open(GSHEETS_SPREADSHEET_NAME)
        
        # 尝试获取工作表，如果不存在则创建
        try:
            worksheet = spreadsheet.worksheet(GSHEETS_WORKSHEET_NAME)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=GSHEETS_WORKSHEET_NAME, rows=1000, cols=30)
        
        # 清空现有数据（保留表头）
        worksheet.clear()
        
        # 写入表头
        headers = ['symbol', 'data_json']
        worksheet.append_row(headers)
        
        # 写入数据
        rows = []
        for symbol, data in tracking_data.items():
            data_json = json.dumps(data, ensure_ascii=False, default=str)
            rows.append([symbol, data_json])
        
        if rows:
            worksheet.append_rows(rows)
        
        return True
    except Exception as e:
        st.warning(f"保存到Google Sheets失败: {e}")
        return False

def load_worksheet_data(ws_name):
    """从指定worksheet加载数据（通用函数）"""
    client = get_gsheets_client()
    if not client:
        return None
    
    try:
        spreadsheet = client.open(GSHEETS_SPREADSHEET_NAME)
        
        # 尝试获取工作表，如果不存在则创建
        try:
            worksheet = spreadsheet.worksheet(ws_name)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=ws_name, rows=1000, cols=30)
            worksheet.append_row(['key', 'data_json'])
            return {}
        
        all_values = worksheet.get_all_values()
        
        if len(all_values) <= 1:
            return {}
        
        headers = all_values[0]
        data = {}
        
        try:
            key_idx = headers.index('key') if 'key' in headers else headers.index('symbol')
            data_idx = headers.index('data_json')
        except ValueError:
            worksheet.clear()
            worksheet.append_row(['key', 'data_json'])
            return {}
        
        for row in all_values[1:]:
            if len(row) > max(key_idx, data_idx):
                key = row[key_idx]
                data_json = row[data_idx]
                if key and data_json:
                    try:
                        data[key] = json.loads(data_json)
                    except json.JSONDecodeError:
                        pass
        
        return data
    except Exception as e:
        return None

def save_worksheet_data(ws_name, data):
    """保存数据到指定worksheet（通用函数）"""
    client = get_gsheets_client()
    if not client:
        return False
    
    try:
        spreadsheet = client.open(GSHEETS_SPREADSHEET_NAME)
        
        try:
            worksheet = spreadsheet.worksheet(ws_name)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=ws_name, rows=1000, cols=30)
        
        worksheet.clear()
        worksheet.append_row(['key', 'data_json'])
        
        rows = []
        for key, record in data.items():
            data_json = json.dumps(record, ensure_ascii=False, default=str)
            rows.append([key, data_json])
        
        if rows:
            worksheet.append_rows(rows)
        
        return True
    except Exception as e:
        return False

def sync_tracking_data():
    """同步追踪数据（Google Sheets优先，本地JSON作为备份）"""
    # 1. 尝试从Google Sheets加载
    gsheets_data = load_tracking_from_gsheets()
    
    # 2. 加载本地JSON
    local_data = {}
    if os.path.exists(TRACKING_FILE):
        try:
            with open(TRACKING_FILE, 'r', encoding='utf-8') as f:
                local_data = json.load(f)
        except:
            pass
    
    # 3. 合并数据（Google Sheets优先，但保留本地新增的记录）
    if gsheets_data is not None:
        # Google Sheets可用，以它为主
        merged_data = gsheets_data.copy()
        
        # 检查本地是否有新增记录（Google Sheets中没有的）
        for symbol, record in local_data.items():
            if symbol not in merged_data:
                merged_data[symbol] = record
        
        return merged_data, True  # 返回数据和是否连接成功
    else:
        # Google Sheets不可用，使用本地数据
        return local_data, False

# ===== 原有的本地存储函数（作为备份）=====

def load_tracking_data():
    """加载追踪数据（优先从Google Sheets，失败则从本地）"""
    # 尝试同步
    data, gsheets_connected = sync_tracking_data()
    
    # 存储连接状态到session_state
    if 'gsheets_connected' not in st.session_state:
        st.session_state.gsheets_connected = gsheets_connected
    else:
        st.session_state.gsheets_connected = gsheets_connected
    
    return data

def save_tracking_data(data):
    """保存追踪数据（同时保存到Google Sheets和本地JSON）"""
    # 1. 保存到本地JSON（作为备份）
    try:
        with open(TRACKING_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        st.warning(f"本地保存失败: {e}")
    
    # 2. 保存到Google Sheets
    gsheets_success = save_tracking_to_gsheets(data)
    
    return gsheets_success

def get_current_price(symbol):
    """获取当前价格"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except:
        pass
    return None

def get_price_history(symbol, start_date, end_date=None):
    """获取历史价格"""
    try:
        ticker = yf.Ticker(symbol)
        if end_date:
            hist = ticker.history(start=start_date, end=end_date)
        else:
            hist = ticker.history(start=start_date)
        return hist
    except:
        return None

def update_tracking_record(symbol, tracking_data, current_price):
    """更新单个追踪记录"""
    if symbol not in tracking_data:
        return None
    
    record = tracking_data[symbol]
    today = datetime.now().strftime('%Y-%m-%d')
    
    # 更新每日价格
    if 'daily_prices' not in record:
        record['daily_prices'] = {}
    
    if current_price:
        record['daily_prices'][today] = current_price
    
    # 计算指标
    entry_price = record.get('entry_price', 0)
    if entry_price > 0 and current_price:
        prices = list(record['daily_prices'].values())
        
        # 当前涨幅（从D0到当前价格）
        record['current_return'] = ((current_price - entry_price) / entry_price * 100)
        
        # 最大涨幅
        record['max_gain'] = max([(p - entry_price) / entry_price * 100 for p in prices]) if prices else 0
        
        # 最大回撤
        record['max_drawdown'] = min([(p - entry_price) / entry_price * 100 for p in prices]) if prices else 0
        
        # 判断是否确认squeeze（当前涨幅>=5%就确认）
        record['squeeze_confirmed'] = record['current_return'] >= SQUEEZE_THRESHOLD
        
        # 判断信号方向是否正确
        signal_direction = record.get('signal_direction', 'neutral')
        current_return = record['current_return']
        
        if signal_direction == 'bullish':
            # 多头信号：涨了就正确
            record['direction_correct'] = current_return > 0
        elif signal_direction == 'bearish':
            # 空头信号：跌了就正确
            record['direction_correct'] = current_return < 0
        else:
            # 中性信号：不判断
            record['direction_correct'] = None
    
    # 检查是否到达追踪结束日期
    track_end = record.get('track_end_date')
    if track_end:
        try:
            end_date = datetime.strptime(track_end, '%Y-%m-%d')
            if datetime.now() > end_date:
                record['status'] = 'completed'
        except:
            pass
    
    return record

def add_new_tracking(symbol, row, signal_type, today_str):
    """添加新的追踪记录"""
    # 解析到期日
    top_gamma_exp = row.get('Top Gamma Exp', '')
    try:
        if isinstance(top_gamma_exp, str) and top_gamma_exp:
            exp_date = datetime.strptime(top_gamma_exp, '%Y-%m-%d')
            # 到期日+2个交易日（约4个自然日）
            track_end = (exp_date + timedelta(days=4)).strftime('%Y-%m-%d')
        else:
            # 默认7天后结束追踪
            track_end = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
    except:
        track_end = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
    
    # 判断信号方向 - 使用Signal_Type字段
    sig_type = row.get('Signal_Type', '')
    if sig_type in ['bullish', 'bullish_watch']:
        signal_direction = 'bullish'
    elif sig_type in ['bearish', 'bearish_watch']:
        signal_direction = 'bearish'
    else:
        signal_direction = 'neutral'
    
    return {
        'signal_date': today_str,
        'entry_price': float(row['Current Price']),
        'top_gamma_exp': str(top_gamma_exp) if top_gamma_exp else '',
        'track_end_date': track_end,
        'signal_type': signal_type,
        'signal_direction': signal_direction,
        'vol_regime': row.get('Vol_Regime', '未知'),
        'delta_ratio': float(row.get('Delta Ratio', 0)),
        'gamma_ratio': float(row.get('Gamma Ratio', 0)),
        'volume_ratio': float(row.get('Volume Ratio', 0)) if pd.notna(row.get('Volume Ratio')) else 0,
        'next_exp_gamma': float(row.get('Next Exp Gamma', 0)) if pd.notna(row.get('Next Exp Gamma')) else 0,
        'options_impact': float(row.get('Options Impact', 0)) if pd.notna(row.get('Options Impact')) else 0,
        'put_wall': float(row.get('Put Wall', 0)),
        'call_wall': float(row.get('Call Wall', 0)),
        'hedge_wall': float(row.get('Hedge Wall', 0)) if pd.notna(row.get('Hedge Wall')) else 0,
        'cw_increase': row.get('CW_Increase', False),
        'daily_prices': {today_str: float(row['Current Price'])},
        'current_return': 0,
        'max_gain': 0,
        'max_drawdown': 0,
        'squeeze_confirmed': False,
        'direction_correct': None,
        'status': 'tracking',
        'is_new': True
    }

def calculate_tracking_stats(tracking_data):
    """计算追踪统计"""
    tracking_count = 0
    completed_count = 0
    squeeze_count = 0
    failed_count = 0
    
    for symbol, record in tracking_data.items():
        status = record.get('status', 'tracking')
        current_return = record.get('current_return', 0)
        squeeze_confirmed = current_return >= SQUEEZE_THRESHOLD  # 当前涨幅>=5%就确认
        
        if status == 'tracking':
            tracking_count += 1
            if squeeze_confirmed:
                squeeze_count += 1
        elif status == 'completed':
            completed_count += 1
            if squeeze_confirmed:
                squeeze_count += 1
            else:
                failed_count += 1
    
    win_rate = (squeeze_count / completed_count * 100) if completed_count > 0 else 0
    
    return {
        'tracking': tracking_count,
        'completed': completed_count,
        'squeeze': squeeze_count,
        'failed': failed_count,
        'win_rate': win_rate
    }

def calculate_signal_accuracy_stats(tracking_data):
    """计算信号方向正确率统计 - 实时计算"""
    stats = {
        'bullish': {'total': 0, 'correct': 0, 'wrong': 0},
        'bearish': {'total': 0, 'correct': 0, 'wrong': 0},
        'neutral': {'total': 0},
        'overall': {'total': 0, 'correct': 0, 'wrong': 0}
    }
    
    for symbol, record in tracking_data.items():
        direction = record.get('signal_direction', '')
        signal_type_text = record.get('signal_type', '')
        
        # 如果没有signal_direction字段（旧记录），从signal_type文本推断
        # 【更新】添加新的信号关键词
        if not direction or direction == 'neutral':
            # 做多信号关键词
            bullish_keywords = [
                '做多', '反弹', '偏多', 'bullish', 
                'Squeeze Up', '突破CW', 'PW支撑区', '弹簧蓄势', '正Gamma轧空'
            ]
            # 做空信号关键词
            bearish_keywords = [
                '做空', '压力', '偏空', '破位', 'bearish',
                'Squeeze Down', '跌破PW', 'CW阻力区', '负Gamma螺旋'
            ]
            
            if any(x in signal_type_text for x in bullish_keywords):
                direction = 'bullish'
            elif any(x in signal_type_text for x in bearish_keywords):
                direction = 'bearish'
            else:
                direction = 'neutral'
        
        # 实时计算涨跌幅
        entry_price = record.get('entry_price', 0)
        daily_prices = record.get('daily_prices', {})
        if daily_prices and entry_price > 0:
            latest_date = max(daily_prices.keys())
            current_price = daily_prices[latest_date]
            current_return = ((current_price - entry_price) / entry_price) * 100
        else:
            current_return = 0
        
        # 实时判断方向正确性
        if direction == 'bullish':
            stats['bullish']['total'] += 1
            stats['overall']['total'] += 1
            if current_return > 0:
                stats['bullish']['correct'] += 1
                stats['overall']['correct'] += 1
            else:
                stats['bullish']['wrong'] += 1
                stats['overall']['wrong'] += 1
        elif direction == 'bearish':
            stats['bearish']['total'] += 1
            stats['overall']['total'] += 1
            if current_return < 0:
                stats['bearish']['correct'] += 1
                stats['overall']['correct'] += 1
            else:
                stats['bearish']['wrong'] += 1
                stats['overall']['wrong'] += 1
        else:
            stats['neutral']['total'] += 1
    
    # 计算正确率
    for key in ['bullish', 'bearish', 'overall']:
        total = stats[key]['total']
        stats[key]['accuracy'] = (stats[key]['correct'] / total * 100) if total > 0 else 0
    
    return stats


# ============================================================
# ETF板块资金流扫描模块
# ============================================================

@st.cache_data(ttl=300)
def get_etf_data(ticker: str, period: str = "3mo"):
    """获取ETF数据"""
    try:
        data = yf.download(ticker, period=period, progress=False)
        return data
    except:
        return None


def analyze_etf_flow(ticker: str, data: pd.DataFrame) -> dict:
    """分析单个ETF的资金流入信号"""
    try:
        if data is None or data.empty or len(data) < 25:
            return None
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        df = data.copy()
        
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        df['Vol_SMA20'] = df['Volume'].rolling(20).mean()
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        latest = df.iloc[-1]
        prev_5d = df.iloc[-5]
        prev_20d = df.iloc[-20] if len(df) >= 20 else df.iloc[0]
        
        close = float(latest['Close'])
        sma20 = float(latest['SMA20'])
        sma50 = float(latest['SMA50'])
        volume = float(latest['Volume'])
        vol_sma20 = float(latest['Vol_SMA20'])
        obv_now = float(latest['OBV'])
        obv_5d_ago = float(prev_5d['OBV'])
        
        price_above_sma20 = close > sma20
        price_above_sma50 = close > sma50
        volume_expanding = volume > vol_sma20
        obv_rising = obv_now > obv_5d_ago
        returns_20d = (close / float(prev_20d['Close']) - 1) * 100
        vol_ratio = volume / vol_sma20 if vol_sma20 > 0 else 1
        
        score = sum([price_above_sma20, price_above_sma50, volume_expanding, obv_rising, returns_20d > 0])
        
        # 资金流状态判断
        if score >= 4:
            flow_status = "流入"
        elif score <= 2:
            flow_status = "流出"
        else:
            flow_status = "中性"
        
        return {
            'ETF': ticker,
            '板块': SECTOR_ETFS.get(ticker, ticker),
            '价格': round(close, 2),
            '>SMA20': '✅' if price_above_sma20 else '❌',
            '>SMA50': '✅' if price_above_sma50 else '❌',
            '放量': '✅' if volume_expanding else '❌',
            'OBV↑': '✅' if obv_rising else '❌',
            '量比': round(vol_ratio, 2),
            '20日涨幅%': round(returns_20d, 2),
            '评分': score,
            '资金流状态': flow_status,
        }
    except:
        return None


def scan_etf_flows():
    """扫描所有板块ETF"""
    results = []
    for ticker in SECTOR_ETFS.keys():
        data = get_etf_data(ticker)
        if data is not None:
            result = analyze_etf_flow(ticker, data)
            if result:
                results.append(result)
    
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('评分', ascending=False)
        return df
    return None


def get_sector_flow_status(etf_df: pd.DataFrame) -> dict:
    """从ETF数据生成板块资金流状态字典"""
    if etf_df is None:
        return {}
    
    status_dict = {}
    for _, row in etf_df.iterrows():
        status_dict[row['板块']] = row['资金流状态']
    
    return status_dict


# ============================================================
# 个股技术筛选模块 (Level 0-4)
# ============================================================

@st.cache_data(ttl=300)
def get_stock_data(ticker: str, period: str = "6mo"):
    """获取个股数据"""
    try:
        data = yf.download(ticker, period=period, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except:
        return None


@st.cache_data(ttl=3600)
def get_stock_info(ticker: str):
    """获取股票基本信息"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'marketCap': info.get('marketCap', 0),
            'shortName': info.get('shortName', ticker),
        }
    except:
        return {'sector': 'Unknown', 'industry': 'Unknown', 'marketCap': 0, 'shortName': ticker}


def level_0_filter(df: pd.DataFrame, ticker: str) -> tuple:
    """Level 0: 基础过滤"""
    if df is None or df.empty or len(df) < 50:
        return False, "数据不足"
    
    latest = df.iloc[-1]
    close = float(latest['Close'])
    
    if close < 10:
        return False, f"股价过低: ${close:.2f}"
    
    df['DollarVol'] = df['Close'] * df['Volume']
    avg_dollar_vol = df['DollarVol'].rolling(20).mean().iloc[-1]
    
    if avg_dollar_vol < 10_000_000:
        return False, f"成交额不足: ${avg_dollar_vol/1e6:.1f}M"
    
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    atr_pct = float(df['ATR'].iloc[-1] / close)
    
    if atr_pct < 0.02:
        return False, f"波动不足: ATR {atr_pct:.1%}"
    
    return True, "通过"


def level_1_classify(df: pd.DataFrame) -> dict:
    """Level 1: 市场状态分类"""
    df['EMA20'] = ta.ema(df['Close'], length=20)
    df['EMA50'] = ta.ema(df['Close'], length=50)
    df['EMA200'] = ta.ema(df['Close'], length=200)
    
    latest = df.iloc[-1]
    close = float(latest['Close'])
    ema20 = float(latest['EMA20'])
    ema50 = float(latest['EMA50'])
    ema200 = float(latest['EMA200']) if not pd.isna(latest['EMA200']) else ema50
    
    if ema20 > ema50 > ema200:
        if close > ema20:
            trend = "强多头"
        else:
            trend = "多头回调"
    elif ema20 < ema50 < ema200:
        if close < ema20:
            trend = "强空头"
        else:
            trend = "空头反弹"
    else:
        trend = "震荡"
    
    if len(df) >= 10:
        ema20_10d_ago = float(df['EMA20'].iloc[-10])
        trend_strength = (ema20 - ema20_10d_ago) / ema20
    else:
        trend_strength = 0
    
    return {
        'trend': trend,
        'trend_strength': trend_strength,
        'close': close,
        'ema20': ema20,
        'ema50': ema50,
        'ema200': ema200,
    }


def level_2_3_signals(df: pd.DataFrame, trend_info: dict) -> tuple:
    """Level 2 & 3: 核心信号检测"""
    signals = []
    direction = "中性"  # 信号方向：看多/看空/中性
    
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['ATR_Pct'] = df['ATR'] / df['Close']
    
    # 布林带 - 兼容不同版本的pandas_ta
    bb = ta.bbands(df['Close'], length=20, std=2.0)
    if bb is not None and not bb.empty:
        bb_cols = bb.columns.tolist()
        # 查找包含BBU/BBL/BBM的列名
        bbu_col = [c for c in bb_cols if 'BBU' in c]
        bbl_col = [c for c in bb_cols if 'BBL' in c]
        bbm_col = [c for c in bb_cols if 'BBM' in c]
        if bbu_col and bbl_col and bbm_col:
            df['BB_Upper'] = bb[bbu_col[0]]
            df['BB_Lower'] = bb[bbl_col[0]]
            df['BB_Mid'] = bb[bbm_col[0]]
    
    # 肯特纳通道 - 兼容不同版本
    kc = ta.kc(df['High'], df['Low'], df['Close'], length=20, scalar=1.5)
    if kc is not None and not kc.empty:
        kc_cols = kc.columns.tolist()
        kcu_col = [c for c in kc_cols if 'KCU' in c]
        kcl_col = [c for c in kc_cols if 'KCL' in c]
        if kcu_col and kcl_col:
            df['KC_Upper'] = kc[kcu_col[0]]
            df['KC_Lower'] = kc[kcl_col[0]]
    
    df['Vol_SMA20'] = df['Volume'].rolling(20).mean()
    
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    rsi = float(latest['RSI']) if not pd.isna(latest['RSI']) else 50
    close = float(latest['Close'])
    low = float(latest['Low'])
    volume = float(latest['Volume'])
    vol_sma = float(latest['Vol_SMA20']) if not pd.isna(latest['Vol_SMA20']) else volume
    
    trend = trend_info['trend']
    ema20 = trend_info['ema20']
    
    # ===== 多头信号 =====
    
    # A. 多头回调买点
    if trend in ["强多头", "多头回调"]:
        touched_ema = low <= ema20 * 1.02
        rsi_pullback = 40 < rsi < 55
        
        if touched_ema and rsi_pullback:
            signals.append("🟢 多头回调买点")
            direction = "看多"
    
    # B. 超卖反转
    if rsi < 30:
        signals.append("🔵 超卖")
        prev_rsi = float(prev['RSI']) if not pd.isna(prev['RSI']) else 50
        if prev_rsi < 30 and rsi > 30:
            signals.append("🔵 超卖反转确认")
        direction = "看多"
    
    # ===== 空头信号 =====
    
    # C. 空头反弹做空
    if trend in ["强空头", "空头反弹"] and rsi > 60:
        signals.append("🔴 空头反弹做空点")
        direction = "看空"
    
    # D. 超买
    if rsi > 70:
        signals.append("🟠 超买")
        if trend in ["强空头", "空头反弹", "震荡"]:
            direction = "看空"
    
    # ===== Squeeze信号 =====
    
    if 'BB_Upper' in df.columns and 'KC_Upper' in df.columns:
        bb_upper = float(latest['BB_Upper']) if not pd.isna(latest['BB_Upper']) else close * 1.1
        bb_lower = float(latest['BB_Lower']) if not pd.isna(latest['BB_Lower']) else close * 0.9
        kc_upper = float(latest['KC_Upper']) if not pd.isna(latest['KC_Upper']) else close * 1.1
        kc_lower = float(latest['KC_Lower']) if not pd.isna(latest['KC_Lower']) else close * 0.9
        
        squeeze_on = (bb_upper < kc_upper) and (bb_lower > kc_lower)
        
        prev_bb_upper = float(prev['BB_Upper']) if not pd.isna(prev['BB_Upper']) else close * 1.1
        prev_bb_lower = float(prev['BB_Lower']) if not pd.isna(prev['BB_Lower']) else close * 0.9
        prev_kc_upper = float(prev['KC_Upper']) if not pd.isna(prev['KC_Upper']) else close * 1.1
        prev_kc_lower = float(prev['KC_Lower']) if not pd.isna(prev['KC_Lower']) else close * 0.9
        prev_squeeze = (prev_bb_upper < prev_kc_upper) and (prev_bb_lower > prev_kc_lower)
        
        if squeeze_on:
            signals.append("⏳ Squeeze蓄势")
        
        if prev_squeeze and not squeeze_on:
            if close > bb_upper:
                signals.append("🔥 Squeeze向上突破")
                direction = "看多"
            elif close < bb_lower:
                signals.append("💥 Squeeze向下突破")
                direction = "看空"
    
    # ===== 成交量异动 =====
    vol_ratio = volume / vol_sma if vol_sma > 0 else 1
    if 1.5 < vol_ratio < 3:
        signals.append("📊 放量")
    elif vol_ratio >= 3:
        signals.append("⚠️ 极端放量")
    
    return signals, direction, {
        'rsi': rsi,
        'atr_pct': float(latest['ATR_Pct']) if not pd.isna(latest['ATR_Pct']) else 0,
        'vol_ratio': vol_ratio,
    }


def calculate_score(trend: str, signals: list, indicators: dict) -> int:
    """Level 4: 综合评分"""
    score = 0
    
    if trend in ["强多头", "强空头"]:
        score += 1
    
    if "🔥 Squeeze向上突破" in signals or "💥 Squeeze向下突破" in signals:
        score += 3
    elif "⏳ Squeeze蓄势" in signals:
        score += 1
    
    if "🟢 多头回调买点" in signals:
        score += 2
    
    if "🔴 空头反弹做空点" in signals:
        score += 2
    
    if "🔵 超卖反转确认" in signals:
        score += 2
    elif "🔵 超卖" in signals:
        score += 1
    
    if 1.5 < indicators.get('vol_ratio', 1) < 3:
        score += 1
    
    if indicators.get('atr_pct', 0) > 0.03:
        score += 1
    
    return score


def match_stock_to_sector(stock_sector: str, stock_industry: str) -> str:
    """将股票板块映射到ETF板块"""
    if not stock_sector or stock_sector == 'Unknown':
        return "未知"
    
    combined = f"{stock_sector} {stock_industry}".lower()
    
    for etf_sector, keywords in SECTOR_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in combined:
                return etf_sector
    
    return stock_sector  # 返回原始板块名


def determine_wind_direction(signal_direction: str, sector_flow: str) -> str:
    """判断顺风/逆风"""
    if signal_direction == "中性" or sector_flow == "中性" or sector_flow == "未知":
        return "—"
    
    # 看多 + 资金流入 = 顺风
    # 看多 + 资金流出 = 逆风
    # 看空 + 资金流出 = 顺风
    # 看空 + 资金流入 = 逆风
    
    if signal_direction == "看多":
        if sector_flow == "流入":
            return "🌬️ 顺风"
        else:
            return "🌪️ 逆风"
    elif signal_direction == "看空":
        if sector_flow == "流出":
            return "🌬️ 顺风"
        else:
            return "🌪️ 逆风"
    
    return "—"


def screen_single_stock(ticker: str, sector_flow_dict: dict = None) -> dict:
    """筛选单只股票"""
    result = {
        'ticker': ticker,
        'name': ticker,
        'passed': False,
        'reason': '',
        'trend': '',
        'direction': '中性',
        'signals': [],
        'score': 0,
        'rsi': 0,
        'atr_pct': 0,
        'vol_ratio': 0,
        'sector': 'Unknown',
        'mapped_sector': '未知',
        'sector_flow': '未知',
        'wind': '—',
        'price': 0,
    }
    
    df = get_stock_data(ticker)
    if df is None or df.empty:
        result['reason'] = "无法获取数据"
        return result
    
    # Level 0
    passed, reason = level_0_filter(df, ticker)
    if not passed:
        result['reason'] = reason
        return result
    
    # Level 1
    trend_info = level_1_classify(df)
    result['trend'] = trend_info['trend']
    result['price'] = trend_info['close']
    
    # Level 2 & 3
    signals, direction, indicators = level_2_3_signals(df, trend_info)
    result['signals'] = signals
    result['direction'] = direction
    result['rsi'] = indicators['rsi']
    result['atr_pct'] = indicators['atr_pct']
    result['vol_ratio'] = indicators['vol_ratio']
    
    # Level 4
    score = calculate_score(trend_info['trend'], signals, indicators)
    result['score'] = score
    
    # 获取板块信息
    info = get_stock_info(ticker)
    result['sector'] = info['sector']
    result['name'] = info['shortName']
    
    # 映射到ETF板块
    mapped_sector = match_stock_to_sector(info['sector'], info['industry'])
    result['mapped_sector'] = mapped_sector
    
    # 获取板块资金流状态
    if sector_flow_dict and mapped_sector in sector_flow_dict:
        result['sector_flow'] = sector_flow_dict[mapped_sector]
    else:
        result['sector_flow'] = '未知'
    
    # 判断顺风/逆风
    result['wind'] = determine_wind_direction(direction, result['sector_flow'])
    
    # 判断是否通过
    if len(signals) > 0 and score >= 2:
        result['passed'] = True
        result['reason'] = "通过筛选"
    else:
        result['reason'] = "无有效信号"
    
    return result


# ============================================================
# QQQ/NQ 盘前分析函数
# ============================================================

def parse_qqq_premarket_text(text):
    """解析QQQ盘前粘贴数据"""
    import re
    result = {
        'qqq': {'current': None, 'prev_close': None, 'levels': {}},
        'nq': {'current': None, 'prev_close': None, 'levels': {}}
    }
    
    lines = text.strip().split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 检测QQQ部分
        if 'QQQ' in line.upper() and ('盘前' in line or '现价' in line):
            current_section = 'qqq'
            prices = re.findall(r'[\d.]+', line)
            if len(prices) >= 2:
                result['qqq']['current'] = float(prices[0])
                result['qqq']['prev_close'] = float(prices[1])
            continue
        
        # 检测NQ部分
        if 'NQ' in line.upper() and ('盘前' in line or '现价' in line):
            current_section = 'nq'
            prices = re.findall(r'[\d.]+', line)
            if len(prices) >= 2:
                result['nq']['current'] = float(prices[0])
                result['nq']['prev_close'] = float(prices[1])
            continue
        
        # 解析关键位
        if current_section:
            parts = line.split()
            
            if current_section == 'qqq' and len(parts) >= 2:
                try:
                    price = float(parts[0])
                    level_name = ' '.join(parts[1:])
                    result['qqq']['levels'][level_name] = price
                except:
                    pass
            
            elif current_section == 'nq' and len(parts) >= 3:
                try:
                    # 第二列是NQ值（忽略第一列NDX）
                    price = float(parts[1])
                    level_name = ' '.join(parts[2:])
                    result['nq']['levels'][level_name] = price
                except:
                    pass
    
    return result

def analyze_qqq_nq(premarket_data, csv_data=None):
    """分析QQQ/NQ数据，包含方向性分析和情景分析"""
    analysis = {
        'qqq': {},
        'nq': {},
        'cross_validation': {},
        'prediction': {},
        'directional': {},
        'scenarios': []
    }
    
    for market in ['qqq', 'nq']:
        data = premarket_data.get(market, {})
        levels = data.get('levels', {})
        
        # 查找关键位
        zero_gamma = None
        call_wall = None
        put_wall = None
        vol_trigger = None
        
        for name, price in levels.items():
            name_lower = name.lower()
            if 'zero gamma' in name_lower:
                zero_gamma = price
            elif 'call wall' in name_lower:
                call_wall = price
            elif 'put wall' in name_lower:
                put_wall = price
            elif 'volatility trigger' in name_lower:
                vol_trigger = price
        
        current = data.get('current')
        prev_close = data.get('prev_close')
        
        analysis[market] = {
            'current': current,
            'prev_close': prev_close,
            'zero_gamma': zero_gamma,
            'call_wall': call_wall,
            'put_wall': put_wall,
            'vol_trigger': vol_trigger,
            'levels': levels
        }
        
        if current and prev_close:
            analysis[market]['change_pct'] = (current - prev_close) / prev_close * 100
        
        # Gamma环境
        if current and zero_gamma:
            if current > zero_gamma:
                analysis[market]['gamma_env'] = "正Gamma"
                analysis[market]['gamma_env_type'] = "positive"
            else:
                analysis[market]['gamma_env'] = "负Gamma"
                analysis[market]['gamma_env_type'] = "negative"
        
        # 波动环境
        if current and vol_trigger:
            if current > vol_trigger:
                analysis[market]['vol_regime'] = "均值回归"
                analysis[market]['vol_regime_type'] = "mean_reversion"
            else:
                analysis[market]['vol_regime'] = "高波动/趋势"
                analysis[market]['vol_regime_type'] = "trending"
    
    # 交叉验证
    qqq_env = analysis['qqq'].get('gamma_env_type')
    nq_env = analysis['nq'].get('gamma_env_type')
    
    if qqq_env and nq_env:
        if qqq_env == nq_env:
            analysis['cross_validation'] = {
                'status': '一致',
                'message': f"✅ QQQ和NQ均处于{analysis['qqq'].get('gamma_env')}环境，信号可靠",
                'confidence': 'high'
            }
        else:
            analysis['cross_validation'] = {
                'status': '矛盾',
                'message': f"⚠️ 信号矛盾！QQQ={analysis['qqq'].get('gamma_env')}, NQ={analysis['nq'].get('gamma_env')}，以NQ为主",
                'confidence': 'low',
                'dominant': 'nq'
            }
    
    # 生成预测
    qqq = analysis['qqq']
    nq = analysis['nq']
    
    # 使用NQ为主导（如果矛盾）
    dominant = 'nq' if analysis['cross_validation'].get('dominant') == 'nq' else 'qqq'
    dominant_data = analysis[dominant]
    
    if qqq.get('call_wall') and qqq.get('put_wall'):
        analysis['prediction'] = {
            'resistance': f"{qqq.get('call_wall', 'N/A')}",
            'support': f"{qqq.get('put_wall', 'N/A')}",
            'gamma_env': dominant_data.get('gamma_env', '未知'),
            'vol_regime': dominant_data.get('vol_regime', '未知')
        }
        
        # 收盘预测
        if qqq.get('zero_gamma') and qqq.get('current'):
            zg = qqq['zero_gamma']
            close_low = zg * 0.995
            close_high = zg * 1.005
            analysis['prediction']['close_range'] = f"{close_low:.2f} - {close_high:.2f}"
            analysis['prediction']['close_target'] = zg
    
    # ===== CSV数据的方向性分析 =====
    if csv_data is not None:
        directional = []
        
        # Delta Ratio分析
        dr = csv_data.get('Delta Ratio')
        if dr is not None:
            dr_val = parse_number_safe(str(dr).replace("'", "-"))
            if dr_val is not None:
                if dr_val > -1:
                    directional.append(f"• Delta Ratio {dr_val:.2f} > -1: **Call Delta主导，偏多** 📈")
                    analysis['directional']['delta_ratio'] = ('bullish', dr_val)
                elif dr_val < -2:
                    directional.append(f"• Delta Ratio {dr_val:.2f} < -2: **Put Delta主导，偏空** 📉")
                    analysis['directional']['delta_ratio'] = ('bearish', dr_val)
                else:
                    directional.append(f"• Delta Ratio {dr_val:.2f}: 中性")
                    analysis['directional']['delta_ratio'] = ('neutral', dr_val)
        
        # Gamma Ratio分析
        gr = csv_data.get('Gamma Ratio')
        if gr is not None:
            gr_val = parse_number_safe(str(gr))
            if gr_val is not None:
                if gr_val < 1:
                    directional.append(f"• Gamma Ratio {gr_val:.2f} < 1: **Call Gamma主导，上涨加速** 📈")
                    analysis['directional']['gamma_ratio'] = ('bullish', gr_val)
                elif gr_val > 1.5:
                    directional.append(f"• Gamma Ratio {gr_val:.2f} > 1.5: **Put Gamma主导，下跌加速** 📉")
                    analysis['directional']['gamma_ratio'] = ('bearish', gr_val)
                else:
                    directional.append(f"• Gamma Ratio {gr_val:.2f}: 均衡")
                    analysis['directional']['gamma_ratio'] = ('neutral', gr_val)
        
        # Volume Ratio分析
        vr = csv_data.get('Volume Ratio')
        if vr is not None:
            vr_val = parse_number_safe(str(vr))
            if vr_val is not None:
                if vr_val < 0.7:
                    directional.append(f"• Volume Ratio {vr_val:.2f} < 0.7: **Call交易活跃，偏多** 📈")
                    analysis['directional']['volume_ratio'] = ('bullish', vr_val)
                elif vr_val > 1.3:
                    directional.append(f"• Volume Ratio {vr_val:.2f} > 1.3: **Put交易活跃，偏空** 📉")
                    analysis['directional']['volume_ratio'] = ('bearish', vr_val)
                else:
                    directional.append(f"• Volume Ratio {vr_val:.2f}: 均衡")
                    analysis['directional']['volume_ratio'] = ('neutral', vr_val)
        
        # Hedge Wall分析
        hw = csv_data.get('Hedge Wall')
        current_price = qqq.get('current') or csv_data.get('Current Price')
        if hw is not None and current_price:
            hw_val = parse_number_safe(str(hw))
            price_val = parse_number_safe(str(current_price))
            if hw_val and price_val:
                if price_val > hw_val:
                    directional.append(f"• 价格 > Hedge Wall ({hw_val:.0f}): **均值回归模式**")
                    analysis['directional']['hedge_wall'] = ('mean_reversion', hw_val)
                else:
                    directional.append(f"• 价格 < Hedge Wall ({hw_val:.0f}): **趋势/高波动模式**")
                    analysis['directional']['hedge_wall'] = ('trending', hw_val)
        
        # NEG分析
        neg = csv_data.get('Next Exp Gamma')
        if neg is not None:
            neg_val = parse_number_safe(str(neg).replace('%', ''))
            if neg_val is not None:
                if neg_val < 1:
                    neg_val = neg_val * 100
                if neg_val > 40:
                    directional.append(f"• Next Exp Gamma {neg_val:.1f}%: **极高集中，剧烈波动风险** ⚠️")
                elif neg_val > 25:
                    directional.append(f"• Next Exp Gamma {neg_val:.1f}%: **较高集中，注意到期波动**")
                else:
                    directional.append(f"• Next Exp Gamma {neg_val:.1f}%: 正常分布")
                analysis['directional']['neg'] = neg_val
        
        # IV vs RV分析
        iv = csv_data.get('30 Day IV')
        rv = csv_data.get('Realized Vol')
        if iv is not None and rv is not None:
            iv_val = parse_number_safe(str(iv).replace('%', ''))
            rv_val = parse_number_safe(str(rv).replace('%', ''))
            if iv_val and rv_val:
                if iv_val < 1:
                    iv_val = iv_val * 100
                if rv_val < 1:
                    rv_val = rv_val * 100
                if iv_val > rv_val * 1.2:
                    directional.append(f"• IV {iv_val:.1f}% > RV {rv_val:.1f}%: **期权偏贵，卖方有利**")
                elif iv_val < rv_val * 0.8:
                    directional.append(f"• IV {iv_val:.1f}% < RV {rv_val:.1f}%: **期权偏便宜，买方有利**")
                else:
                    directional.append(f"• IV {iv_val:.1f}% ≈ RV {rv_val:.1f}%: 期权定价合理")
                analysis['directional']['iv_rv'] = (iv_val, rv_val)
        
        # DPI分析
        dpi = csv_data.get('DPI')
        dpi_5d = csv_data.get('5Day DPI')
        if dpi is not None:
            dpi_val = parse_number_safe(str(dpi).replace('%', ''))
            if dpi_val:
                if dpi_val < 1:
                    dpi_val = dpi_val * 100
                if dpi_val > 52:
                    directional.append(f"• DPI {dpi_val:.1f}%: **机构强力买入** 🏦📈")
                elif dpi_val > 48:
                    directional.append(f"• DPI {dpi_val:.1f}%: 机构积极买入 🏦")
                elif dpi_val < 45:
                    directional.append(f"• DPI {dpi_val:.1f}%: **机构买盘减弱** ⚠️")
                else:
                    directional.append(f"• DPI {dpi_val:.1f}%: 机构中性")
                analysis['directional']['dpi'] = dpi_val
        
        if dpi_5d is not None:
            dpi_5d_val = parse_number_safe(str(dpi_5d).replace('%', ''))
            if dpi_5d_val:
                if dpi_5d_val < 1:
                    dpi_5d_val = dpi_5d_val * 100
                directional.append(f"• 5Day DPI {dpi_5d_val:.1f}%")
                analysis['directional']['dpi_5d'] = dpi_5d_val
        
        analysis['directional']['items'] = directional
        
        # ===== 情景分析 =====
        scenarios = []
        
        # 获取关键位
        cw = qqq.get('call_wall') or parse_number_safe(str(csv_data.get('Call Wall', 0)))
        pw = qqq.get('put_wall') or parse_number_safe(str(csv_data.get('Put Wall', 0)))
        zg = qqq.get('zero_gamma')
        current = qqq.get('current') or parse_number_safe(str(csv_data.get('Current Price', 0)))
        
        gamma_env = analysis['qqq'].get('gamma_env_type', 'positive')
        
        if gamma_env == 'positive':
            # 正Gamma环境
            scenarios.append({
                'name': '区间震荡',
                'probability': 55,
                'description': f"在 {zg:.0f}-{cw:.0f} 区间震荡" if zg and cw else "在Zero Gamma和Call Wall之间震荡",
                'strategy': "支撑做多，阻力获利；高抛低吸"
            })
            scenarios.append({
                'name': '冲高回落',
                'probability': 30,
                'description': f"冲击 {cw:.0f} Call Wall 后回落" if cw else "冲击Call Wall后回落",
                'strategy': "不追Call Wall突破；CW附近减仓或做空"
            })
            scenarios.append({
                'name': '下探反弹',
                'probability': 15,
                'description': f"下探 {zg:.0f} Zero Gamma 后反弹" if zg else "下探Zero Gamma后反弹",
                'strategy': "Zero Gamma是做多机会；设止损于ZG下方"
            })
        else:
            # 负Gamma环境
            scenarios.append({
                'name': '趋势延续',
                'probability': 50,
                'description': "负Gamma放大波动，趋势可能延续",
                'strategy': "顺势操作，严格止损；不抄底不摸顶"
            })
            scenarios.append({
                'name': '剧烈震荡',
                'probability': 30,
                'description': "在关键位之间大幅震荡",
                'strategy': "减小仓位；等待方向明确"
            })
            scenarios.append({
                'name': 'V型反转',
                'probability': 20,
                'description': f"触及{pw:.0f} Put Wall后V型反转" if pw else "触及Put Wall后V型反转",
                'strategy': "Put Wall是潜在反转点；需确认信号"
            })
        
        analysis['scenarios'] = scenarios
    
    return analysis

# ============================================================
# 周五到期Gamma分析函数
# ============================================================

def parse_number_safe(value):
    """安全解析数字"""
    if value is None:
        return None
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    
    value = str(value).strip().replace(',', '').replace("'", "-")
    
    multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}
    
    for suffix, mult in multipliers.items():
        if value.upper().endswith(suffix):
            try:
                return float(value[:-1]) * mult
            except:
                return None
    
    try:
        return float(value.replace('%', '').replace('$', ''))
    except:
        return None

def get_neg_strength(neg):
    """
    获取NEG强度 - 基于做市商解绑压力
    >35% = 强信号（高浓度，解绑压力大）
    25-35% = 中等信号
    <25% = 弱信号（不建议操作）
    """
    neg_val = parse_number_safe(str(neg).replace('%', '')) if neg else 0
    if neg_val is None:
        neg_val = 0
    if neg_val < 1:
        neg_val = neg_val * 100
    
    if neg_val >= 35:
        return neg_val, "⭐⭐⭐", "strong"
    elif neg_val >= 25:
        return neg_val, "⭐⭐", "medium"
    else:
        return neg_val, "⭐", "weak"

def analyze_mm_unwinding_v3(row):
    """
    分析做市商解绑效应（MM Unwinding Analysis）- V3版本
    
    整合完整5步分析流程：
    1. 高冲击判断：Next Exp Vol > 25% → 进入分析
    2. 主导方判断：Delta Ratio + Volume Ratio + 到期量 → Call/Put主导
    3. 解绑分数计算：Gap Up Score / Gap Down Score → 量化强度
    4. 位置验证：距离Wall的远近 → 触发敏感度 + DPI确认
    5. 最终信号：综合以上产生可操作的信号
    """
    # ========== 第0步：提取所有字段 ==========
    current_price = parse_number_safe(row.get('Current Price'))
    call_wall = parse_number_safe(row.get('Call Wall'))
    put_wall = parse_number_safe(row.get('Put Wall'))
    hedge_wall = parse_number_safe(row.get('Hedge Wall'))
    key_delta_strike = parse_number_safe(row.get('Key Delta Strike'))
    key_gamma_strike = parse_number_safe(row.get('Key Gamma Strike'))
    
    # 浓度指标
    neg = row.get('Next Exp Gamma')  # Next Expiry Gamma %
    ned = row.get('Next Exp Delta')  # Next Expiry Delta %
    
    # 到期成交量占比
    next_exp_call_vol = parse_number_safe(row.get('Next Exp Call Vol')) or 0
    next_exp_put_vol = parse_number_safe(row.get('Next Exp Put Vol')) or 0
    
    # 方向性指标
    # Delta Ratio = Put Delta / Call Delta，负数，越接近0=Call主导，越负=Put主导
    delta_ratio_raw = row.get('Delta Ratio')
    delta_ratio = -1.0  # 默认值
    if delta_ratio_raw:
        delta_ratio_str = str(delta_ratio_raw).replace("'", "")
        try:
            delta_ratio = float(delta_ratio_str)
        except:
            delta_ratio = -1.0
    
    # Volume Ratio = Put Delta Vol / Call Delta Vol (ATM)
    # > 1.5 = ATM Put头寸大（MM空头重），< 0.8 = ATM Call头寸大（MM多头重）
    volume_ratio = parse_number_safe(row.get('Volume Ratio')) or 1.0
    
    # Gamma Ratio
    gamma_ratio = parse_number_safe(row.get('Gamma Ratio'))
    
    # 其他指标
    options_impact = parse_number_safe(row.get('Options Impact'))
    iv_rank = parse_number_safe(row.get('IV Rank'))
    dpi = parse_number_safe(row.get('DPI'))
    dpi_5d = parse_number_safe(row.get('5 day DPI')) or dpi or 50
    implied_move = parse_number_safe(row.get('Options Implied Move'))
    
    # 财报日期
    earnings_date = row.get('Earnings Date')
    
    # 解析浓度
    neg_val, neg_stars, neg_strength = get_neg_strength(neg)
    ned_val = parse_number_safe(str(ned).replace('%', '')) if ned else 0
    if ned_val and ned_val < 1:
        ned_val = ned_val * 100
    
    # ========== 构建分析结果 ==========
    analysis = {
        'signal_type': 'neutral',
        'prediction': '⚖️ 观望',
        'confidence': neg_stars,
        'dominance': 'neutral',
        'gap_up_score': 0,
        'gap_down_score': 0,
        'position_zone': 'middle',
        'mm_behavior': '',
        'logic_chain': [],
        'step_details': {},  # 每步详细信息
        'warnings': [],
        'data_summary': {
            'neg': neg_val,
            'ned': ned_val,
            'delta_ratio': delta_ratio,
            'volume_ratio': volume_ratio,
            'gamma_ratio': gamma_ratio,
            'next_exp_call_vol': next_exp_call_vol,
            'next_exp_put_vol': next_exp_put_vol,
            'key_delta_strike': key_delta_strike,
            'options_impact': options_impact,
            'iv_rank': iv_rank,
            'dpi': dpi,
            'dpi_5d': dpi_5d,
            'implied_move': implied_move
        }
    }
    
    # ========== 第1步：高冲击判断 ==========
    is_high_impact = (next_exp_call_vol > 0.25) or (next_exp_put_vol > 0.25) or (neg_val >= 25)
    
    analysis['step_details']['step1_impact'] = {
        'is_high_impact': is_high_impact,
        'next_exp_call_vol': next_exp_call_vol,
        'next_exp_put_vol': next_exp_put_vol,
        'neg': neg_val
    }
    
    if not is_high_impact:
        analysis['signal_type'] = 'low_impact'
        analysis['prediction'] = '⚪ 到期冲击弱 - 不建议操作'
        analysis['logic_chain'] = [
            "═══════════════════════════════════════",
            "📋 【第1步：高冲击判断】 ❌ 未通过",
            "═══════════════════════════════════════",
            "",
            f"   • Next Exp Call Vol: {next_exp_call_vol:.1%} {'✓' if next_exp_call_vol > 0.25 else '✗'} (阈值>25%)",
            f"   • Next Exp Put Vol: {next_exp_put_vol:.1%} {'✓' if next_exp_put_vol > 0.25 else '✗'} (阈值>25%)",
            f"   • NEG: {neg_val:.1f}% {'✓' if neg_val >= 25 else '✗'} (阈值≥25%)",
            "",
            "❌ 本周到期期权浓度不足，做市商解绑压力小",
            "→ 信号不可靠，不建议基于此操作"
        ]
        analysis['mm_behavior'] = "到期浓度不足，无明显解绑压力"
        return analysis
    
    analysis['logic_chain'].append("═══════════════════════════════════════")
    analysis['logic_chain'].append("📋 【第1步：高冲击判断】 ✅ 通过")
    analysis['logic_chain'].append("═══════════════════════════════════════")
    analysis['logic_chain'].append("")
    analysis['logic_chain'].append(f"   • Next Exp Call Vol: {next_exp_call_vol:.1%} {'✓' if next_exp_call_vol > 0.25 else ''}")
    analysis['logic_chain'].append(f"   • Next Exp Put Vol: {next_exp_put_vol:.1%} {'✓' if next_exp_put_vol > 0.25 else ''}")
    analysis['logic_chain'].append(f"   • NEG: {neg_val:.1f}% {neg_stars}")
    analysis['logic_chain'].append("")
    
    # ========== 第2步：主导方判断（四维模型）==========
    # 四维指标：Delta Ratio（库存）+ Gamma Ratio（敏感度）+ Volume Ratio（流向）+ Next Exp Vol（能量）
    
    # Call主导评分
    call_score = 0
    call_reasons = []
    
    # 维度1: Delta Ratio（库存）- Call主导时接近0
    if delta_ratio > -1.0:
        call_score += 3
        call_reasons.append(f"[库存] Delta Ratio {delta_ratio:.2f} > -1.0（Call库存极重）")
    elif delta_ratio > -1.5:
        call_score += 2
        call_reasons.append(f"[库存] Delta Ratio {delta_ratio:.2f} > -1.5（Call库存重）")
    
    # 维度2: Gamma Ratio（敏感度）- Call主导时<1
    if gamma_ratio is not None:
        if gamma_ratio < 0.8:
            call_score += 3
            call_reasons.append(f"[敏感度] Gamma Ratio {gamma_ratio:.2f} < 0.8（Call Gamma主导，到期冲击大）")
        elif gamma_ratio < 1.0:
            call_score += 1
            call_reasons.append(f"[敏感度] Gamma Ratio {gamma_ratio:.2f} < 1.0（Call Gamma略占优）")
    
    # 维度3: Volume Ratio（流向）- Call主导时<1
    if volume_ratio < 0.8:
        call_score += 3
        call_reasons.append(f"[流向] Volume Ratio {volume_ratio:.2f} < 0.8（ATM Call头寸大 → MM多头重）")
    elif volume_ratio < 1.0:
        call_score += 1
        call_reasons.append(f"[流向] Volume Ratio {volume_ratio:.2f} < 1.0（ATM Call略多）")
    
    # 维度4: Next Exp Vol（能量）- Call到期量大
    if next_exp_call_vol > next_exp_put_vol * 1.3 and next_exp_call_vol > 0.25:
        call_score += 2
        call_reasons.append(f"[能量] Call到期量 {next_exp_call_vol:.1%} > Put {next_exp_put_vol:.1%}（Call能量占优）")
    elif next_exp_call_vol > 0.35:
        call_score += 2
        call_reasons.append(f"[能量] Next Exp Call Vol {next_exp_call_vol:.1%} > 35%（高浓度）")
    elif next_exp_call_vol > 0.25:
        call_score += 1
        call_reasons.append(f"[能量] Next Exp Call Vol {next_exp_call_vol:.1%} > 25%")
    
    # Put主导评分
    put_score = 0
    put_reasons = []
    
    # 维度1: Delta Ratio（库存）- Put主导时很负
    if delta_ratio < -2.5:
        put_score += 3
        put_reasons.append(f"[库存] Delta Ratio {delta_ratio:.2f} < -2.5（Put库存极重）")
    elif delta_ratio < -1.5:
        put_score += 2
        put_reasons.append(f"[库存] Delta Ratio {delta_ratio:.2f} < -1.5（Put库存重）")
    
    # 维度2: Gamma Ratio（敏感度）- Put主导时>1
    if gamma_ratio is not None:
        if gamma_ratio > 1.5:
            put_score += 3
            put_reasons.append(f"[敏感度] Gamma Ratio {gamma_ratio:.2f} > 1.5（Put Gamma主导，到期冲击大）")
        elif gamma_ratio > 1.2:
            put_score += 2
            put_reasons.append(f"[敏感度] Gamma Ratio {gamma_ratio:.2f} > 1.2（Put Gamma较强）")
        elif gamma_ratio > 1.0:
            put_score += 1
            put_reasons.append(f"[敏感度] Gamma Ratio {gamma_ratio:.2f} > 1.0（Put Gamma略占优）")
    
    # 维度3: Volume Ratio（流向）- Put主导时>1
    if volume_ratio > 1.5:
        put_score += 3
        put_reasons.append(f"[流向] Volume Ratio {volume_ratio:.2f} > 1.5（ATM Put头寸大 → MM空头重）")
    elif volume_ratio > 1.2:
        put_score += 2
        put_reasons.append(f"[流向] Volume Ratio {volume_ratio:.2f} > 1.2（ATM Put头寸较大）")
    elif volume_ratio > 1.0:
        put_score += 1
        put_reasons.append(f"[流向] Volume Ratio {volume_ratio:.2f} > 1.0（ATM Put略多）")
    
    # 维度4: Next Exp Vol（能量）- Put到期量大
    if next_exp_put_vol > next_exp_call_vol * 1.3 and next_exp_put_vol > 0.25:
        put_score += 2
        put_reasons.append(f"[能量] Put到期量 {next_exp_put_vol:.1%} > Call {next_exp_call_vol:.1%}（Put能量占优）")
    elif next_exp_put_vol > 0.35:
        put_score += 2
        put_reasons.append(f"[能量] Next Exp Put Vol {next_exp_put_vol:.1%} > 35%（高浓度）")
    elif next_exp_put_vol > 0.25:
        put_score += 1
        put_reasons.append(f"[能量] Next Exp Put Vol {next_exp_put_vol:.1%} > 25%")
    
    # 判定主导方（四维总分，满分11分）
    if call_score >= 5 and call_score > put_score + 2:
        dominance = 'call_dominant'
        dominance_text = "📉 CALL主导（低开风险）"
    elif put_score >= 5 and put_score > call_score + 2:
        dominance = 'put_dominant'
        dominance_text = "📈 PUT主导（高开机会）"
    elif call_score >= 4 and call_score > put_score:
        dominance = 'call_lean'
        dominance_text = "📉 偏Call主导"
    elif put_score >= 4 and put_score > call_score:
        dominance = 'put_lean'
        dominance_text = "📈 偏Put主导"
    else:
        dominance = 'neutral'
        dominance_text = "⚖️ 多空均衡"
    
    analysis['dominance'] = dominance
    analysis['step_details']['step2_dominance'] = {
        'dominance': dominance,
        'call_score': call_score,
        'put_score': put_score,
        'call_reasons': call_reasons,
        'put_reasons': put_reasons
    }
    
    analysis['logic_chain'].append("═══════════════════════════════════════")
    analysis['logic_chain'].append("📋 【第2步：四维主导方判断】")
    analysis['logic_chain'].append("═══════════════════════════════════════")
    analysis['logic_chain'].append("")
    analysis['logic_chain'].append("四维指标: 库存(Delta Ratio) + 敏感度(Gamma Ratio) + 流向(Volume Ratio) + 能量(Next Exp Vol)")
    analysis['logic_chain'].append("")
    analysis['logic_chain'].append(f"🔴 Call主导评分: {call_score}/11分")
    for r in call_reasons:
        analysis['logic_chain'].append(f"   • {r}")
    if not call_reasons:
        analysis['logic_chain'].append("   • (无满足条件)")
    analysis['logic_chain'].append("")
    analysis['logic_chain'].append(f"🟢 Put主导评分: {put_score}/11分")
    for r in put_reasons:
        analysis['logic_chain'].append(f"   • {r}")
    if not put_reasons:
        analysis['logic_chain'].append("   • (无满足条件)")
    analysis['logic_chain'].append("")
    analysis['logic_chain'].append(f"→ 判定结果: {dominance_text}")
    analysis['logic_chain'].append("")
    
    # ========== 第3步：解绑分数计算 ==========
    # 计算距离（百分比，避免除以0）
    dist_to_cw = None
    dist_to_pw = None
    dist_to_cw_raw = None  # 原始距离（用于显示）
    dist_to_pw_raw = None
    
    if current_price and call_wall and call_wall > 0:
        dist_to_cw_raw = (current_price - call_wall) / call_wall * 100  # 正数=在CW上方
        dist_to_cw = max(0.001, abs(call_wall - current_price) / call_wall)  # 用于分母
        if current_price > call_wall:
            dist_to_cw = 0.001  # 已突破，极端敏感
    
    if current_price and put_wall and put_wall > 0:
        dist_to_pw_raw = (current_price - put_wall) / put_wall * 100  # 正数=在PW上方
        dist_to_pw = max(0.001, abs(current_price - put_wall) / put_wall)  # 用于分母
        if current_price < put_wall:
            dist_to_pw = 0.001  # 已跌破，极端敏感
    
    # Delta Ratio绝对值（用于计算）
    delta_ratio_abs = max(0.1, abs(delta_ratio))
    
    # Gamma Ratio因子（敏感度放大器）
    # Gamma Ratio > 1 时，Put Gamma主导，高开分应放大
    # Gamma Ratio < 1 时，Call Gamma主导，低开分应放大
    gamma_up_factor = 1.0
    gamma_down_factor = 1.0
    if gamma_ratio is not None:
        if gamma_ratio > 1.5:
            gamma_up_factor = 1.5  # Put Gamma极强，高开分×1.5
        elif gamma_ratio > 1.2:
            gamma_up_factor = 1.3
        elif gamma_ratio > 1.0:
            gamma_up_factor = 1.1
        
        if gamma_ratio < 0.8:
            gamma_down_factor = 1.5  # Call Gamma极强，低开分×1.5
        elif gamma_ratio < 0.9:
            gamma_down_factor = 1.3
        elif gamma_ratio < 1.0:
            gamma_down_factor = 1.1
    
    # Gap Up Score = (Put到期量 × |Delta Ratio| × Gamma放大因子) / 距Put Wall距离
    # 逻辑：Put到期越多 + MM空头越重 + Put Gamma越高 + 距离越近 = 买压越大
    gap_up_score = 0
    if dist_to_pw and dist_to_pw > 0:
        gap_up_score = (next_exp_put_vol * delta_ratio_abs * gamma_up_factor) / dist_to_pw
    
    # Gap Down Score = (Call到期量 × 1/|Delta Ratio| × Gamma放大因子) / 距Call Wall距离
    # 逻辑：Call到期越多 + MM多头越重 + Call Gamma越高 + 距离越近 = 卖压越大
    gap_down_score = 0
    if dist_to_cw and dist_to_cw > 0:
        gap_down_score = (next_exp_call_vol * (1 / delta_ratio_abs) * gamma_down_factor) / dist_to_cw
    
    analysis['gap_up_score'] = gap_up_score
    analysis['gap_down_score'] = gap_down_score
    analysis['data_summary']['dist_to_cw'] = dist_to_cw_raw
    analysis['data_summary']['dist_to_pw'] = dist_to_pw_raw
    
    # 分数强度判定
    def score_strength(score):
        if score > 100:
            return "极强 ⭐⭐⭐", "extreme"
        elif score > 50:
            return "强 ⭐⭐", "strong"
        elif score > 20:
            return "中等 ⭐", "medium"
        elif score > 10:
            return "弱", "weak"
        else:
            return "无", "none"
    
    gap_up_text, gap_up_level = score_strength(gap_up_score)
    gap_down_text, gap_down_level = score_strength(gap_down_score)
    
    analysis['step_details']['step3_scores'] = {
        'gap_up_score': gap_up_score,
        'gap_down_score': gap_down_score,
        'gap_up_level': gap_up_level,
        'gap_down_level': gap_down_level,
        'gamma_up_factor': gamma_up_factor,
        'gamma_down_factor': gamma_down_factor
    }
    
    analysis['logic_chain'].append("═══════════════════════════════════════")
    analysis['logic_chain'].append("📋 【第3步：解绑分数计算】")
    analysis['logic_chain'].append("═══════════════════════════════════════")
    analysis['logic_chain'].append("")
    analysis['logic_chain'].append("🚀 【Gap Up Score（高开分）- 空头回补压力】")
    analysis['logic_chain'].append(f"   公式: (Put到期量 × |Delta Ratio| × Gamma因子) ÷ 距Put Wall")
    if dist_to_pw:
        analysis['logic_chain'].append(f"   计算: ({next_exp_put_vol:.2f} × {delta_ratio_abs:.2f} × {gamma_up_factor:.1f}) ÷ {dist_to_pw:.4f}")
    else:
        analysis['logic_chain'].append(f"   计算: N/A (距离数据缺失)")
    analysis['logic_chain'].append(f"   Gamma因子: {gamma_up_factor:.1f}x {'(Put Gamma主导，放大买压)' if gamma_up_factor > 1 else ''}")
    analysis['logic_chain'].append(f"   结果: {gap_up_score:.1f} → {gap_up_text}")
    analysis['logic_chain'].append("")
    analysis['logic_chain'].append("💀 【Gap Down Score（低开分）- 多头平仓压力】")
    analysis['logic_chain'].append(f"   公式: (Call到期量 × 1/|Delta Ratio| × Gamma因子) ÷ 距Call Wall")
    if dist_to_cw:
        analysis['logic_chain'].append(f"   计算: ({next_exp_call_vol:.2f} × {1/delta_ratio_abs:.2f} × {gamma_down_factor:.1f}) ÷ {dist_to_cw:.4f}")
    else:
        analysis['logic_chain'].append(f"   计算: N/A (距离数据缺失)")
    analysis['logic_chain'].append(f"   Gamma因子: {gamma_down_factor:.1f}x {'(Call Gamma主导，放大卖压)' if gamma_down_factor > 1 else ''}")
    analysis['logic_chain'].append(f"   结果: {gap_down_score:.1f} → {gap_down_text}")
    analysis['logic_chain'].append("")
    
    # ========== 第4步：位置验证 ==========
    # 判断位置区域
    position_zone = "middle"
    if dist_to_cw_raw is not None:
        if dist_to_cw_raw > 2:
            position_zone = "far_above_cw"
        elif dist_to_cw_raw > 0:
            position_zone = "above_cw"
        elif dist_to_cw_raw >= -1:
            position_zone = "near_cw"
    
    if dist_to_pw_raw is not None:
        if dist_to_pw_raw < -2:
            position_zone = "far_below_pw"
        elif dist_to_pw_raw < 0:
            position_zone = "below_pw"
        elif dist_to_pw_raw <= 1:
            position_zone = "near_pw"
    
    analysis['position_zone'] = position_zone
    
    # DPI确认
    dpi_confirms_bullish = dpi_5d and dpi_5d > 50
    dpi_confirms_bearish = dpi_5d and dpi_5d < 48
    
    # 财报检查
    earnings_conflict = False
    if earnings_date and pd.notna(earnings_date):
        try:
            if isinstance(earnings_date, str):
                earnings_dt = pd.to_datetime(earnings_date)
            else:
                earnings_dt = earnings_date
            today = pd.Timestamp.now()
            days_diff = (earnings_dt - today).days
            if -1 <= days_diff <= 3:
                earnings_conflict = True
                analysis['warnings'].append(f"⚠️ 财报冲突：财报在{earnings_date}，基本面可能覆盖期权信号")
        except:
            pass
    
    analysis['step_details']['step4_position'] = {
        'position_zone': position_zone,
        'dist_to_cw': dist_to_cw_raw,
        'dist_to_pw': dist_to_pw_raw,
        'dpi_5d': dpi_5d,
        'dpi_confirms_bullish': dpi_confirms_bullish,
        'dpi_confirms_bearish': dpi_confirms_bearish
    }
    
    analysis['logic_chain'].append("═══════════════════════════════════════")
    analysis['logic_chain'].append("📋 【第4步：位置验证】")
    analysis['logic_chain'].append("═══════════════════════════════════════")
    analysis['logic_chain'].append("")
    analysis['logic_chain'].append(f"📍 当前价: ${current_price:.2f}" if current_price else "📍 当前价: N/A")
    analysis['logic_chain'].append(f"   • Call Wall: ${call_wall:.2f} (距离: {dist_to_cw_raw:+.2f}%)" if call_wall and dist_to_cw_raw else "   • Call Wall: N/A")
    analysis['logic_chain'].append(f"   • Put Wall: ${put_wall:.2f} (距离: {dist_to_pw_raw:+.2f}%)" if put_wall and dist_to_pw_raw else "   • Put Wall: N/A")
    analysis['logic_chain'].append(f"   • 位置区域: {position_zone}")
    analysis['logic_chain'].append("")
    analysis['logic_chain'].append(f"🏦 【DPI机构确认】")
    analysis['logic_chain'].append(f"   • 5日DPI: {dpi_5d:.1f}%" if dpi_5d else "   • 5日DPI: N/A")
    if dpi_confirms_bullish:
        analysis['logic_chain'].append(f"   • ✅ DPI > 50%，机构在买入，支持看涨")
    elif dpi_confirms_bearish:
        analysis['logic_chain'].append(f"   • ✅ DPI < 48%，机构在卖出，支持看跌")
    else:
        analysis['logic_chain'].append(f"   • ⚖️ DPI中性，无明确机构方向")
    analysis['logic_chain'].append("")
    
    # ========== 第5步：最终信号 ==========
    analysis['logic_chain'].append("═══════════════════════════════════════")
    analysis['logic_chain'].append("📋 【第5步：最终信号】")
    analysis['logic_chain'].append("═══════════════════════════════════════")
    analysis['logic_chain'].append("")
    
    # 综合判断
    final_signal = "neutral"
    
    # 场景1：极端位置 - 远高于Call Wall
    if position_zone == "far_above_cw":
        final_signal = "strong_bearish"
        analysis['prediction'] = f"💀💀💀 极端低开风险 {neg_stars}"
        analysis['mm_behavior'] = "价格远超Call Wall → MM持有巨量裸多头 → 周一强卖压"
        
        analysis['logic_chain'].append("🔴 【极端场景：价格远超Call Wall】")
        analysis['logic_chain'].append("")
        analysis['logic_chain'].append("做市商行为链：")
        analysis['logic_chain'].append("┌─────────────────────────────────────┐")
        analysis['logic_chain'].append("│ 周五前：投资者持有大量Deep ITM Call    │")
        analysis['logic_chain'].append("│ → MM卖出Call（Short Call）           │")
        analysis['logic_chain'].append("│ → MM买入股票对冲（Long Stock）        │")
        analysis['logic_chain'].append("├─────────────────────────────────────┤")
        analysis['logic_chain'].append("│ 周五到期：Call到期/行权，合约消失      │")
        analysis['logic_chain'].append("│ → MM的Short Call头寸消失             │")
        analysis['logic_chain'].append("│ → MM剩余【大量裸多头股票】            │")
        analysis['logic_chain'].append("├─────────────────────────────────────┤")
        analysis['logic_chain'].append("│ 周一：MM集体卖出股票平仓              │")
        analysis['logic_chain'].append("│ → 大量卖压 → 💀💀💀 强势低开          │")
        analysis['logic_chain'].append("└─────────────────────────────────────┘")
    
    # 场景2：极端位置 - 远低于Put Wall
    elif position_zone == "far_below_pw":
        final_signal = "strong_bullish"
        analysis['prediction'] = f"🚀🚀🚀 极端反弹机会 {neg_stars}"
        analysis['mm_behavior'] = "价格远低于Put Wall → MM持有巨量裸空头 → 周一强买压"
        
        analysis['logic_chain'].append("🟢 【极端场景：价格远低于Put Wall】")
        analysis['logic_chain'].append("")
        analysis['logic_chain'].append("做市商行为链：")
        analysis['logic_chain'].append("┌─────────────────────────────────────┐")
        analysis['logic_chain'].append("│ 周五前：投资者持有大量Deep ITM Put    │")
        analysis['logic_chain'].append("│ → MM卖出Put（Short Put）            │")
        analysis['logic_chain'].append("│ → MM做空股票对冲（Short Stock）      │")
        analysis['logic_chain'].append("├─────────────────────────────────────┤")
        analysis['logic_chain'].append("│ 周五到期：Put到期/行权，合约消失      │")
        analysis['logic_chain'].append("│ → MM的Short Put头寸消失             │")
        analysis['logic_chain'].append("│ → MM剩余【大量裸空头股票】           │")
        analysis['logic_chain'].append("├─────────────────────────────────────┤")
        analysis['logic_chain'].append("│ 周一：MM集体买入股票平仓（空头回补）  │")
        analysis['logic_chain'].append("│ → 大量买压 → 🚀🚀🚀 强势反弹         │")
        analysis['logic_chain'].append("└─────────────────────────────────────┘")
        
        if iv_rank and iv_rank > 0.7:
            analysis['warnings'].append(f"⚠️ IV Rank={iv_rank:.1%}偏高，恐慌可能未结束")
    
    # 场景3：Call主导 + 接近/突破Call Wall
    elif dominance == 'call_dominant' and position_zone in ['above_cw', 'near_cw']:
        if gap_down_score > 50:
            final_signal = "strong_bearish"
            analysis['prediction'] = f"💀💀 CALL UNWINDING 强卖压 {neg_stars}"
        elif gap_down_score > 20:
            final_signal = "bearish"
            analysis['prediction'] = f"💀 CALL PINNING 卖压 {neg_stars}"
        else:
            final_signal = "bearish_watch"
            analysis['prediction'] = f"📉 偏空观察 {neg_stars}"
        
        analysis['mm_behavior'] = "Call主导 + 接近CW → MM多头将平仓 → 卖压"
        
        analysis['logic_chain'].append("🔴 【Call主导 + 接近Call Wall】")
        analysis['logic_chain'].append("")
        analysis['logic_chain'].append("做市商行为链：")
        analysis['logic_chain'].append("┌─────────────────────────────────────┐")
        analysis['logic_chain'].append("│ 投资者买入ATM Call → MM卖出Call     │")
        analysis['logic_chain'].append("│ → Short Call = 负Delta暴露          │")
        analysis['logic_chain'].append("│ → MM买入股票对冲（Long Stock）       │")
        analysis['logic_chain'].append("├─────────────────────────────────────┤")
        analysis['logic_chain'].append("│ 周五到期：ITM Call消失               │")
        analysis['logic_chain'].append("│ → MM剩余裸多头股票                   │")
        analysis['logic_chain'].append("├─────────────────────────────────────┤")
        analysis['logic_chain'].append("│ 周一：MM卖出多头 → 卖压 → 💀 低开    │")
        analysis['logic_chain'].append("└─────────────────────────────────────┘")
        
        if dpi_confirms_bearish:
            analysis['logic_chain'].append("")
            analysis['logic_chain'].append(f"✅ DPI={dpi_5d:.1f}%确认：机构也在卖出")
    
    # 场景4：Put主导 + 接近/跌破Put Wall
    elif dominance == 'put_dominant' and position_zone in ['below_pw', 'near_pw']:
        if gap_up_score > 50:
            final_signal = "strong_bullish"
            analysis['prediction'] = f"🚀🚀 PUT UNWINDING 强买压 {neg_stars}"
        elif gap_up_score > 20:
            final_signal = "bullish"
            analysis['prediction'] = f"🚀 PUT UNWINDING 买压 {neg_stars}"
        else:
            final_signal = "bullish_watch"
            analysis['prediction'] = f"📈 偏多观察 {neg_stars}"
        
        analysis['mm_behavior'] = "Put主导 + 接近PW → MM空头将回补 → 买压"
        
        analysis['logic_chain'].append("🟢 【Put主导 + 接近Put Wall】")
        analysis['logic_chain'].append("")
        analysis['logic_chain'].append("做市商行为链：")
        analysis['logic_chain'].append("┌─────────────────────────────────────┐")
        analysis['logic_chain'].append("│ 投资者买入ATM Put → MM卖出Put       │")
        analysis['logic_chain'].append("│ → Short Put = 正Delta暴露           │")
        analysis['logic_chain'].append("│ → MM做空股票对冲（Short Stock）      │")
        analysis['logic_chain'].append("├─────────────────────────────────────┤")
        analysis['logic_chain'].append("│ 周五到期：ITM Put消失                │")
        analysis['logic_chain'].append("│ → MM剩余裸空头股票                   │")
        analysis['logic_chain'].append("├─────────────────────────────────────┤")
        analysis['logic_chain'].append("│ 周一：MM买入平空头 → 买压 → 🚀 高开  │")
        analysis['logic_chain'].append("└─────────────────────────────────────┘")
        
        if dpi_confirms_bullish:
            analysis['logic_chain'].append("")
            analysis['logic_chain'].append(f"✅ DPI={dpi_5d:.1f}%确认：机构也在买入")
        elif dpi_5d and dpi_5d < 48:
            analysis['warnings'].append(f"⚠️ DPI={dpi_5d:.1f}%偏低，机构在卖出，反弹可能受限")
    
    # 场景5：有主导方但位置不理想
    elif dominance == 'call_dominant':
        final_signal = "bearish_watch"
        analysis['prediction'] = f"📉 Call主导 - 等待接近CW {neg_stars}"
        analysis['mm_behavior'] = "Call主导但距离CW较远，等待价格接近再操作"
        
        analysis['logic_chain'].append("🟡 【Call主导但位置不理想】")
        analysis['logic_chain'].append("")
        analysis['logic_chain'].append(f"   • Call主导评分: {call_score}/11分 ✓")
        analysis['logic_chain'].append(f"   • 但距离Call Wall: {dist_to_cw_raw:+.2f}%" if dist_to_cw_raw else "   • 但距离Call Wall: N/A")
        analysis['logic_chain'].append("")
        analysis['logic_chain'].append("→ 等待价格接近Call Wall时再确认做空机会")
    
    elif dominance == 'put_dominant':
        final_signal = "bullish_watch"
        analysis['prediction'] = f"📈 Put主导 - 等待接近PW {neg_stars}"
        analysis['mm_behavior'] = "Put主导但距离PW较远，等待价格接近再操作"
        
        analysis['logic_chain'].append("🟡 【Put主导但位置不理想】")
        analysis['logic_chain'].append("")
        analysis['logic_chain'].append(f"   • Put主导评分: {put_score}/11分 ✓")
        analysis['logic_chain'].append(f"   • 但距离Put Wall: {dist_to_pw_raw:+.2f}%" if dist_to_pw_raw else "   • 但距离Put Wall: N/A")
        analysis['logic_chain'].append("")
        analysis['logic_chain'].append("→ 等待价格接近Put Wall时再确认做多机会")
    
    # 场景5b：偏主导（四维评分不够强）+ 位置配合
    elif dominance == 'call_lean' and position_zone in ['above_cw', 'near_cw']:
        if gap_down_score > 30:
            final_signal = "bearish_watch"
            analysis['prediction'] = f"📉 偏Call + 接近CW - 观察卖压 {neg_stars}"
        else:
            final_signal = "neutral"
            analysis['prediction'] = f"⚖️ 偏Call但分数不足 {neg_stars}"
        analysis['mm_behavior'] = "偏Call主导 + 接近CW，但四维指标不够强"
        
        analysis['logic_chain'].append("🟡 【偏Call主导 + 接近Call Wall】")
        analysis['logic_chain'].append("")
        analysis['logic_chain'].append(f"   • Call评分: {call_score}/11分（未达强主导阈值5分）")
        analysis['logic_chain'].append(f"   • 位置: 接近Call Wall")
        analysis['logic_chain'].append(f"   • Gap Down Score: {gap_down_score:.1f}")
        analysis['logic_chain'].append("")
        analysis['logic_chain'].append("→ 四维指标偏弱，建议谨慎观察")
    
    elif dominance == 'put_lean' and position_zone in ['below_pw', 'near_pw']:
        if gap_up_score > 30:
            final_signal = "bullish_watch"
            analysis['prediction'] = f"📈 偏Put + 接近PW - 观察买压 {neg_stars}"
        else:
            final_signal = "neutral"
            analysis['prediction'] = f"⚖️ 偏Put但分数不足 {neg_stars}"
        analysis['mm_behavior'] = "偏Put主导 + 接近PW，但四维指标不够强"
        
        analysis['logic_chain'].append("🟡 【偏Put主导 + 接近Put Wall】")
        analysis['logic_chain'].append("")
        analysis['logic_chain'].append(f"   • Put评分: {put_score}/11分（未达强主导阈值5分）")
        analysis['logic_chain'].append(f"   • 位置: 接近Put Wall")
        analysis['logic_chain'].append(f"   • Gap Up Score: {gap_up_score:.1f}")
        analysis['logic_chain'].append("")
        analysis['logic_chain'].append("→ 四维指标偏弱，建议谨慎观察")
    
    # 场景6：分数主导（当主导方不明确时看分数）
    elif gap_up_score > gap_down_score * 2 and gap_up_score > 30:
        final_signal = "bullish"
        analysis['prediction'] = f"🚀 高开分主导 ({gap_up_score:.0f}) {neg_stars}"
        analysis['mm_behavior'] = "解绑分数显示买压占优"
        
        analysis['logic_chain'].append("🟡 【分数主导：高开分占优】")
        analysis['logic_chain'].append(f"   • Gap Up Score: {gap_up_score:.1f}")
        analysis['logic_chain'].append(f"   • Gap Down Score: {gap_down_score:.1f}")
        analysis['logic_chain'].append(f"   • 高开分 > 低开分×2，偏向高开")
    
    elif gap_down_score > gap_up_score * 2 and gap_down_score > 30:
        final_signal = "bearish"
        analysis['prediction'] = f"💀 低开分主导 ({gap_down_score:.0f}) {neg_stars}"
        analysis['mm_behavior'] = "解绑分数显示卖压占优"
        
        analysis['logic_chain'].append("🟡 【分数主导：低开分占优】")
        analysis['logic_chain'].append(f"   • Gap Up Score: {gap_up_score:.1f}")
        analysis['logic_chain'].append(f"   • Gap Down Score: {gap_down_score:.1f}")
        analysis['logic_chain'].append(f"   • 低开分 > 高开分×2，偏向低开")
    
    # 场景7：无明确信号
    else:
        final_signal = "neutral"
        analysis['prediction'] = f"⚖️ 方向不明 {neg_stars}"
        analysis['mm_behavior'] = "主导方和位置条件均不满足，观望"
        
        analysis['logic_chain'].append("⚖️ 【无明确信号】")
        analysis['logic_chain'].append("")
        analysis['logic_chain'].append(f"   • Call评分: {call_score}分")
        analysis['logic_chain'].append(f"   • Put评分: {put_score}分")
        analysis['logic_chain'].append(f"   • Gap Up Score: {gap_up_score:.1f}")
        analysis['logic_chain'].append(f"   • Gap Down Score: {gap_down_score:.1f}")
        analysis['logic_chain'].append("")
        analysis['logic_chain'].append("→ 条件不足，建议观望或等待更好时机")
    
    analysis['signal_type'] = final_signal
    
    # 添加预期波动
    if implied_move:
        analysis['logic_chain'].append("")
        analysis['logic_chain'].append(f"📏 预期波动: ±${implied_move:.2f}")
    
    # 添加Options Impact
    if options_impact:
        if options_impact > 30:
            analysis['logic_chain'].append(f"📊 Options Impact: {options_impact:.1f}%（高，期权主导股价）")
        elif options_impact < 10:
            analysis['warnings'].append(f"⚠️ Options Impact={options_impact:.1f}%偏低，基本面可能主导")
    
    return analysis


def analyze_friday_expiry_v3(df):
    """
    分析周五到期Gamma数据 - V3版本
    使用完整5步分析流程
    """
    results = []
    
    for _, row in df.iterrows():
        symbol = row.get('Symbol', '')
        if not symbol or pd.isna(symbol):
            continue
        
        current_price = parse_number_safe(row.get('Current Price'))
        call_wall = parse_number_safe(row.get('Call Wall'))
        put_wall = parse_number_safe(row.get('Put Wall'))
        
        # 做市商解绑分析 V3
        analysis = analyze_mm_unwinding_v3(row)
        
        # 构建结果
        results.append({
            'Symbol': symbol,
            'Current Price': current_price,
            'Call Wall': call_wall,
            'Put Wall': put_wall,
            'Key Delta Strike': analysis['data_summary'].get('key_delta_strike'),
            'NEG': analysis['data_summary'].get('neg', 0),
            'NED': analysis['data_summary'].get('ned', 0),
            'Delta Ratio': analysis['data_summary'].get('delta_ratio'),
            'Volume Ratio': analysis['data_summary'].get('volume_ratio'),
            'Gamma Ratio': analysis['data_summary'].get('gamma_ratio'),
            'Dist_to_CW': analysis['data_summary'].get('dist_to_cw'),
            'Dist_to_PW': analysis['data_summary'].get('dist_to_pw'),
            'Next Exp Call Vol': analysis['data_summary'].get('next_exp_call_vol'),
            'Next Exp Put Vol': analysis['data_summary'].get('next_exp_put_vol'),
            'Gap Up Score': analysis['gap_up_score'],
            'Gap Down Score': analysis['gap_down_score'],
            'Dominance': analysis['dominance'],
            'Position Zone': analysis['position_zone'],
            'Options Impact': analysis['data_summary'].get('options_impact'),
            'DPI': analysis['data_summary'].get('dpi'),
            'DPI 5D': analysis['data_summary'].get('dpi_5d'),
            'Implied Move': analysis['data_summary'].get('implied_move'),
            'Signal Type': analysis['signal_type'],
            'Prediction': analysis['prediction'],
            'MM Behavior': analysis['mm_behavior'],
            'Logic Chain': analysis['logic_chain'],
            'Step Details': analysis['step_details'],
            'Warnings': analysis['warnings'],
            'Pinning Range': f"{put_wall:.0f} - {call_wall:.0f}" if put_wall and call_wall else "N/A"
        })
    
    return pd.DataFrame(results)


# ============================================================
# Streamlit 界面
# ============================================================

def main():
    st.title("🎯 股票波段期权筛选系统")
    st.caption(f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 QQQ/NQ分析", 
        "📊 Equity Hub", 
        "📅 周五到期Gamma",
        "📊 板块资金流", 
        "🔍 个股筛选", 
        "🎯 综合名单"
    ])
    
    # ========== Tab 1: QQQ/NQ盘前分析 ==========
    with tab1:
        st.header("📈 QQQ/NQ 盘前分析")
        
        # 数据输入区
        col_input1, col_input2 = st.columns(2)
        
        with col_input1:
            # QQQ CSV上传
            qqq_csv_file = st.file_uploader(
                "上传QQQ历史数据CSV（可选）",
                type=['csv'],
                key='qqq_csv_upload',
                help="SpotGamma导出的QQQ数据，用于方向性分析"
            )
        
        with col_input2:
            st.info("💡 上传CSV可获得详细的方向性分析和情景分析")
        
        # QQQ/NQ盘前数据粘贴框
        premarket_text = st.text_area(
            "粘贴盘前数据（QQQ和NQ）",
            height=200,
            placeholder="""QQQ盘前现价：__619.14__，昨收__620.78__ 
630 Call Wall 
620 Volatility Trigger 
619 Zero Gamma 
600 Put Wall 

NQ盘前现价__25587__，昨收__25646__，第二列为NQ的数值 
25901 26020 Combo 4 
...""",
            key="premarket_input"
        )
        
        # 解析CSV数据
        csv_data = None
        if qqq_csv_file is not None:
            try:
                qqq_csv_df = pd.read_csv(qqq_csv_file)
                # 取最新一行
                if not qqq_csv_df.empty:
                    csv_data = qqq_csv_df.iloc[-1].to_dict()
                    st.success(f"✅ 已加载QQQ CSV数据")
            except Exception as e:
                st.warning(f"⚠️ CSV读取失败: {e}")
        
        if premarket_text.strip():
            # 解析盘前数据
            premarket_data = parse_qqq_premarket_text(premarket_text)
            
            # 分析（传入CSV数据）
            analysis = analyze_qqq_nq(premarket_data, csv_data)
            
            # 显示分析结果
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("QQQ")
                qqq = analysis.get('qqq', {})
                if qqq.get('current'):
                    change_pct = qqq.get('change_pct', 0)
                    st.metric(
                        "盘前价",
                        f"${qqq['current']:.2f}",
                        f"{change_pct:+.2f}% vs 昨收"
                    )
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        gamma_env = qqq.get('gamma_env', 'N/A')
                        gamma_color = "🟢" if qqq.get('gamma_env_type') == 'positive' else "🔴"
                        st.write(f"**Gamma环境**: {gamma_color} {gamma_env}")
                    with col_b:
                        vol_env = qqq.get('vol_regime', 'N/A')
                        st.write(f"**波动环境**: {vol_env}")
                    
                    # 关键位
                    with st.expander("📊 QQQ关键位置", expanded=True):
                        for name, price in sorted(qqq.get('levels', {}).items(), key=lambda x: -x[1]):
                            if qqq['current'] and abs(price - qqq['current']) / qqq['current'] < 0.005:
                                st.write(f"• **{price:.2f} {name}** ← 当前")
                            else:
                                st.write(f"• {price:.2f} {name}")
            
            with col2:
                st.subheader("NQ")
                nq = analysis.get('nq', {})
                if nq.get('current'):
                    change_pct = nq.get('change_pct', 0)
                    st.metric(
                        "盘前价",
                        f"{nq['current']:.0f}",
                        f"{change_pct:+.2f}% vs 昨收"
                    )
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        gamma_env = nq.get('gamma_env', 'N/A')
                        gamma_color = "🟢" if nq.get('gamma_env_type') == 'positive' else "🔴"
                        st.write(f"**Gamma环境**: {gamma_color} {gamma_env}")
                    with col_b:
                        vol_env = nq.get('vol_regime', 'N/A')
                        st.write(f"**波动环境**: {vol_env}")
                    
                    with st.expander("📊 NQ关键位置", expanded=True):
                        for name, price in sorted(nq.get('levels', {}).items(), key=lambda x: -x[1]):
                            if nq['current'] and abs(price - nq['current']) / nq['current'] < 0.005:
                                st.write(f"• **{price:.0f} {name}** ← 当前")
                            else:
                                st.write(f"• {price:.0f} {name}")
            
            # 交叉验证
            st.subheader("🔍 NQ/QQQ 交叉验证")
            cv = analysis.get('cross_validation', {})
            if cv:
                if cv.get('status') == '矛盾':
                    st.warning(cv.get('message', ''))
                else:
                    st.success(cv.get('message', ''))
            
            st.divider()
            
            # ===== 方向性分析 =====
            if analysis.get('directional', {}).get('items'):
                st.subheader("📈 方向性分析")
                
                directional = analysis['directional']
                for item in directional.get('items', []):
                    st.markdown(item)
                
                st.divider()
            
            # ===== 情景分析 =====
            if analysis.get('scenarios'):
                st.subheader("🔮 情景分析")
                
                scenarios = analysis['scenarios']
                cols = st.columns(len(scenarios))
                
                for i, scenario in enumerate(scenarios):
                    with cols[i]:
                        prob = scenario['probability']
                        color = "🟢" if prob >= 50 else ("🟡" if prob >= 25 else "🔴")
                        st.markdown(f"**{scenario['name']}** ({prob}%) {color}")
                        st.caption(scenario['description'])
                        st.info(f"策略: {scenario['strategy']}")
                
                st.divider()
            
            # ===== 详细操作建议 =====
            st.subheader("📋 详细操作建议")
            
            gamma_env_type = analysis.get('nq', {}).get('gamma_env_type') or analysis.get('qqq', {}).get('gamma_env_type')
            qqq = analysis.get('qqq', {})
            
            if gamma_env_type == 'positive':
                cw = qqq.get('call_wall')
                pw = qqq.get('put_wall')
                zg = qqq.get('zero_gamma')
                vt = qqq.get('vol_trigger')
                
                # 格式化数值
                zg_str = f"{zg:.0f}" if zg else 'N/A'
                pw_str = f"{pw:.0f}" if pw else 'N/A'
                cw_str = f"{cw:.0f}" if cw else 'N/A'
                vt_str = f"{vt:.0f}" if vt else 'N/A'
                
                advice = f"""
                **正Gamma环境 - 均值回归策略**
                
                🎯 **做多区域**: 
                - Zero Gamma ({zg_str}) 附近是最佳做多位置
                - Put Wall ({pw_str}) 是强支撑，可加仓
                
                🎯 **减仓/做空区域**:
                - Call Wall ({cw_str}) 附近减仓或轻仓做空
                - 不追Call Wall突破！正Gamma会压制涨幅
                
                ⚠️ **风险控制**:
                - 止损设在Zero Gamma下方2-3点
                - 如果跌破Zero Gamma，观望等待企稳
                - 如果价格跌破Volatility Trigger ({vt_str})，环境可能转为负Gamma
                """
                st.success(advice)
            elif gamma_env_type == 'negative':
                cw = qqq.get('call_wall')
                pw = qqq.get('put_wall')
                zg = qqq.get('zero_gamma')
                
                # 格式化数值
                zg_str = f"{zg:.0f}" if zg else 'N/A'
                pw_str = f"{pw:.0f}" if pw else 'N/A'
                cw_str = f"{cw:.0f}" if cw else 'N/A'
                
                advice = f"""
                **负Gamma环境 - 趋势跟随策略**
                
                ⚡ **趋势特征**:
                - 波动放大，趋势延续性强
                - Call Wall突破会加速上涨
                - Put Wall跌破会加速下跌
                
                🎯 **操作建议**:
                - 顺势操作，不抄底不摸顶
                - 突破Call Wall ({cw_str}) 可追多
                - 跌破Put Wall ({pw_str}) 可追空
                
                ⚠️ **风险控制**:
                - 严格止损，波动可能很大
                - 减小仓位，负Gamma环境风险高
                - Zero Gamma ({zg_str}) 是关键分界线
                """
                st.error(advice)
            
            st.divider()
            
            # ===== 历史数据图表 =====
            st.subheader("📊 关键位置历史走势")
            
            # 加载历史数据
            qqq_history = load_worksheet_data("QQQ_History") or {}
            today_str = datetime.now().strftime('%Y-%m-%d')
            
            # 保存今日数据按钮
            col1, col2, col3 = st.columns([1,1,2])
            with col1:
                if st.button("💾 保存今日数据", key="save_qqq_history"):
                    if qqq.get('call_wall') and qqq.get('put_wall'):
                        qqq_history[today_str] = {
                            'date': today_str,
                            'call_wall': qqq.get('call_wall'),
                            'put_wall': qqq.get('put_wall'),
                            'zero_gamma': qqq.get('zero_gamma'),
                            'vol_trigger': qqq.get('vol_trigger'),
                            'current_price': qqq.get('current'),
                            'gamma_env': gamma_env_type
                        }
                        save_worksheet_data("QQQ_History", qqq_history)
                        st.success("✅ 已保存今日数据")
                    else:
                        st.warning("⚠️ 缺少关键位数据")
            
            with col2:
                if st.button("🗑️ 清空历史", key="clear_qqq_history"):
                    save_worksheet_data("QQQ_History", {})
                    qqq_history = {}
                    st.success("✅ 已清空")
            
            # 显示历史图表
            if qqq_history and len(qqq_history) >= 2:
                # 构建DataFrame
                history_rows = []
                for date_key, data in sorted(qqq_history.items()):
                    history_rows.append({
                        '日期': date_key,
                        'Call Wall': data.get('call_wall'),
                        'Put Wall': data.get('put_wall'),
                        'Zero Gamma': data.get('zero_gamma'),
                        'Vol Trigger': data.get('vol_trigger'),
                        '收盘价': data.get('current_price')
                    })
                
                history_df = pd.DataFrame(history_rows)
                history_df['日期'] = pd.to_datetime(history_df['日期'])
                history_df = history_df.set_index('日期')
                
                # 绘制图表
                st.line_chart(history_df[['Call Wall', 'Put Wall', 'Zero Gamma', 'Vol Trigger']])
                
                # 显示数据表
                with st.expander("📋 查看历史数据"):
                    st.dataframe(history_df.reset_index(), use_container_width=True, hide_index=True)
            else:
                st.info("💡 保存至少2天的数据后可查看走势图")
        else:
            st.info("👆 请在上方粘贴QQQ/NQ盘前数据")
    
    # ========== Tab 3: 周五到期Gamma分析 ==========
    with tab3:
        st.header("📅 周五到期 Gamma 分析")
        st.caption("分析本周五大量Gamma到期的标的，预测下周一跳空方向（基于做市商解绑效应）")
        
        friday_file = st.file_uploader(
            "上传 Top Gamma Expiring This Friday (CSV/Excel)",
            type=['csv', 'xlsx', 'xls'],
            key='friday_expiry_upload'
        )
        
        if friday_file is not None:
            try:
                # 支持CSV和Excel
                if friday_file.name.endswith('.csv'):
                    friday_df = pd.read_csv(friday_file)
                else:
                    friday_df = pd.read_excel(friday_file)
                
                friday_df = friday_df.dropna(subset=['Symbol'])
                
                st.success(f"✅ 已加载 {len(friday_df)} 只标的")
                
                # 显示可用字段
                with st.expander("📋 检测到的数据字段"):
                    st.write(friday_df.columns.tolist())
                
                # 分析 - 使用V2函数
                friday_results = analyze_friday_expiry_v3(friday_df)
                
                if not friday_results.empty:
                    # 统计
                    st.subheader("📊 做市商解绑信号概览（5步分析）")
                    st.caption("核心逻辑：高冲击判断 → 主导方判断 → 解绑分数 → 位置验证 → 最终信号")
                    
                    strong_bearish = len(friday_results[friday_results['Signal Type'] == 'strong_bearish'])
                    bearish = len(friday_results[friday_results['Signal Type'] == 'bearish'])
                    bearish_watch = len(friday_results[friday_results['Signal Type'] == 'bearish_watch'])
                    strong_bullish = len(friday_results[friday_results['Signal Type'] == 'strong_bullish'])
                    bullish = len(friday_results[friday_results['Signal Type'] == 'bullish'])
                    bullish_watch = len(friday_results[friday_results['Signal Type'] == 'bullish_watch'])
                    low_impact = len(friday_results[friday_results['Signal Type'] == 'low_impact'])
                    neutral_count = len(friday_results[friday_results['Signal Type'] == 'neutral'])
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        total_bullish = strong_bullish + bullish + bullish_watch
                        st.metric("🚀 高开信号", total_bullish, f"强{strong_bullish} 中{bullish} 观察{bullish_watch}")
                    with col2:
                        total_bearish = strong_bearish + bearish + bearish_watch
                        st.metric("💀 低开信号", total_bearish, f"强{strong_bearish} 中{bearish} 观察{bearish_watch}")
                    with col3:
                        st.metric("⚖️ 中性", neutral_count)
                    with col4:
                        st.metric("⚪ 低冲击", low_impact, "浓度不足")
                    with col5:
                        st.metric("📊 总计", len(friday_results))
                        st.metric("⚖️ 中性", neutral_count)
                    with col4:
                        st.metric("⚪ 弱信号", weak_count, "浓度不足")
                    with col5:
                        st.metric("📊 总计", len(friday_results))
                    
                    st.divider()
                    
                    # 强信号标的 - 显示详细逻辑链
                    st.subheader("🎯 信号标的详细分析（5步逻辑）")
                    
                    # 筛选有信号的标的（包括观察信号）
                    signal_results = friday_results[
                        friday_results['Signal Type'].isin([
                            'strong_bearish', 'bearish', 'bearish_watch',
                            'strong_bullish', 'bullish', 'bullish_watch'
                        ])
                    ].sort_values('NEG', ascending=False)
                    
                    if not signal_results.empty:
                        for idx, row in signal_results.iterrows():
                            sig_type = row.get('Signal Type', 'neutral')
                            
                            # 信号样式
                            if sig_type == 'strong_bearish':
                                sig_icon = "💀💀💀"
                                border_color = "#ff4b4b"
                            elif sig_type == 'bearish':
                                sig_icon = "💀"
                                border_color = "#ff6b6b"
                            elif sig_type == 'bearish_watch':
                                sig_icon = "📉"
                                border_color = "#ffa500"
                            elif sig_type == 'strong_bullish':
                                sig_icon = "🚀🚀🚀"
                                border_color = "#00cc00"
                            elif sig_type == 'bullish':
                                sig_icon = "🚀"
                                border_color = "#00dd66"
                            elif sig_type == 'bullish_watch':
                                sig_icon = "📈"
                                border_color = "#00aaff"
                            else:
                                sig_icon = "⚖️"
                                border_color = "#888888"
                            
                            # 标的卡片
                            with st.container():
                                st.markdown(f"""
                                <div style="border-left: 4px solid {border_color}; padding-left: 15px; margin-bottom: 10px;">
                                <h4>{sig_icon} {row['Symbol']} - {row['Prediction']}</h4>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # 核心指标行
                                core_col1, core_col2, core_col3, core_col4, core_col5 = st.columns(5)
                                with core_col1:
                                    st.metric("当前价", f"${row['Current Price']:.2f}" if row['Current Price'] else "N/A")
                                with core_col2:
                                    gap_up = row.get('Gap Up Score', 0)
                                    st.metric("🚀 高开分", f"{gap_up:.1f}")
                                with core_col3:
                                    gap_down = row.get('Gap Down Score', 0)
                                    st.metric("💀 低开分", f"{gap_down:.1f}")
                                with core_col4:
                                    dominance = row.get('Dominance', 'neutral')
                                    dom_text = "Call主导" if dominance == 'call_dominant' else ("Put主导" if dominance == 'put_dominant' else "均衡")
                                    st.metric("主导方", dom_text)
                                with core_col5:
                                    st.metric("NEG", f"{row['NEG']:.1f}%")
                                
                                # 关键位置行
                                pos_col1, pos_col2, pos_col3, pos_col4 = st.columns(4)
                                with pos_col1:
                                    st.metric("Call Wall", f"${row['Call Wall']:.2f}" if row['Call Wall'] else "N/A")
                                with pos_col2:
                                    st.metric("Put Wall", f"${row['Put Wall']:.2f}" if row['Put Wall'] else "N/A")
                                with pos_col3:
                                    dist_cw = row.get('Dist_to_CW')
                                    if dist_cw is not None:
                                        st.metric("距CW", f"{dist_cw:+.2f}%")
                                with pos_col4:
                                    dist_pw = row.get('Dist_to_PW')
                                    if dist_pw is not None:
                                        st.metric("距PW", f"{dist_pw:+.2f}%")
                                
                                # 期权结构行
                                struct_col1, struct_col2, struct_col3, struct_col4 = st.columns(4)
                                with struct_col1:
                                    dr = row.get('Delta Ratio')
                                    if dr is not None:
                                        st.metric("Delta Ratio", f"{dr:.2f}")
                                with struct_col2:
                                    vr = row.get('Volume Ratio')
                                    if vr is not None:
                                        st.metric("Volume Ratio", f"{vr:.2f}")
                                with struct_col3:
                                    call_vol = row.get('Next Exp Call Vol')
                                    if call_vol is not None:
                                        st.metric("Call到期量", f"{call_vol:.1%}")
                                with struct_col4:
                                    put_vol = row.get('Next Exp Put Vol')
                                    if put_vol is not None:
                                        st.metric("Put到期量", f"{put_vol:.1%}")
                                
                                # MM行为简述
                                mm_behavior = row.get('MM Behavior', '')
                                if mm_behavior:
                                    st.info(f"**🏦 做市商行为**: {mm_behavior}")
                                
                                # 详细逻辑链（5步分析）
                                logic_chain = row.get('Logic Chain', [])
                                if logic_chain:
                                    with st.expander("🔍 查看完整5步分析逻辑", expanded=False):
                                        for line in logic_chain:
                                            if line.strip():
                                                # 使用代码块样式显示框架图
                                                if line.startswith("┌") or line.startswith("│") or line.startswith("├") or line.startswith("└"):
                                                    st.code(line, language=None)
                                                else:
                                                    st.markdown(line)
                                
                                # 警告
                                warnings = row.get('Warnings', [])
                                if warnings:
                                    for warning in warnings:
                                        st.warning(warning)
                                
                                # 其他指标
                                with st.expander("📊 辅助指标"):
                                    other_col1, other_col2, other_col3, other_col4 = st.columns(4)
                                    with other_col1:
                                        oi = row.get('Options Impact')
                                        if oi:
                                            st.metric("Options Impact", f"{oi:.1f}%")
                                    with other_col2:
                                        dpi = row.get('DPI')
                                        if dpi:
                                            st.metric("DPI", f"{dpi:.1f}%")
                                    with other_col3:
                                        im = row.get('Implied Move')
                                        if im:
                                            st.metric("Implied Move", f"${im:.2f}")
                                    with other_col4:
                                        kds = row.get('Key Delta Strike')
                                        if kds:
                                            st.metric("Key Delta Strike", f"${kds:.0f}")
                                
                                st.divider()
                    else:
                        st.info("暂无有效信号标的（价格未接近关键位或浓度不足）")
                    
                    # 完整表格
                    with st.expander("📋 查看完整分析表"):
                        display_cols = ['Symbol', 'Current Price', 'Call Wall', 'Put Wall', 
                                       'NEG', 'NED', 'Dist_to_CW', 'Dist_to_PW',
                                       'Position Zone', 'Signal Type', 'Prediction', 'MM Behavior']
                        available_cols = [c for c in display_cols if c in friday_results.columns]
                        st.dataframe(friday_results[available_cols].round(2), use_container_width=True, hide_index=True)
                    
                    # 追踪功能
                    st.subheader("📈 周五到期 追踪验证")
                    st.caption("追踪周期：周五 → 下周二，验证跳空预测准确性")
                    
                    # 加载追踪数据
                    fe_tracking = load_worksheet_data("Friday_Expiry") or {}
                    today_str = datetime.now().strftime('%Y-%m-%d')
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("➕ 添加到追踪", key="add_fe_tracking"):
                            new_count = 0
                            
                            # 只添加有方向信号的标的
                            valid_signals = friday_results[
                                friday_results['Signal Type'].isin(['strong_bearish', 'bearish', 'strong_bullish', 'bullish'])
                            ]
                            
                            for _, row in valid_signals.iterrows():
                                symbol = row['Symbol']
                                
                                if symbol not in fe_tracking:
                                    # 计算追踪结束日期（下周二）
                                    days_to_tuesday = (8 - datetime.now().weekday()) % 7
                                    if days_to_tuesday == 0:
                                        days_to_tuesday = 7
                                    track_end = (datetime.now() + timedelta(days=days_to_tuesday + 1)).strftime('%Y-%m-%d')
                                    
                                    fe_tracking[symbol] = {
                                        'week_start': today_str,
                                        'friday_close': float(row['Current Price']) if row['Current Price'] else 0,
                                        'call_wall': float(row['Call Wall']) if row['Call Wall'] else 0,
                                        'put_wall': float(row['Put Wall']) if row['Put Wall'] else 0,
                                        'neg': row['NEG'],
                                        'ned': row.get('NED'),
                                        'dist_to_cw': row.get('Dist_to_CW'),
                                        'dist_to_pw': row.get('Dist_to_PW'),
                                        'position_zone': row.get('Position Zone'),
                                        'signal_type': row['Signal Type'],
                                        'prediction': row['Prediction'],
                                        'mm_behavior': row.get('MM Behavior', ''),
                                        'logic_chain': row.get('Logic Chain', []),
                                        'monday_open': None,  # 周一手动填入
                                        'tuesday_close': None,  # 周二手动填入
                                        'gap_pct': None,  # 跳空百分比
                                        'trend_pct': None,  # 趋势百分比
                                        'gap_correct': None,  # 跳空方向正确
                                        'trend_correct': None,  # 趋势方向正确
                                        'track_end_date': track_end,
                                        'status': 'tracking'
                                    }
                                    new_count += 1
                            
                            if new_count > 0:
                                save_worksheet_data("Friday_Expiry", fe_tracking)
                                st.success(f"✅ 新增{new_count}个（仅添加有方向信号的标的）")
                    
                    with col2:
                        if st.button("🔄 刷新价格", key="refresh_fe_prices"):
                            updated = 0
                            for symbol in fe_tracking:
                                if fe_tracking[symbol].get('status') == 'tracking':
                                    price = get_current_price(symbol)
                                    if price:
                                        fe_tracking[symbol]['current_price'] = price
                                        friday_close = fe_tracking[symbol].get('friday_close', 0)
                                        if friday_close > 0:
                                            fe_tracking[symbol]['gap_pct'] = ((price - friday_close) / friday_close) * 100
                                        updated += 1
                            
                            save_worksheet_data("Friday_Expiry", fe_tracking)
                            st.success(f"✅ 更新了{updated}个标的价格")
                    
                    with col3:
                        if st.button("🗑️ 清空追踪", key="clear_fe_tracking"):
                            save_worksheet_data("Friday_Expiry", {})
                            fe_tracking = {}
                            st.success("✅ 已清空")
                    
                    # 显示追踪记录
                    if fe_tracking:
                        st.write(f"**追踪中: {len(fe_tracking)}个标的**")
                        
                        # 统计准确率
                        gap_correct = sum(1 for r in fe_tracking.values() if r.get('gap_correct') == True)
                        gap_wrong = sum(1 for r in fe_tracking.values() if r.get('gap_correct') == False)
                        trend_correct = sum(1 for r in fe_tracking.values() if r.get('trend_correct') == True)
                        trend_wrong = sum(1 for r in fe_tracking.values() if r.get('trend_correct') == False)
                        
                        gap_verified = gap_correct + gap_wrong
                        trend_verified = trend_correct + trend_wrong
                        gap_accuracy = (gap_correct / gap_verified * 100) if gap_verified > 0 else 0
                        trend_accuracy = (trend_correct / trend_verified * 100) if trend_verified > 0 else 0
                        
                        stat_col1, stat_col2, stat_col3 = st.columns(3)
                        with stat_col1:
                            if gap_verified > 0:
                                st.metric("跳空准确率", f"{gap_accuracy:.1f}%", f"{gap_correct}/{gap_verified} 正确")
                        with stat_col2:
                            if trend_verified > 0:
                                st.metric("趋势准确率", f"{trend_accuracy:.1f}%", f"{trend_correct}/{trend_verified} 正确")
                        with stat_col3:
                            st.metric("追踪标的", len(fe_tracking))
                        
                        # 追踪表格
                        tracking_rows = []
                        for symbol, record in fe_tracking.items():
                            # 信号图标
                            sig_type = record.get('signal_type', 'neutral')
                            if 'bullish' in sig_type:
                                sig_icon = "📈"
                            elif 'bearish' in sig_type:
                                sig_icon = "📉"
                            else:
                                sig_icon = "⚖️"
                            
                            # 验证状态
                            gap_status = "✅" if record.get('gap_correct') == True else (
                                "❌" if record.get('gap_correct') == False else "⏳"
                            )
                            trend_status = "✅" if record.get('trend_correct') == True else (
                                "❌" if record.get('trend_correct') == False else "⏳"
                            )
                            
                            tracking_rows.append({
                                '标的': symbol,
                                '周五收盘': f"${record.get('friday_close', 0):.2f}" if record.get('friday_close') else "⏳",
                                'NEG': f"{record.get('neg', 0):.1f}%",
                                '距CW%': f"{record.get('dist_to_cw', 0):+.2f}%" if record.get('dist_to_cw') is not None else "-",
                                '距PW%': f"{record.get('dist_to_pw', 0):+.2f}%" if record.get('dist_to_pw') is not None else "-",
                                '预测': f"{sig_icon} {str(record.get('prediction', ''))[:18]}",
                                '周一开盘': f"${record.get('monday_open', 0):.2f}" if record.get('monday_open') else "⏳",
                                '跳空%': f"{record.get('gap_pct', 0):+.2f}%" if record.get('gap_pct') is not None else "-",
                                '周二收盘': f"${record.get('tuesday_close', 0):.2f}" if record.get('tuesday_close') else "⏳",
                                '趋势%': f"{record.get('trend_pct', 0):+.2f}%" if record.get('trend_pct') is not None else "-",
                                '跳空': gap_status,
                                '趋势': trend_status
                            })
                        
                        st.dataframe(pd.DataFrame(tracking_rows), use_container_width=True, hide_index=True)
                        
                        # 自动获取价格按钮
                        st.subheader("📊 自动获取历史价格")
                        st.caption("从Yahoo Finance自动获取周五收盘、周一开盘、周二收盘价格")
                        
                        btn_col1, btn_col2, btn_col3 = st.columns(3)
                        
                        with btn_col1:
                            if st.button("📅 获取周五收盘价", key="fetch_friday_close"):
                                updated = 0
                                failed = []
                                with st.spinner("正在获取周五收盘价..."):
                                    for symbol, record in fe_tracking.items():
                                        if record.get('friday_close'):
                                            continue  # 已有数据跳过
                                        try:
                                            ticker = yf.Ticker(symbol)
                                            # 获取最近5天数据找周五
                                            hist = ticker.history(period="5d")
                                            if not hist.empty:
                                                # 找最近的周五（weekday=4）
                                                for idx in range(len(hist)-1, -1, -1):
                                                    date = hist.index[idx]
                                                    if date.weekday() == 4:  # 周五
                                                        friday_close = hist.iloc[idx]['Close']
                                                        record['friday_close'] = float(friday_close)
                                                        
                                                        # 更新偏离度
                                                        cw = record.get('call_wall', 0)
                                                        pw = record.get('put_wall', 0)
                                                        if cw and cw > 0:
                                                            record['dist_to_cw'] = (friday_close - cw) / friday_close * 100
                                                        if pw and pw > 0:
                                                            record['dist_to_pw'] = (friday_close - pw) / friday_close * 100
                                                        
                                                        updated += 1
                                                        break
                                        except Exception as e:
                                            failed.append(f"{symbol}: {str(e)[:30]}")
                                
                                save_worksheet_data("Friday_Expiry", fe_tracking)
                                st.success(f"✅ 更新了{updated}个标的周五收盘价")
                                if failed:
                                    st.warning(f"⚠️ 失败: {', '.join(failed[:5])}")
                        
                        with btn_col2:
                            if st.button("📅 获取周一开盘价", key="fetch_monday_open"):
                                updated = 0
                                failed = []
                                with st.spinner("正在获取周一开盘价..."):
                                    for symbol, record in fe_tracking.items():
                                        if record.get('monday_open'):
                                            continue
                                        if not record.get('friday_close'):
                                            continue  # 需要先有周五收盘
                                        try:
                                            ticker = yf.Ticker(symbol)
                                            hist = ticker.history(period="5d")
                                            if not hist.empty:
                                                # 找最近的周一（weekday=0）
                                                for idx in range(len(hist)-1, -1, -1):
                                                    date = hist.index[idx]
                                                    if date.weekday() == 0:  # 周一
                                                        monday_open = hist.iloc[idx]['Open']
                                                        record['monday_open'] = float(monday_open)
                                                        
                                                        # 计算跳空
                                                        friday_close = record.get('friday_close', 0)
                                                        if friday_close > 0:
                                                            gap_pct = ((monday_open - friday_close) / friday_close) * 100
                                                            record['gap_pct'] = gap_pct
                                                            
                                                            # 验证跳空方向
                                                            sig_type = record.get('signal_type', 'neutral')
                                                            if sig_type in ['bullish', 'strong_bullish', 'bullish_watch'] and gap_pct > 0:
                                                                record['gap_correct'] = True
                                                            elif sig_type in ['bearish', 'strong_bearish', 'bearish_watch'] and gap_pct < 0:
                                                                record['gap_correct'] = True
                                                            elif 'neutral' in sig_type or 'weak' in sig_type:
                                                                record['gap_correct'] = None
                                                            else:
                                                                record['gap_correct'] = False
                                                        
                                                        updated += 1
                                                        break
                                        except Exception as e:
                                            failed.append(f"{symbol}: {str(e)[:30]}")
                                
                                save_worksheet_data("Friday_Expiry", fe_tracking)
                                st.success(f"✅ 更新了{updated}个标的周一开盘价")
                                if failed:
                                    st.warning(f"⚠️ 失败: {', '.join(failed[:5])}")
                        
                        with btn_col3:
                            if st.button("📅 获取周二收盘价", key="fetch_tuesday_close"):
                                updated = 0
                                failed = []
                                with st.spinner("正在获取周二收盘价..."):
                                    for symbol, record in fe_tracking.items():
                                        if record.get('tuesday_close'):
                                            continue
                                        if not record.get('friday_close'):
                                            continue
                                        try:
                                            ticker = yf.Ticker(symbol)
                                            hist = ticker.history(period="5d")
                                            if not hist.empty:
                                                # 找最近的周二（weekday=1）
                                                for idx in range(len(hist)-1, -1, -1):
                                                    date = hist.index[idx]
                                                    if date.weekday() == 1:  # 周二
                                                        tuesday_close = hist.iloc[idx]['Close']
                                                        record['tuesday_close'] = float(tuesday_close)
                                                        
                                                        # 计算趋势
                                                        friday_close = record.get('friday_close', 0)
                                                        if friday_close > 0:
                                                            trend_pct = ((tuesday_close - friday_close) / friday_close) * 100
                                                            record['trend_pct'] = trend_pct
                                                            
                                                            # 验证趋势方向
                                                            sig_type = record.get('signal_type', 'neutral')
                                                            if sig_type in ['bullish', 'strong_bullish', 'bullish_watch'] and trend_pct > 0:
                                                                record['trend_correct'] = True
                                                            elif sig_type in ['bearish', 'strong_bearish', 'bearish_watch'] and trend_pct < 0:
                                                                record['trend_correct'] = True
                                                            elif 'neutral' in sig_type or 'weak' in sig_type:
                                                                record['trend_correct'] = None
                                                            else:
                                                                record['trend_correct'] = False
                                                            
                                                            record['status'] = 'completed'
                                                        
                                                        updated += 1
                                                        break
                                        except Exception as e:
                                            failed.append(f"{symbol}: {str(e)[:30]}")
                                
                                save_worksheet_data("Friday_Expiry", fe_tracking)
                                st.success(f"✅ 更新了{updated}个标的周二收盘价")
                                if failed:
                                    st.warning(f"⚠️ 失败: {', '.join(failed[:5])}")
                        
                        # 一键获取所有价格
                        st.divider()
                        if st.button("🔄 一键获取所有价格", key="fetch_all_prices"):
                            total_updated = 0
                            with st.spinner("正在获取所有历史价格..."):
                                for symbol, record in fe_tracking.items():
                                    try:
                                        ticker = yf.Ticker(symbol)
                                        hist = ticker.history(period="10d")
                                        if hist.empty:
                                            continue
                                        
                                        # 按日期排序
                                        hist = hist.sort_index()
                                        
                                        # 查找周五、周一、周二
                                        friday_data = None
                                        monday_data = None
                                        tuesday_data = None
                                        
                                        for idx in range(len(hist)):
                                            date = hist.index[idx]
                                            weekday = date.weekday()
                                            
                                            if weekday == 4:  # 周五
                                                friday_data = hist.iloc[idx]
                                            elif weekday == 0 and friday_data is not None:  # 周一（周五后的）
                                                monday_data = hist.iloc[idx]
                                            elif weekday == 1 and monday_data is not None:  # 周二（周一后的）
                                                tuesday_data = hist.iloc[idx]
                                        
                                        updated_this = False
                                        
                                        # 填入周五收盘
                                        if friday_data is not None and not record.get('friday_close'):
                                            friday_close = float(friday_data['Close'])
                                            record['friday_close'] = friday_close
                                            cw = record.get('call_wall', 0)
                                            pw = record.get('put_wall', 0)
                                            if cw and cw > 0:
                                                record['dist_to_cw'] = (friday_close - cw) / friday_close * 100
                                            if pw and pw > 0:
                                                record['dist_to_pw'] = (friday_close - pw) / friday_close * 100
                                            updated_this = True
                                        
                                        # 填入周一开盘
                                        if monday_data is not None and not record.get('monday_open') and record.get('friday_close'):
                                            monday_open = float(monday_data['Open'])
                                            record['monday_open'] = monday_open
                                            friday_close = record['friday_close']
                                            gap_pct = ((monday_open - friday_close) / friday_close) * 100
                                            record['gap_pct'] = gap_pct
                                            
                                            sig_type = record.get('signal_type', 'neutral')
                                            if sig_type in ['bullish', 'strong_bullish', 'bullish_watch'] and gap_pct > 0:
                                                record['gap_correct'] = True
                                            elif sig_type in ['bearish', 'strong_bearish', 'bearish_watch'] and gap_pct < 0:
                                                record['gap_correct'] = True
                                            elif 'neutral' in sig_type or 'weak' in sig_type:
                                                record['gap_correct'] = None
                                            else:
                                                record['gap_correct'] = False
                                            updated_this = True
                                        
                                        # 填入周二收盘
                                        if tuesday_data is not None and not record.get('tuesday_close') and record.get('friday_close'):
                                            tuesday_close = float(tuesday_data['Close'])
                                            record['tuesday_close'] = tuesday_close
                                            friday_close = record['friday_close']
                                            trend_pct = ((tuesday_close - friday_close) / friday_close) * 100
                                            record['trend_pct'] = trend_pct
                                            
                                            sig_type = record.get('signal_type', 'neutral')
                                            if sig_type in ['bullish', 'strong_bullish', 'bullish_watch'] and trend_pct > 0:
                                                record['trend_correct'] = True
                                            elif sig_type in ['bearish', 'strong_bearish', 'bearish_watch'] and trend_pct < 0:
                                                record['trend_correct'] = True
                                            elif 'neutral' in sig_type or 'weak' in sig_type:
                                                record['trend_correct'] = None
                                            else:
                                                record['trend_correct'] = False
                                            record['status'] = 'completed'
                                            updated_this = True
                                        
                                        if updated_this:
                                            total_updated += 1
                                    
                                    except Exception as e:
                                        continue
                            
                            save_worksheet_data("Friday_Expiry", fe_tracking)
                            st.success(f"✅ 更新了{total_updated}个标的的价格数据")
                            st.rerun()
                
            except Exception as e:
                st.error(f"❌ 读取失败: {e}")
        else:
            st.info("👆 请上传 Top Gamma Expiring This Friday CSV文件")
    
    # ========== Tab 4: 板块资金流 ==========
    with tab4:
        st.header("板块资金流扫描")
        st.caption("作为参考信息，辅助判断信号置信度")
        
        if st.button("🔍 扫描板块资金流", key="etf_scan"):
            with st.spinner("正在获取ETF数据..."):
                etf_df = scan_etf_flows()
                
                if etf_df is not None:
                    st.session_state['etf_data'] = etf_df
                    st.session_state['sector_flow_dict'] = get_sector_flow_status(etf_df)
                    
                    st.subheader("全部板块排名")
                    display_cols = ['ETF', '板块', '价格', '>SMA20', '>SMA50', '放量', 'OBV↑', '量比', '20日涨幅%', '评分', '资金流状态']
                    st.dataframe(etf_df[display_cols], use_container_width=True, hide_index=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("🔥 资金流入")
                        inflow = etf_df[etf_df['资金流状态'] == '流入']
                        for _, row in inflow.iterrows():
                            st.write(f"**{row['ETF']}** {row['板块']} (+{row['20日涨幅%']}%)")
                    
                    with col2:
                        st.subheader("⚠️ 资金流出")
                        outflow = etf_df[etf_df['资金流状态'] == '流出']
                        for _, row in outflow.iterrows():
                            st.write(f"**{row['ETF']}** {row['板块']} ({row['20日涨幅%']}%)")
                    
                    with col3:
                        st.subheader("➖ 中性")
                        neutral = etf_df[etf_df['资金流状态'] == '中性']
                        for _, row in neutral.iterrows():
                            st.write(f"**{row['ETF']}** {row['板块']}")
                else:
                    st.error("获取数据失败")
        
        if 'etf_data' in st.session_state:
            st.success("✅ 板块数据已缓存")
    
    # ========== Tab 2: 个股筛选 ==========
    with tab5:
        st.header("个股技术筛选")
        
        # 股票池选择
        pool_option = st.selectbox(
            "选择股票池",
            ["Nasdaq 100", "S&P 500", "Nasdaq 100 + S&P 500", "自定义输入"]
        )
        
        if pool_option == "自定义输入":
            ticker_input = st.text_area(
                "输入股票代码（逗号分隔）",
                value="AAPL,MSFT,NVDA,TSLA",
                height=100
            )
            tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
        else:
            tickers = get_stock_pool(pool_option)
            st.info(f"已选择 **{pool_option}**，共 {len(tickers)} 只股票")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            min_score = st.slider("最低评分", 0, 5, 2)
        with col2:
            direction_filter = st.selectbox("信号方向", ["全部", "看多", "看空"])
        with col3:
            wind_filter = st.selectbox("顺风/逆风", ["全部", "顺风", "逆风"])
        
        if st.button("🔍 开始筛选", key="stock_scan"):
            if not tickers:
                st.warning("请输入至少一个股票代码")
            else:
                # 获取板块资金流数据
                sector_flow_dict = st.session_state.get('sector_flow_dict', {})
                if not sector_flow_dict:
                    st.info("💡 提示：先在「板块资金流」Tab扫描，可获得顺风/逆风标记")
                
                progress = st.progress(0)
                results = []
                
                for i, ticker in enumerate(tickers):
                    progress.progress((i + 1) / len(tickers))
                    result = screen_single_stock(ticker, sector_flow_dict)
                    results.append(result)
                
                progress.empty()
                
                results_df = pd.DataFrame(results)
                st.session_state['stock_results'] = results_df
                
                # 过滤
                filtered = results_df[results_df['passed'] == True].copy()
                
                if min_score > 0:
                    filtered = filtered[filtered['score'] >= min_score]
                
                if direction_filter == "看多":
                    filtered = filtered[filtered['direction'] == '看多']
                elif direction_filter == "看空":
                    filtered = filtered[filtered['direction'] == '看空']
                
                if wind_filter == "顺风":
                    filtered = filtered[filtered['wind'].str.contains('顺风')]
                elif wind_filter == "逆风":
                    filtered = filtered[filtered['wind'].str.contains('逆风')]
                
                st.subheader(f"筛选结果 ({len(filtered)}/{len(results)})")
                
                if len(filtered) > 0:
                    filtered = filtered.sort_values('score', ascending=False)
                    
                    display_df = filtered[['ticker', 'name', 'price', 'direction', 'trend', 'score', 
                                          'rsi', 'atr_pct', 'vol_ratio', 'mapped_sector', 
                                          'sector_flow', 'wind', 'signals']].copy()
                    
                    display_df['price'] = display_df['price'].apply(lambda x: f"${x:.2f}")
                    display_df['atr_pct'] = display_df['atr_pct'].apply(lambda x: f"{x:.1%}")
                    display_df['vol_ratio'] = display_df['vol_ratio'].apply(lambda x: f"{x:.2f}")
                    display_df['rsi'] = display_df['rsi'].apply(lambda x: f"{x:.1f}")
                    display_df['signals'] = display_df['signals'].apply(lambda x: ' | '.join(x) if x else '-')
                    
                    display_df.columns = ['代码', '名称', '价格', '方向', '趋势', '评分', 
                                         'RSI', 'ATR%', '量比', '板块', '板块资金流', '顺逆风', '信号']
                    
                    st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)
                    
                    csv = display_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        "📥 下载CSV",
                        csv,
                        f"stock_screen_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        "text/csv"
                    )
                else:
                    st.warning("无符合条件的股票")
                
                with st.expander("查看未通过筛选的股票"):
                    failed = results_df[results_df['passed'] == False]
                    if len(failed) > 0:
                        st.dataframe(failed[['ticker', 'reason']], use_container_width=True, hide_index=True)
    
    # ========== Tab 3: 综合名单 ==========
    with tab6:
        st.header("综合筛选名单")
        
        if 'stock_results' not in st.session_state:
            st.info("请先在「个股筛选」Tab完成筛选")
        else:
            stock_df = st.session_state['stock_results']
            passed = stock_df[stock_df['passed'] == True].copy()
            passed = passed.sort_values('score', ascending=False)
            
            # 分组显示
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🟢 看多信号")
                bullish = passed[passed['direction'] == '看多']
                
                if len(bullish) > 0:
                    for _, row in bullish.iterrows():
                        wind_icon = row['wind']
                        flow_info = f"板块{row['sector_flow']}" if row['sector_flow'] != '未知' else ""
                        
                        with st.container():
                            st.markdown(f"""
                            **{row['ticker']}** ${row['price']:.2f} | 评分: {row['score']}  
                            {row['trend']} | {row['mapped_sector']} {flow_info} {wind_icon}  
                            信号: {' '.join(row['signals'])}
                            """)
                            st.divider()
                else:
                    st.write("无")
            
            with col2:
                st.subheader("🔴 看空信号")
                bearish = passed[passed['direction'] == '看空']
                
                if len(bearish) > 0:
                    for _, row in bearish.iterrows():
                        wind_icon = row['wind']
                        flow_info = f"板块{row['sector_flow']}" if row['sector_flow'] != '未知' else ""
                        
                        with st.container():
                            st.markdown(f"""
                            **{row['ticker']}** ${row['price']:.2f} | 评分: {row['score']}  
                            {row['trend']} | {row['mapped_sector']} {flow_info} {wind_icon}  
                            信号: {' '.join(row['signals'])}
                            """)
                            st.divider()
                else:
                    st.write("无")
            
            # 统计
            st.subheader("📈 统计")
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            with stat_col1:
                st.metric("总通过", len(passed))
            with stat_col2:
                st.metric("看多", len(bullish))
            with stat_col3:
                st.metric("看空", len(bearish))
            with stat_col4:
                tailwind = len(passed[passed['wind'].str.contains('顺风')])
                st.metric("顺风", tailwind)
    
    # ========== Tab 2: SpotGamma Equity Hub ==========
    with tab2:
        st.header("SpotGamma Equity Hub 分析")
        
        # 参数设置
        with st.expander("⚙️ 分析参数设置"):
            col1, col2, col3 = st.columns(3)
            with col1:
                near_wall_threshold = st.slider("关键位置阈值 (%)", 3, 15, 5, 
                    help="价格距离Put Wall或Call Wall小于此值视为'接近关键位置'")
            with col2:
                min_options_impact = st.slider("最低Options Impact (%)", 0, 50, 20,
                    help="过滤掉期权影响力低的标的")
            with col3:
                high_oi_threshold = st.slider("高OI阈值 (%)", 30, 80, 50,
                    help="Options Impact高于此值视为'期权主导'")
        
        # 文件上传
        col1, col2 = st.columns(2)
        with col1:
            uploaded_file = st.file_uploader("上传Squeeze CSV文件", type=['csv'], key='squeeze_upload')
        with col2:
            cw_increase_file = st.file_uploader("上传Call Wall Increase CSV（可选）", type=['csv'], key='cw_increase_upload')
        
        # 处理CW上移数据
        cw_increase_symbols = set()
        if cw_increase_file is not None:
            try:
                cw_df = pd.read_csv(cw_increase_file)
                cw_df = cw_df.dropna(subset=['Symbol'])
                cw_increase_symbols = set(cw_df['Symbol'].tolist())
                st.success(f"✅ CW上移名单: {len(cw_increase_symbols)}只标的")
            except Exception as e:
                st.warning(f"⚠️ CW上移CSV读取失败: {e}")
        
        if uploaded_file is not None:
            try:
                # 读取并解析SpotGamma数据
                first_line = uploaded_file.readline().decode('utf-8')
                uploaded_file.seek(0)
                
                if 'Ticker Information' in first_line:
                    sg_df = pd.read_csv(uploaded_file, skiprows=1)
                else:
                    sg_df = pd.read_csv(uploaded_file)
                
                sg_df = sg_df.dropna(subset=['Symbol'])
                
                # 处理Delta Ratio中的引号前缀
                if 'Delta Ratio' in sg_df.columns:
                    sg_df['Delta Ratio'] = sg_df['Delta Ratio'].astype(str).str.replace("'", "", regex=False)
                    sg_df['Delta Ratio'] = pd.to_numeric(sg_df['Delta Ratio'], errors='coerce')
                
                # 处理其他数值列
                numeric_cols = ['Current Price', 'Call Wall', 'Put Wall', 'Hedge Wall', 
                               'Options Impact', 'Gamma Ratio', 'Key Gamma Strike', 'Key Delta Strike',
                               'Next Exp Gamma', 'Next Exp Delta', 'Put/Call OI Ratio', 'Volume Ratio']
                for col in numeric_cols:
                    if col in sg_df.columns:
                        sg_df[col] = pd.to_numeric(sg_df[col], errors='coerce')
                
                # 检查必需列
                required_cols = ['Symbol', 'Current Price', 'Delta Ratio', 'Gamma Ratio', 'Put Wall', 'Call Wall']
                missing_cols = [col for col in required_cols if col not in sg_df.columns]
                
                if missing_cols:
                    st.error(f"❌ 数据缺少必需列: {', '.join(missing_cols)}")
                    st.info("请上传包含 Delta Ratio 和 Gamma Ratio 的 SpotGamma Equity Hub 数据")
                    st.write("当前数据列:", list(sg_df.columns))
                else:
                    # ===== 核心分析函数（基于SpotGamma官方定义）=====
                    
                    def get_option_structure(row):
                        """
                        判断期权结构 - 基于回测数据优化
                        
                        【关键发现】
                        - Gamma Ratio > 1: 100%命中Squeeze Up
                        - Volume Ratio < 1: 100%命中Squeeze Up
                        - Delta Ratio相对分散，作为辅助
                        
                        【指标含义】
                        - Delta Ratio = Put Delta ÷ Call Delta（负值，越负=Put Delta越大）
                        - Gamma Ratio = Put Gamma ÷ Call Gamma（>1=Put Gamma主导）
                        - Volume Ratio = ATM Put Vol ÷ ATM Call Vol（<1=Call交易更活跃）
                        """
                        dr = row['Delta Ratio']
                        gr = row['Gamma Ratio']
                        vr = row.get('Volume Ratio', None)
                        
                        if pd.isna(dr) or pd.isna(gr):
                            return "数据缺失", "unknown"
                        
                        # Volume Ratio处理
                        if vr is None or pd.isna(vr):
                            vr = 1.0  # 默认中性
                        
                        # ===== 新的判断逻辑（基于回测数据）=====
                        
                        # Put主导（强）: GR > 1.5 AND DR < -2
                        if gr > 1.5 and dr < -2:
                            return "Put主导", "put_dominant"
                        
                        # Put偏多: GR > 1.3 AND (DR < -1 OR VR < 0.5)
                        elif gr > 1.3 and (dr < -1 or vr < 0.5):
                            return "Put偏多", "put_leaning"
                        
                        # Put轻微: GR > 1（关键阈值，100%命中）
                        elif gr > 1:
                            return "Put轻微", "put_slight"
                        
                        # Call主导（强）: GR < 0.8 AND DR > -0.5
                        elif gr < 0.8 and dr > -0.5:
                            return "Call主导", "call_dominant"
                        
                        # Call偏多: GR < 1 AND DR > -1
                        elif gr < 1 and dr > -1:
                            return "Call偏多", "call_leaning"
                        
                        # 中性
                        else:
                            return "中性", "neutral"
                    
                    def get_volatility_regime(row):
                        """
                        判断波动环境（基于Hedge Wall）
                        官方定义：
                        - 价格 > Hedge Wall → 均值回归环境，波动率低
                        - 价格 < Hedge Wall → 趋势/高波动环境
                        """
                        price = row['Current Price']
                        hw = row.get('Hedge Wall', None)
                        
                        if hw is None or pd.isna(hw) or hw <= 1:
                            return "未知", "unknown"
                        
                        if price > hw:
                            return "均值回归", "mean_reversion"
                        else:
                            return "趋势/高波动", "trending"
                    
                    def get_position_zone(row, threshold):
                        """判断价格位置（相对于Put Wall和Call Wall）"""
                        price = row['Current Price']
                        cw = row['Call Wall']
                        pw = row['Put Wall']
                        
                        dist_to_cw = (cw - price) / price * 100
                        dist_to_pw = (price - pw) / price * 100
                        
                        if dist_to_cw < threshold:
                            return "近Call Wall", dist_to_cw, dist_to_pw
                        elif dist_to_pw < threshold:
                            return "近Put Wall", dist_to_cw, dist_to_pw
                        else:
                            return "中间区域", dist_to_cw, dist_to_pw
                    
                    def get_gamma_magnet(row):
                        """
                        判断Gamma磁吸效应
                        官方定义：股价围绕Key Gamma Strike产生磁吸效应
                        """
                        price = row['Current Price']
                        kgs = row.get('Key Gamma Strike', None)
                        
                        if kgs is None or pd.isna(kgs):
                            return None, None
                        
                        dist_pct = abs(price - kgs) / price * 100
                        if dist_pct < 2:
                            return "强磁吸", dist_pct
                        elif dist_pct < 5:
                            return "弱磁吸", dist_pct
                        else:
                            return "无磁吸", dist_pct
                    
                    def get_trade_signal(position, structure, vol_regime, options_impact, high_oi_thresh, dist_pw, dist_cw, next_gamma, gr=None, vr=None):
                        """
                        生成交易信号 - 基于价格区域 + 做市商盈利逻辑
                        
                        【核心原理】
                        1. 价格区域判断：
                           - 突破CW / 跌破PW → 加速信号
                           - Pin Zone（两边≤5%）→ 区间震荡
                           - 近CW/PW（≤3%）→ 结合期权结构判断
                           - 中间区域（>5%）→ 看期权结构判断方向
                        
                        2. 期权结构：
                           - Put主导 = MM处于负Delta，希望涨（杀Put）
                           - Call主导 = MM处于正Delta，希望跌（杀Call）
                        
                        3. 特殊信号：
                           - 负Gamma螺旋：Put主导+跌破PW → MM被迫追涨杀跌
                           - 弹簧效应：GR极高+VR极低 → Put动能衰竭，超跌反弹潜力
                        """
                        # 判断Squeeze条件
                        has_squeeze_potential = (options_impact >= 20 and 
                                                next_gamma is not None and not pd.isna(next_gamma) and next_gamma >= 0.25)
                        
                        # 置信度
                        if options_impact > high_oi_thresh:
                            confidence = "⭐⭐⭐"
                        elif options_impact > high_oi_thresh * 0.6:
                            confidence = "⭐⭐"
                        else:
                            confidence = "⭐"
                        
                        # 归类期权结构（新增Put轻微）
                        is_put_side = structure in ["Put主导", "Put偏多", "Put轻微"]
                        is_call_side = structure in ["Call主导", "Call偏多"]
                        is_strong_put = structure == "Put主导"
                        is_medium_put = structure == "Put偏多"
                        is_strong_call = structure == "Call主导"
                        
                        # NEG信息
                        neg_str = f"NEG={next_gamma*100:.0f}%" if (next_gamma is not None and not pd.isna(next_gamma)) else "NEG=N/A"
                        
                        # ===== 价格区域判断 =====
                        
                        # 1. 已突破CW（dist_cw < 0）
                        if dist_cw < 0:
                            # Call主导+突破CW = 正Gamma轧空，更强
                            if is_call_side:
                                return (f"🔥⚡ 正Gamma轧空 {confidence}", 
                                       f"Call主导+突破CW，MM被迫疯狂买股对冲，暴涨风险！", 
                                       "bullish")
                            else:
                                return (f"🔥 突破CW加速 {confidence}", 
                                       f"价格已突破Call Wall，MM被迫买股对冲，加速上涨", 
                                       "bullish")
                        
                        # 2. 已跌破PW（dist_pw < 0）- 区分负Gamma螺旋
                        if dist_pw < 0:
                            if is_put_side:
                                # Put主导+跌破PW = 负Gamma螺旋，最危险
                                return (f"💥⚠️ 负Gamma螺旋 {confidence}", 
                                       f"{structure}+跌破PW，MM从'希望涨'变'绝望抛'，跌幅可能失控！", 
                                       "bearish")
                            else:
                                return (f"💥 跌破PW加速 {confidence}", 
                                       f"价格已跌破Put Wall，MM被迫卖股对冲，加速下跌", 
                                       "bearish")
                        
                        # 3. Pin Zone钉价区（两边都在5%以内）
                        if dist_cw <= 5 and dist_pw <= 5:
                            pin_note = ""
                            if has_squeeze_potential:
                                pin_note = f" | ⚠️{neg_str}高，到期可能打破区间"
                            return (f"📊 Pin Zone钉价区 {confidence}", 
                                   f"CW={dist_cw:.1f}%/PW={dist_pw:.1f}%，区间震荡，高抛低吸{pin_note}", 
                                   "neutral")
                        
                        # 4. 近CW区域（≤5%）- 结合期权结构判断
                        if dist_cw <= 5:
                            if dist_cw <= 3:
                                # 阻力区（≤3%）
                                if is_put_side and has_squeeze_potential:
                                    # Put主导+高OI/NEG = MM有动机推涨，可能突破
                                    return (f"🟢 CW突破潜力 {confidence}", 
                                           f"{structure}+距CW仅{dist_cw:.1f}%+{neg_str}，MM有动机推涨突破", 
                                           "bullish")
                                elif is_put_side:
                                    return (f"⏳ CW观察 {confidence}", 
                                           f"{structure}+距CW {dist_cw:.1f}%，关注能否突破", 
                                           "neutral")
                                else:
                                    return (f"🔴 CW阻力区 {confidence}", 
                                           f"距CW仅{dist_cw:.1f}%，{structure}，MM会压价，谨慎做多", 
                                           "bearish_watch")
                            else:
                                # 预警区（3-5%）
                                if is_put_side:
                                    return (f"⏳ 接近CW {confidence}", 
                                           f"{structure}+距CW {dist_cw:.1f}%，接近阻力，关注突破", 
                                           "neutral")
                                else:
                                    return (f"⚠️ 接近CW阻力 {confidence}", 
                                           f"距CW {dist_cw:.1f}%，{structure}，接近阻力区", 
                                           "neutral")
                        
                        # 5. 近PW区域（≤5%）- 结合期权结构判断
                        if dist_pw <= 5:
                            if dist_pw <= 3:
                                # 支撑区（≤3%）
                                if is_call_side and has_squeeze_potential:
                                    # Call主导+高OI/NEG = MM有动机压价，可能跌破
                                    return (f"🔴 PW破位风险 {confidence}", 
                                           f"{structure}+距PW仅{dist_pw:.1f}%+{neg_str}，MM有动机压价破位", 
                                           "bearish")
                                elif is_call_side:
                                    return (f"⏳ PW观察 {confidence}", 
                                           f"{structure}+距PW {dist_pw:.1f}%，关注能否守住", 
                                           "neutral")
                                else:
                                    return (f"🟢 PW支撑区 {confidence}", 
                                           f"距PW仅{dist_pw:.1f}%，{structure}，MM会托价，可博反弹", 
                                           "bullish_watch")
                            else:
                                # 预警区（3-5%）
                                if is_call_side:
                                    return (f"⏳ 接近PW {confidence}", 
                                           f"{structure}+距PW {dist_pw:.1f}%，接近支撑，关注破位", 
                                           "neutral")
                                else:
                                    return (f"⚠️ 接近PW支撑 {confidence}", 
                                           f"距PW {dist_pw:.1f}%，{structure}，接近支撑区", 
                                           "neutral")
                        
                        # ===== 中间区域（dist_pw > 5% AND dist_cw > 5%）：看期权结构 =====
                        
                        # 【弹簧效应检测】GR极高 + VR极低 = Put动能衰竭，超跌反弹潜力
                        if gr is not None and vr is not None:
                            if gr > 2 and vr < 0.3:
                                return (f"🔋 弹簧蓄势 {confidence}", 
                                       f"GR={gr:.1f}极高+VR={vr:.2f}极低，Put动能衰竭，存在超跌反弹潜力！", 
                                       "bullish_watch")
                        
                        # Put侧（主导/偏多/轻微）：MM希望价格涨（杀Put）
                        if is_put_side:
                            if has_squeeze_potential:
                                # 根据Put强度给不同信号
                                if is_strong_put:
                                    return (f"🟢 Squeeze Up潜力 ⭐⭐⭐", 
                                           f"{structure}+OI={options_impact:.0f}%+{neg_str}，MM有动机推涨杀Put", 
                                           "bullish")
                                elif is_medium_put:
                                    return (f"🟢 Squeeze Up潜力 ⭐⭐", 
                                           f"{structure}+OI={options_impact:.0f}%+{neg_str}，MM有动机推涨杀Put", 
                                           "bullish")
                                else:  # Put轻微
                                    return (f"🟢 偏多观察 ⭐", 
                                           f"{structure}(GR>1)+OI={options_impact:.0f}%，轻微偏多，关注", 
                                           "bullish_watch")
                            else:
                                if is_strong_put or is_medium_put:
                                    return (f"🟢 偏多蓄势 {confidence}", 
                                           f"{structure}，MM倾向推涨，但Squeeze条件不完整", 
                                           "bullish_watch")
                                else:  # Put轻微
                                    return (f"⏳ 轻微偏多 {confidence}", 
                                           f"{structure}(GR>1)，轻微偏多但条件不足", 
                                           "neutral")
                        
                        # Call主导/偏多：MM希望价格跌（杀Call）
                        elif is_call_side:
                            if has_squeeze_potential:
                                strength = "⭐⭐⭐" if is_strong_call else "⭐⭐"
                                return (f"🔴 Squeeze Down风险 {strength}", 
                                       f"{structure}+OI={options_impact:.0f}%+{neg_str}，MM有动机压价杀Call", 
                                       "bearish")
                            else:
                                return (f"🔴 偏空蓄势 {confidence}", 
                                       f"{structure}，MM倾向压价，但Squeeze条件不完整", 
                                       "bearish_watch")
                        
                        # 中性结构
                        else:
                            return (f"⚪ 中性观望 {confidence}", 
                                   f"期权结构中性，等待方向明确", 
                                   "neutral")
                        
                        return ("❓ 未知", "数据异常", "unknown")
                    
                    def detect_special_signals(row, dist_to_pw, dist_to_cw):
                        """
                        检测特殊信号和风险（基于官方定义）
                        """
                        signals = []
                        dr = row['Delta Ratio']
                        gr = row['Gamma Ratio']
                        vr = row.get('Volume Ratio', None)
                        oi = row['Options Impact']
                        pc_oi = row.get('Put/Call OI Ratio', None)
                        next_gamma = row.get('Next Exp Gamma', None)
                        next_delta = row.get('Next Exp Delta', None)
                        price = row['Current Price']
                        hw = row.get('Hedge Wall', None)
                        pw = row['Put Wall']
                        
                        # 计算距离Hedge Wall的距离
                        dist_to_hw = None
                        if hw is not None and not pd.isna(hw) and hw > 1:
                            dist_to_hw = ((price - hw) / price) * 100
                        
                        # 0. Gamma陷阱警告（跌破Put Wall + 大量Gamma即将释放）
                        # 做市商正在连环抛售，千万不要抄底！
                        if (dist_to_pw < 0 and  # 已跌破Put Wall
                            next_gamma is not None and not pd.isna(next_gamma) and next_gamma > 0.25):
                            signals.append((
                                "💀 Gamma陷阱", 
                                f"已跌破PW且{next_gamma*100:.0f}%Gamma待释放，MM连环抛售中，勿抄底！",
                                "gamma_trap"
                            ))
                        
                        # 1. 到期反弹潜力（4个条件 + Gamma环境修正）
                        # 逻辑：MM short put→正Delta→卖股票对冲→到期后买回股票平仓→反弹
                        elif (vr is not None and not pd.isna(vr) and vr > 1.2 and  # 条件1: 降低到1.2
                            dr < -3 and  # 条件2: Put Delta占优
                            next_gamma is not None and not pd.isna(next_gamma) and next_gamma > 0.25 and  # 条件3
                            dist_to_pw > 2):  # 条件4: 降低到2%，蓝筹股5%已是巨大回撤
                            
                            # 判断Gamma环境（基于Hedge Wall）
                            if dist_to_hw is not None and dist_to_hw > 0:
                                regime = "正Gamma区"
                                regime_note = "价格>HW，均值回归环境，反弹更稳健"
                            elif dist_to_hw is not None:
                                regime = "负Gamma区"
                                regime_note = "价格<HW，高波动环境，反弹可能剧烈但风险更高"
                            else:
                                regime = "未知环境"
                                regime_note = "Hedge Wall数据缺失"
                            
                            signals.append((
                                f"⚡ 到期反弹【{regime}】", 
                                f"MM short put持空头股票对冲，到期后买回→反弹 | {regime_note} | VR={vr:.1f} DR={dr:.1f} Gamma={next_gamma*100:.0f}%",
                                "bounce"
                            ))
                        
                        # 2. Next Exp Gamma风险（官方：>25%集中，到期前后剧烈波动）
                        if next_gamma is not None and not pd.isna(next_gamma):
                            if next_gamma > 0.5:
                                signals.append(("🔴 Gamma极度集中", f"{next_gamma*100:.0f}%将在下次到期释放，剧烈波动风险", "gamma_risk_high"))
                            elif next_gamma > 0.25:
                                # 只有在没有触发反弹或陷阱信号时才显示一般性警告
                                has_bounce_or_trap = any(s[2] in ['bounce', 'gamma_trap'] for s in signals)
                                if not has_bounce_or_trap:
                                    signals.append(("🟠 Gamma集中警告", f"{next_gamma*100:.0f}%将在下次到期释放（官方警戒线25%）", "gamma_risk_medium"))
                        
                        # 3. Squeeze Up潜力提示（Put主导+远离PW+高OI/NEG）
                        # 核心逻辑：Put主导=MM Short Put多→MM希望涨→有动机推涨杀Put
                        is_put_dominant = dr < -3 or gr > 2
                        has_squeeze_condition = (oi >= 20 and next_gamma is not None and not pd.isna(next_gamma) and next_gamma > 0.25)
                        
                        if is_put_dominant and dist_to_pw > 10 and has_squeeze_condition:
                            if dist_to_cw < 5:
                                signals.append(("🚀 Squeeze Up临界点", f"Put主导+远离PW+近CW，突破后MM买股对冲加速上涨", "squeeze_up_imminent"))
                            elif dist_to_cw < 15:
                                signals.append(("🟢 Squeeze Up潜力", f"Put主导+OI={oi:.0f}%+NEG={next_gamma*100:.0f}%，MM有动机推涨", "squeeze_up_potential"))
                        
                        # 4. Squeeze Down风险提示（Call主导+远离CW+高OI/NEG）
                        # 核心逻辑：Call主导=MM Short Call多→MM希望跌→有动机压价杀Call
                        is_call_dominant = dr > -1 or gr < 1
                        
                        if is_call_dominant and dist_to_cw > 10 and has_squeeze_condition:
                            if dist_to_pw < 5:
                                signals.append(("💥 Squeeze Down临界点", f"Call主导+远离CW+近PW，跌破后MM卖股对冲加速下跌", "squeeze_down_imminent"))
                            elif dist_to_pw < 15:
                                signals.append(("🔴 Squeeze Down风险", f"Call主导+OI={oi:.0f}%+NEG={next_gamma*100:.0f}%，MM有动机压价", "squeeze_down_risk"))
                        
                        # 5. Delta Ratio与P/C OI一致性验证
                        if pc_oi is not None and not pd.isna(pc_oi):
                            if dr > -1 and pc_oi > 1.5:
                                signals.append(("❓ 指标分歧", "Delta偏多但Put OI更多，需谨慎", "divergence"))
                            elif dr < -3 and pc_oi < 0.5:
                                signals.append(("❓ 指标分歧", "Delta偏空但Call OI更多，需谨慎", "divergence"))
                        
                        # 6. Options Impact极端
                        if oi > 100:
                            signals.append(("🔴 期权完全主导", f"OI={oi:.0f}%，股价完全由期权流驱动", "oi_extreme"))
                        
                        # 7. 高Volume Ratio但条件不完整时的提示
                        if (vr is not None and not pd.isna(vr) and vr > 1.2):
                            # 检查是否已经触发了反弹或陷阱信号
                            has_bounce_or_trap = any(s[2] in ['bounce', 'gamma_trap'] for s in signals)
                            if not has_bounce_or_trap:
                                missing = []
                                if dr >= -3:
                                    missing.append("DR未偏Put(<-3)")
                                if next_gamma is None or pd.isna(next_gamma) or next_gamma <= 0.25:
                                    missing.append("Gamma未集中(>25%)")
                                if dist_to_pw <= 2:
                                    missing.append("太近PW(<2%)")
                                if dist_to_pw < 0:
                                    missing.append("已破PW")
                                if missing:
                                    signals.append((
                                        "📊 高VR观察", 
                                        f"ATM Put活跃(VR={vr:.1f})，但缺少: {', '.join(missing)}",
                                        "vr_watch"
                                    ))
                        
                        # 8. 负Gamma区高波动警告（价格低于Hedge Wall）
                        if dist_to_hw is not None and dist_to_hw < -5 and oi > 30:
                            signals.append((
                                "⚠️ 深度负Gamma区", 
                                f"价格低于HW {abs(dist_to_hw):.1f}%，高波动趋势环境，波动可能放大",
                                "negative_gamma_zone"
                            ))
                        
                        return signals
                    
                    # ===== 应用分析函数 =====
                    
                    # 计算距离
                    sg_df['Dist_to_PW_%'] = ((sg_df['Current Price'] - sg_df['Put Wall']) / sg_df['Put Wall'] * 100).round(1)
                    sg_df['Dist_to_CW_%'] = ((sg_df['Call Wall'] - sg_df['Current Price']) / sg_df['Current Price'] * 100).round(1)
                    
                    # 期权结构
                    structure_results = sg_df.apply(get_option_structure, axis=1)
                    sg_df['Option_Structure'] = structure_results.apply(lambda x: x[0])
                    sg_df['Structure_Type'] = structure_results.apply(lambda x: x[1])
                    
                    # 波动环境（基于Hedge Wall）
                    vol_regime_results = sg_df.apply(get_volatility_regime, axis=1)
                    sg_df['Vol_Regime'] = vol_regime_results.apply(lambda x: x[0])
                    sg_df['Vol_Regime_Type'] = vol_regime_results.apply(lambda x: x[1])
                    
                    # 价格位置
                    position_results = sg_df.apply(lambda row: get_position_zone(row, near_wall_threshold), axis=1)
                    sg_df['Price_Position'] = position_results.apply(lambda x: x[0])
                    sg_df['Dist_CW_Calc'] = position_results.apply(lambda x: x[1])
                    sg_df['Dist_PW_Calc'] = position_results.apply(lambda x: x[2])
                    
                    # Gamma磁吸效应
                    magnet_results = sg_df.apply(get_gamma_magnet, axis=1)
                    sg_df['Gamma_Magnet'] = magnet_results.apply(lambda x: x[0])
                    sg_df['Dist_to_KGS'] = magnet_results.apply(lambda x: x[1])
                    
                    # 交易信号（基于做市商盈利逻辑）
                    signal_results = sg_df.apply(
                        lambda row: get_trade_signal(
                            row['Price_Position'], 
                            row['Option_Structure'],
                            row['Vol_Regime'],
                            row['Options Impact'], 
                            high_oi_threshold,
                            row['Dist_PW_Calc'],  # 距离Put Wall
                            row['Dist_CW_Calc'],  # 距离Call Wall
                            row['Next Exp Gamma'] if 'Next Exp Gamma' in row and pd.notna(row['Next Exp Gamma']) else None,
                            row['Gamma Ratio'] if 'Gamma Ratio' in row and pd.notna(row['Gamma Ratio']) else None,
                            row['Volume Ratio'] if 'Volume Ratio' in row and pd.notna(row['Volume Ratio']) else None
                        ), axis=1)
                    sg_df['Trade_Signal'] = signal_results.apply(lambda x: x[0])
                    sg_df['Signal_Logic'] = signal_results.apply(lambda x: x[1])
                    sg_df['Signal_Type'] = signal_results.apply(lambda x: x[2])
                    
                    # ===== CW上移叠加逻辑 =====
                    sg_df['CW_Increase'] = sg_df['Symbol'].isin(cw_increase_symbols)
                    
                    def apply_cw_increase_boost(row):
                        """应用CW上移叠加"""
                        if not row['CW_Increase']:
                            return row['Trade_Signal'], row['Signal_Logic'], row['Signal_Type']
                        
                        signal = row['Trade_Signal']
                        logic = row['Signal_Logic']
                        sig_type = row['Signal_Type']
                        
                        # 做多信号 + CW上移 → 增强
                        if sig_type in ['bullish', 'bullish_watch']:
                            signal = signal.replace("⭐⭐", "⭐⭐⭐").replace("⭐", "⭐⭐") + " 🚀CW↑"
                            logic = logic + " | 🚀CW上移确认，上方空间打开"
                        # 阻力减仓不叠加（信号矛盾）
                        elif 'CW阻力' in signal or '阻力区' in signal:
                            pass  # 保持原信号
                        # 做空信号 + CW上移 → 冲突警告
                        elif sig_type in ['bearish', 'bearish_watch']:
                            signal = signal + " ⚠️CW↑冲突"
                            logic = logic + " | ⚠️CW上移与做空信号冲突，谨慎"
                        # 中性信号 + CW上移 → 偏多
                        elif sig_type == 'neutral':
                            signal = "📈 CW上移偏多 ⭐"
                            logic = logic + " | CW上移，偏多观察"
                            sig_type = 'bullish_watch'
                        
                        return signal, logic, sig_type
                    
                    cw_boost_results = sg_df.apply(apply_cw_increase_boost, axis=1)
                    sg_df['Trade_Signal'] = cw_boost_results.apply(lambda x: x[0])
                    sg_df['Signal_Logic'] = cw_boost_results.apply(lambda x: x[1])
                    sg_df['Signal_Type'] = cw_boost_results.apply(lambda x: x[2])
                    
                    # 特殊信号检测
                    sg_df['Special_Signals'] = sg_df.apply(
                        lambda row: detect_special_signals(row, row['Dist_PW_Calc'], row['Dist_CW_Calc']), axis=1)
                    
                    # 过滤低OI标的
                    sg_filtered = sg_df[sg_df['Options Impact'] >= min_options_impact].copy()
                    
                    # ===== 显示统计 =====
                    st.subheader("📊 分析概览")
                    
                    # 统计各类信号
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    
                    bullish_count = len(sg_filtered[sg_filtered['Signal_Type'] == 'bullish'])
                    bearish_count = len(sg_filtered[sg_filtered['Signal_Type'] == 'bearish'])
                    watch_bull = len(sg_filtered[sg_filtered['Signal_Type'] == 'bullish_watch'])
                    watch_bear = len(sg_filtered[sg_filtered['Signal_Type'] == 'bearish_watch'])
                    spring_count = len(sg_filtered[sg_filtered['Trade_Signal'].str.contains('弹簧蓄势', na=False)])
                    
                    # 统计波动环境
                    mean_rev_count = len(sg_filtered[sg_filtered['Vol_Regime_Type'] == 'mean_reversion'])
                    trending_count = len(sg_filtered[sg_filtered['Vol_Regime_Type'] == 'trending'])
                    cw_increase_count = len(sg_filtered[sg_filtered['CW_Increase'] == True])
                    
                    with col1:
                        st.metric("🟢 高确信做多", bullish_count)
                    with col2:
                        st.metric("🔴 高确信做空", bearish_count)
                    with col3:
                        st.metric("🔋 弹簧蓄势", spring_count)
                    with col4:
                        st.metric("🟢 偏多观察", watch_bull)
                    with col5:
                        st.metric("🔴 偏空观察", watch_bear)
                    with col6:
                        st.metric("🚀 CW上移", cw_increase_count, help="Call Wall上移确认")
                    
                    st.caption(f"已分析 {len(sg_filtered)} 只标的 (Options Impact ≥ {min_options_impact}%)")
                    
                    # 三列分布统计
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("**期权结构:**")
                        for struct, count in sg_filtered['Option_Structure'].value_counts().items():
                            st.write(f"  {struct}: {count}")
                    with col2:
                        st.markdown("**价格位置:**")
                        for pos, count in sg_filtered['Price_Position'].value_counts().items():
                            st.write(f"  {pos}: {count}")
                    with col3:
                        st.markdown("**波动环境:**")
                        for regime, count in sg_filtered['Vol_Regime'].value_counts().items():
                            st.write(f"  {regime}: {count}")
                    
                    # ===== 🟢 高确信做多信号 =====
                    st.subheader("🟢 高确信做多信号")
                    st.caption("中间区域: Put主导+高OI+高NEG → Squeeze Up | 近PW(≤3%): 支撑区可博反弹 | 突破CW: 加速上涨")
                    
                    bullish_signals = sg_filtered[sg_filtered['Signal_Type'] == 'bullish'].copy()
                    bullish_signals = bullish_signals.sort_values('Options Impact', ascending=False)
                    
                    if len(bullish_signals) > 0:
                        for _, row in bullish_signals.iterrows():
                            special_sigs = row['Special_Signals']
                            special_str = ''
                            if special_sigs:
                                special_str = '\n'.join([f"  - {s[0]}: {s[1]}" for s in special_sigs])
                            
                            # 磁吸效应
                            magnet_str = f" | 磁吸: {row['Gamma_Magnet']}" if row['Gamma_Magnet'] else ""
                            
                            with st.container():
                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    st.markdown(f"**{row['Symbol']}** ${row['Current Price']:.2f}")
                                    st.caption(f"{row['Trade_Signal']}")
                                with col2:
                                    st.markdown(f"""
                                    - **位置**: {row['Price_Position']} | **结构**: {row['Option_Structure']} | **环境**: {row['Vol_Regime']}
                                    - DR: {row['Delta Ratio']:.2f} | GR: {row['Gamma Ratio']:.2f} | OI: {row['Options Impact']:.1f}%{magnet_str}
                                    - PW: {row['Put Wall']} → 现价 → CW: {row['Call Wall']}
                                    - 逻辑: {row['Signal_Logic']}
                                    {f'- **特殊信号**:{chr(10)}{special_str}' if special_str else ''}
                                    """)
                                st.divider()
                    else:
                        st.info("无高确信做多信号")
                    
                    # ===== 🔴 高确信做空信号 =====
                    st.subheader("🔴 高确信做空信号")
                    st.caption("中间区域: Call主导+高OI+高NEG → Squeeze Down | 近CW(≤3%): 阻力区谨慎做多 | 跌破PW: 加速下跌")
                    
                    bearish_signals = sg_filtered[sg_filtered['Signal_Type'] == 'bearish'].copy()
                    bearish_signals = bearish_signals.sort_values('Options Impact', ascending=False)
                    
                    if len(bearish_signals) > 0:
                        for _, row in bearish_signals.iterrows():
                            special_sigs = row['Special_Signals']
                            special_str = ''
                            if special_sigs:
                                special_str = '\n'.join([f"  - {s[0]}: {s[1]}" for s in special_sigs])
                            
                            magnet_str = f" | 磁吸: {row['Gamma_Magnet']}" if row['Gamma_Magnet'] else ""
                            
                            with st.container():
                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    st.markdown(f"**{row['Symbol']}** ${row['Current Price']:.2f}")
                                    st.caption(f"{row['Trade_Signal']}")
                                with col2:
                                    st.markdown(f"""
                                    - **位置**: {row['Price_Position']} | **结构**: {row['Option_Structure']} | **环境**: {row['Vol_Regime']}
                                    - DR: {row['Delta Ratio']:.2f} | GR: {row['Gamma Ratio']:.2f} | OI: {row['Options Impact']:.1f}%{magnet_str}
                                    - PW: {row['Put Wall']} → 现价 → CW: {row['Call Wall']}
                                    - 逻辑: {row['Signal_Logic']}
                                    {f'- **特殊信号**:{chr(10)}{special_str}' if special_str else ''}
                                    """)
                                st.divider()
                    else:
                        st.info("无高确信做空信号")
                    
                    # ===== 🔋 弹簧蓄势信号 =====
                    st.subheader("🔋 弹簧蓄势信号")
                    st.caption("GR极高(>2) + VR极低(<0.3) = Put动能衰竭，存在超跌反弹潜力！")
                    
                    # 筛选弹簧蓄势信号
                    spring_signals = sg_filtered[sg_filtered['Trade_Signal'].str.contains('弹簧蓄势', na=False)].copy()
                    spring_signals = spring_signals.sort_values('Options Impact', ascending=False)
                    
                    if len(spring_signals) > 0:
                        for _, row in spring_signals.iterrows():
                            special_sigs = row['Special_Signals']
                            special_str = ''
                            if special_sigs:
                                special_str = '\n'.join([f"  - {s[0]}: {s[1]}" for s in special_sigs])
                            
                            magnet_str = f" | 磁吸: {row['Gamma_Magnet']}" if row['Gamma_Magnet'] else ""
                            
                            with st.container():
                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    st.markdown(f"**{row['Symbol']}** ${row['Current Price']:.2f}")
                                    st.caption(f"{row['Trade_Signal']}")
                                with col2:
                                    vr_val = row['Volume Ratio'] if pd.notna(row['Volume Ratio']) else 0
                                    st.markdown(f"""
                                    - **位置**: {row['Price_Position']} | **结构**: {row['Option_Structure']} | **环境**: {row['Vol_Regime']}
                                    - DR: {row['Delta Ratio']:.2f} | **GR: {row['Gamma Ratio']:.2f}** | **VR: {vr_val:.4f}** | OI: {row['Options Impact']:.1f}%{magnet_str}
                                    - PW: {row['Put Wall']} → 现价 → CW: {row['Call Wall']}
                                    - 逻辑: {row['Signal_Logic']}
                                    {f'- **特殊信号**:{chr(10)}{special_str}' if special_str else ''}
                                    """)
                                st.divider()
                    else:
                        st.info("无弹簧蓄势信号")
                    
                    # ===== 观察名单 =====
                    with st.expander("👀 观察名单（等待接近关键位置）"):
                        watch_signals = sg_filtered[sg_filtered['Signal_Type'].isin(['bullish_watch', 'bearish_watch'])].copy()
                        watch_signals = watch_signals.sort_values('Options Impact', ascending=False)
                        
                        if len(watch_signals) > 0:
                            display_cols = ['Symbol', 'Current Price', 'Trade_Signal', 'Price_Position', 
                                          'Option_Structure', 'Vol_Regime', 'Delta Ratio', 'Gamma Ratio', 'Options Impact',
                                          'Put Wall', 'Call Wall']
                            available_cols = [c for c in display_cols if c in watch_signals.columns]
                            st.dataframe(watch_signals[available_cols].round(2), use_container_width=True, hide_index=True)
                        else:
                            st.info("无观察标的")
                    
                    # ===== 特殊信号汇总 =====
                    with st.expander("⚡ 特殊信号汇总（Gamma陷阱/反弹潜力/到期风险）"):
                        has_special = sg_filtered[sg_filtered['Special_Signals'].apply(len) > 0].copy()
                        
                        if len(has_special) > 0:
                            # 分类显示
                            gamma_traps = []
                            bounce_candidates = []
                            gamma_risks = []
                            squeeze_risks = []
                            negative_gamma_zones = []
                            divergences = []
                            vr_watches = []
                            
                            for _, row in has_special.iterrows():
                                for sig in row['Special_Signals']:
                                    sig_type = sig[2]
                                    entry = f"**{row['Symbol']}** ${row['Current Price']:.2f}: {sig[1]}"
                                    
                                    if sig_type == 'gamma_trap':
                                        gamma_traps.append(entry)
                                    elif sig_type == 'bounce':
                                        bounce_candidates.append(entry)
                                    elif sig_type in ['gamma_risk_high', 'gamma_risk_medium']:
                                        gamma_risks.append(entry)
                                    elif sig_type in ['short_squeeze', 'long_liquidation']:
                                        squeeze_risks.append(entry)
                                    elif sig_type == 'negative_gamma_zone':
                                        negative_gamma_zones.append(entry)
                                    elif sig_type == 'divergence':
                                        divergences.append(entry)
                                    elif sig_type == 'vr_watch':
                                        vr_watches.append(entry)
                            
                            # 按优先级显示
                            if gamma_traps:
                                st.markdown("**💀 Gamma陷阱（勿抄底！）:**")
                                for item in gamma_traps:
                                    st.error(item)
                            
                            if bounce_candidates:
                                st.markdown("**⚡ 到期反弹潜力:**")
                                for item in bounce_candidates:
                                    st.success(item)
                            
                            if gamma_risks:
                                st.markdown("**🔴 到期Gamma集中:**")
                                for item in gamma_risks:
                                    st.warning(item)
                            
                            if squeeze_risks:
                                st.markdown("**⚠️ 挤压/踩踏风险:**")
                                for item in squeeze_risks:
                                    st.warning(item)
                            
                            if negative_gamma_zones:
                                st.markdown("**⚠️ 深度负Gamma区:**")
                                for item in negative_gamma_zones:
                                    st.warning(item)
                            
                            if divergences:
                                st.markdown("**❓ 指标分歧:**")
                                for item in divergences:
                                    st.info(item)
                            
                            if vr_watches:
                                st.markdown("**📊 高VR观察（条件不完整）:**")
                                for item in vr_watches:
                                    st.info(item)
                        else:
                            st.info("无特殊信号")
                    
                    # ===== 完整分析表 =====
                    st.subheader("📋 完整分析表")
                    
                    # 添加过滤选项
                    show_all_stocks = st.checkbox(
                        "显示所有股票（忽略Options Impact过滤）", 
                        value=False,
                        key="show_all_stocks",
                        help=f"勾选后显示CSV中所有{len(sg_df)}只股票，否则只显示Options Impact ≥ {min_options_impact}%的{len(sg_filtered)}只"
                    )
                    
                    display_df = sg_df if show_all_stocks else sg_filtered
                    
                    full_cols = ['Symbol', 'Current Price', 'Trade_Signal', 'Price_Position', 
                                'Option_Structure', 'Vol_Regime', 'Gamma_Magnet', 'Delta Ratio', 'Gamma Ratio',
                                'Put Wall', 'Call Wall', 'Hedge Wall', 'Dist_to_PW_%', 'Dist_to_CW_%', 
                                'Options Impact', 'Volume Ratio', 'Next Exp Gamma', 'CW_Increase']
                    available_cols = [c for c in full_cols if c in display_df.columns]
                    df_sorted = display_df.sort_values('Options Impact', ascending=False)
                    
                    st.caption(f"显示: {len(display_df)} 只标的 {'(全部)' if show_all_stocks else f'(Options Impact ≥ {min_options_impact}%)'}")
                    st.dataframe(df_sorted[available_cols].round(2), use_container_width=True, hide_index=True)
                    
                    # ===== 交叉验证 =====
                    st.subheader("🎯 与技术筛选交叉验证")
                    
                    if 'stock_results' in st.session_state:
                        watchlist = st.session_state['stock_results']
                        passed_tickers = watchlist[watchlist['passed'] == True]['ticker'].tolist()
                        
                        # 找出同时在两个名单中的股票
                        sg_tickers = sg_filtered['Symbol'].tolist()
                        overlap = [t for t in sg_tickers if t in passed_tickers]
                        
                        if overlap:
                            st.success(f"✅ 同时出现在两个名单: **{', '.join(overlap)}**")
                            
                            for ticker in overlap:
                                sg_row = sg_filtered[sg_filtered['Symbol'] == ticker].iloc[0]
                                stock_row = watchlist[watchlist['ticker'] == ticker].iloc[0]
                                
                                # 判断信号是否一致
                                tech_direction = stock_row['direction']
                                sg_signal = sg_row['Trade_Signal']
                                sg_type = sg_row['Signal_Type']
                                
                                # 方向一致性判断
                                tech_bullish = '多' in tech_direction
                                tech_bearish = '空' in tech_direction
                                sg_bullish = sg_type in ['bullish', 'bullish_watch']
                                sg_bearish = sg_type in ['bearish', 'bearish_watch']
                                
                                if (tech_bullish and sg_bullish) or (tech_bearish and sg_bearish):
                                    consistency = "✅ 方向一致"
                                elif sg_type == 'neutral':
                                    consistency = "⚪ Gamma中性"
                                else:
                                    consistency = "⚠️ 方向冲突"
                                
                                # 特殊信号
                                special_sigs = sg_row['Special_Signals']
                                special_str = ''
                                if special_sigs:
                                    special_str = ' | '.join([s[0] for s in special_sigs])
                                
                                with st.container():
                                    st.markdown(f"""
                                    ---
                                    **{ticker}** - {consistency}
                                    - 技术信号: {tech_direction} | 评分: {stock_row['score']} | {' '.join(stock_row['signals'])}
                                    - Gamma信号: {sg_signal}
                                    - 位置: {sg_row['Price_Position']} | 结构: {sg_row['Option_Structure']} | 环境: {sg_row['Vol_Regime']}
                                    - DR: {sg_row['Delta Ratio']:.2f} | GR: {sg_row['Gamma Ratio']:.2f} | OI: {sg_row['Options Impact']:.1f}%
                                    - PW: {sg_row['Put Wall']} | CW: {sg_row['Call Wall']} | HW: {sg_row.get('Hedge Wall', 'N/A')}
                                    {f'- **特殊信号**: {special_str}' if special_str else ''}
                                    """)
                        else:
                            st.info("无重叠股票。技术筛选名单中的股票未出现在SpotGamma数据中。")
                    else:
                        st.info("💡 提示：先在「个股筛选」Tab完成筛选，可进行交叉验证")
                    
                    # ===== 交易计划 =====
                    st.subheader("📈 交易计划")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 🟢 做多计划")
                        if len(bullish_signals) > 0:
                            for _, row in bullish_signals.head(5).iterrows():
                                if row['Price_Position'] == '近Call Wall':
                                    entry = f"突破 {row['Call Wall']:.0f} 确认"
                                    stop = f"{row['Call Wall'] * 0.97:.0f}"
                                    target = f"{row['Call Wall'] * 1.05:.0f}+"
                                    strategy = "突破追多"
                                else:  # 近Put Wall - 反弹做多
                                    entry = f"{row['Put Wall']:.0f} - {row['Current Price']:.0f}"
                                    stop = f"{row['Put Wall'] * 0.97:.0f}"
                                    target = f"{row['Call Wall']:.0f}"
                                    strategy = "支撑反弹"
                                
                                st.markdown(f"""
                                **{row['Symbol']}** [{strategy}]
                                - 入场: {entry}
                                - 止损: {stop}
                                - 目标: {target}
                                - OI: {row['Options Impact']:.0f}%
                                """)
                                st.divider()
                        else:
                            st.info("无高确信做多信号")
                    
                    with col2:
                        st.markdown("### 🔴 做空计划")
                        if len(bearish_signals) > 0:
                            for _, row in bearish_signals.head(5).iterrows():
                                if row['Price_Position'] == '近Put Wall':
                                    entry = f"跌破 {row['Put Wall']:.0f} 确认"
                                    stop = f"{row['Put Wall'] * 1.03:.0f}"
                                    target = f"{row['Put Wall'] * 0.95:.0f}-"
                                    strategy = "破位追空"
                                else:  # 近Call Wall - 压力做空
                                    entry = f"{row['Current Price']:.0f} - {row['Call Wall']:.0f}"
                                    stop = f"{row['Call Wall'] * 1.03:.0f}"
                                    target = f"{row['Put Wall']:.0f}"
                                    strategy = "阻力回落"
                                
                                st.markdown(f"""
                                **{row['Symbol']}** [{strategy}]
                                - 入场: {entry}
                                - 止损: {stop}
                                - 目标: {target}
                                - OI: {row['Options Impact']:.0f}%
                                """)
                                st.divider()
                        else:
                            st.info("无高确信做空信号")
                    
                    # ===== Squeeze追踪面板 =====
                    st.subheader("📈 Squeeze追踪面板")
                    
                    # 加载追踪数据（这会设置gsheets_connected状态）
                    tracking_data = load_tracking_data()
                    today_str = datetime.now().strftime('%Y-%m-%d')
                    
                    # Google Sheets连接状态显示（在load之后）
                    gsheets_status = st.session_state.get('gsheets_connected', False)
                    if gsheets_status:
                        st.caption(f"☁️ **云端同步已启用** (Google Sheets: {GSHEETS_SPREADSHEET_NAME}) | Squeeze标准: ≥{SQUEEZE_THRESHOLD}%涨幅")
                    else:
                        st.caption(f"💾 **本地存储模式** ({os.path.abspath(TRACKING_FILE)}) | Squeeze标准: ≥{SQUEEZE_THRESHOLD}%涨幅")
                        if GSHEETS_AVAILABLE:
                            st.info("💡 Google Sheets凭证未配置或连接失败，数据仅保存在本地。重启后数据可能丢失。")
                    
                    # 识别新标的并添加到追踪
                    new_symbols = []
                    updated_signals = []
                    for _, row in sg_filtered.iterrows():
                        symbol = row['Symbol']
                        # 正确获取Trade_Signal
                        signal_type = row['Trade_Signal'] if 'Trade_Signal' in row.index and pd.notna(row['Trade_Signal']) else '未知信号'
                        
                        if symbol not in tracking_data:
                            # 新标的
                            tracking_data[symbol] = add_new_tracking(symbol, row, signal_type, today_str)
                            new_symbols.append(symbol)
                        else:
                            # 已存在的标的，更新信号类型（如果信号变化）
                            old_signal = tracking_data[symbol].get('signal_type', '')
                            if signal_type != old_signal and signal_type != '未知信号':
                                tracking_data[symbol]['signal_type'] = signal_type
                                updated_signals.append(f"{symbol}: {old_signal[:15]}→{signal_type[:15]}")
                            tracking_data[symbol]['is_new'] = False
                    
                    # 保存更新
                    if new_symbols or updated_signals:
                        save_tracking_data(tracking_data)
                        if new_symbols:
                            st.success(f"🆕 新增追踪: {', '.join(new_symbols)}")
                        if updated_signals:
                            st.info(f"🔄 信号更新: {'; '.join(updated_signals)}")
                    
                    # 操作按钮行
                    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                    with col1:
                        refresh_btn = st.button("🔄 刷新价格", type="primary")
                    with col2:
                        if st.button("☁️ 强制同步云端"):
                            success = save_tracking_to_gsheets(tracking_data)
                            if success:
                                st.success("✅ 已同步到Google Sheets")
                            else:
                                st.error("❌ 同步失败，请检查凭证配置")
                    with col3:
                        clear_completed = st.button("🗑️ 清除已完成")
                    with col4:
                        if st.button("🗑️ 清空所有追踪"):
                            tracking_data = {}
                            save_tracking_data(tracking_data)
                            st.rerun()
                    
                    if clear_completed:
                        tracking_data = {k: v for k, v in tracking_data.items() if v.get('status') != 'completed'}
                        save_tracking_data(tracking_data)
                        st.rerun()
                    
                    if refresh_btn:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        symbols_to_update = list(tracking_data.keys())
                        total = len(symbols_to_update)
                        
                        for i, symbol in enumerate(symbols_to_update):
                            status_text.text(f"更新 {symbol}...")
                            current_price = get_current_price(symbol)
                            if current_price:
                                update_tracking_record(symbol, tracking_data, current_price)
                            progress_bar.progress((i + 1) / total)
                        
                        save_tracking_data(tracking_data)
                        status_text.text("✅ 价格更新完成!")
                        st.rerun()
                    
                    # 显示统计
                    stats = calculate_tracking_stats(tracking_data)
                    
                    # 计算弹簧蓄势数量
                    spring_tracking_count = sum(1 for r in tracking_data.values() if '弹簧蓄势' in r.get('signal_type', ''))
                    
                    stat_col1, stat_col2, stat_col3, stat_col4, stat_col5, stat_col6 = st.columns(6)
                    with stat_col1:
                        st.metric("⏳ 追踪中", stats['tracking'])
                    with stat_col2:
                        st.metric("✅ 已完成", stats['completed'])
                    with stat_col3:
                        st.metric("🎯 确认Squeeze", stats['squeeze'])
                    with stat_col4:
                        st.metric("🔋 弹簧蓄势", spring_tracking_count)
                    with stat_col5:
                        st.metric("❌ 失败", stats['failed'])
                    with stat_col6:
                        st.metric("📊 胜率", f"{stats['win_rate']:.1f}%")
                    
                    # 显示追踪表格
                    if tracking_data:
                        # 构建显示DataFrame
                        display_rows = []
                        for symbol, record in tracking_data.items():
                            status = record.get('status', 'tracking')
                            squeeze_confirmed = record.get('squeeze_confirmed', False)
                            is_new = record.get('is_new', False)
                            
                            # 状态图标
                            if status == 'completed':
                                status_icon = "✅ 确认" if squeeze_confirmed else "❌ 失败"
                            else:
                                status_icon = "⏳ 追踪中"
                            
                            # 新标的标注
                            symbol_display = f"🆕 {symbol}" if is_new else symbol
                            
                            # 涨幅颜色标注
                            current_return = record.get('current_return', 0)
                            max_gain = record.get('max_gain', 0)
                            max_dd = record.get('max_drawdown', 0)
                            
                            # 获取当前价格（最新的daily_prices）
                            daily_prices = record.get('daily_prices', {})
                            if daily_prices:
                                latest_date = max(daily_prices.keys())
                                current_price = daily_prices[latest_date]
                            else:
                                current_price = record.get('entry_price', 0)
                            
                            # Squeeze判断：当前涨幅>=5%就确认
                            squeeze_confirmed = current_return >= SQUEEZE_THRESHOLD
                            
                            display_rows.append({
                                '标的': symbol_display,
                                '信号日期': record.get('signal_date', ''),
                                'D0价格': record.get('entry_price', 0),
                                '当前价格': current_price,
                                '当前涨幅%': current_return,
                                '最大涨幅%': max_gain,
                                '最大回撤%': max_dd,
                                '信号类型': record.get('signal_type', '')[:15],
                                '波动环境': record.get('vol_regime', ''),
                                '到期日': record.get('top_gamma_exp', ''),
                                'Squeeze': "✅" if squeeze_confirmed else ("❌" if status == 'completed' else "⏳"),
                                '状态': status_icon
                            })
                        
                        display_df = pd.DataFrame(display_rows)
                        
                        # 按Squeeze确认优先，然后按当前涨幅排序
                        display_df['sort_key'] = display_df['Squeeze'].apply(lambda x: 0 if x == '✅' else (1 if x == '⏳' else 2))
                        display_df = display_df.sort_values(['sort_key', '当前涨幅%'], ascending=[True, False])
                        display_df = display_df.drop('sort_key', axis=1)
                        
                        # 样式化显示
                        def color_returns(val):
                            if isinstance(val, (int, float)):
                                if val >= SQUEEZE_THRESHOLD:
                                    return 'background-color: #90EE90'  # 浅绿
                                elif val >= 0:
                                    return 'background-color: #FFFACD'  # 浅黄
                                else:
                                    return 'background-color: #FFB6C1'  # 浅红
                            return ''
                        
                        styled_df = display_df.style.applymap(
                            color_returns, 
                            subset=['当前涨幅%', '最大涨幅%']
                        ).format({
                            'D0价格': '${:.2f}',
                            '当前价格': '${:.2f}',
                            '当前涨幅%': '{:+.2f}%',
                            '最大涨幅%': '{:+.2f}%',
                            '最大回撤%': '{:+.2f}%'
                        })
                        
                        st.dataframe(styled_df, use_container_width=True, hide_index=True)
                        
                        # 详细视图（可展开）
                        with st.expander("📋 详细追踪记录"):
                            for symbol, record in tracking_data.items():
                                is_new = record.get('is_new', False)
                                new_badge = "🆕 " if is_new else ""
                                
                                # 获取当前价格
                                daily_prices = record.get('daily_prices', {})
                                if daily_prices:
                                    latest_date = max(daily_prices.keys())
                                    current_price = daily_prices[latest_date]
                                else:
                                    current_price = record.get('entry_price', 0)
                                
                                current_return = record.get('current_return', 0)
                                squeeze_status = '✅ 是' if current_return >= SQUEEZE_THRESHOLD else '❌ 否'
                                
                                st.markdown(f"""
                                ---
                                **{new_badge}{symbol}** | {record.get('signal_type', '')} | {record.get('vol_regime', '')}
                                - 信号日期: {record.get('signal_date', '')} | D0价格: ${record.get('entry_price', 0):.2f} | 当前价格: ${current_price:.2f}
                                - 当前涨幅: {current_return:+.2f}% | 最大涨幅: {record.get('max_gain', 0):+.2f}% | 最大回撤: {record.get('max_drawdown', 0):+.2f}%
                                - DR: {record.get('delta_ratio', 0):.2f} | GR: {record.get('gamma_ratio', 0):.2f} | VR: {record.get('volume_ratio', 0):.2f}
                                - PW: {record.get('put_wall', 0)} | CW: {record.get('call_wall', 0)} | HW: {record.get('hedge_wall', 0)}
                                - 到期日: {record.get('top_gamma_exp', '')} | 追踪结束: {record.get('track_end_date', '')}
                                - Squeeze确认(≥5%): {squeeze_status} | 状态: {record.get('status', 'tracking')}
                                """)
                                
                                # 显示每日价格
                                if daily_prices:
                                    price_str = " → ".join([f"{d}: ${p:.2f}" for d, p in sorted(daily_prices.items())])
                                    st.caption(f"价格记录: {price_str}")
                    else:
                        st.info("暂无追踪记录。上传SpotGamma CSV后，符合条件的标的会自动添加到追踪列表。")
                    
                    # ===== 信号方向验证面板 =====
                    st.subheader("🎯 信号方向验证面板")
                    st.caption("验证信号方向是否正确：多头信号涨了=正确，空头信号跌了=正确 | 追踪至到期日+5个交易日")
                    
                    if tracking_data:
                        # 计算信号正确率统计
                        signal_stats = calculate_signal_accuracy_stats(tracking_data)
                        
                        # 计算弹簧蓄势的统计
                        spring_stats = {'total': 0, 'correct': 0}
                        for symbol, record in tracking_data.items():
                            if '弹簧蓄势' in record.get('signal_type', ''):
                                spring_stats['total'] += 1
                                # 计算当前涨跌幅
                                entry_price = record.get('entry_price', 0)
                                daily_prices = record.get('daily_prices', {})
                                if daily_prices and entry_price > 0:
                                    latest_date = max(daily_prices.keys())
                                    current_price = daily_prices[latest_date]
                                    current_return = ((current_price - entry_price) / entry_price) * 100
                                    if current_return > 0:  # 弹簧蓄势是做多信号，涨了就正确
                                        spring_stats['correct'] += 1
                        
                        # 显示统计
                        st.markdown("#### 📊 信号正确率统计")
                        
                        sig_col1, sig_col2, sig_col3, sig_col4, sig_col5 = st.columns(5)
                        with sig_col1:
                            bullish_total = signal_stats['bullish']['total']
                            bullish_correct = signal_stats['bullish']['correct']
                            st.metric(
                                f"🟢 多头信号 ({bullish_total}个)", 
                                f"{signal_stats['bullish']['accuracy']:.1f}%",
                                f"{bullish_correct}/{bullish_total} 正确"
                            )
                        with sig_col2:
                            bearish_total = signal_stats['bearish']['total']
                            bearish_correct = signal_stats['bearish']['correct']
                            st.metric(
                                f"🔴 空头信号 ({bearish_total}个)", 
                                f"{signal_stats['bearish']['accuracy']:.1f}%",
                                f"{bearish_correct}/{bearish_total} 正确"
                            )
                        with sig_col3:
                            spring_total = spring_stats['total']
                            spring_correct = spring_stats['correct']
                            spring_accuracy = (spring_correct / spring_total * 100) if spring_total > 0 else 0
                            st.metric(
                                f"🔋 弹簧蓄势 ({spring_total}个)", 
                                f"{spring_accuracy:.1f}%",
                                f"{spring_correct}/{spring_total} 正确"
                            )
                        with sig_col4:
                            st.metric(
                                "⚪ 中性信号", 
                                f"{signal_stats['neutral']['total']}个",
                                "不计入正确率"
                            )
                        with sig_col5:
                            overall_total = signal_stats['overall']['total']
                            overall_correct = signal_stats['overall']['correct']
                            st.metric(
                                f"📈 整体正确率 ({overall_total}个)", 
                                f"{signal_stats['overall']['accuracy']:.1f}%",
                                f"{overall_correct}/{overall_total} 正确"
                            )
                        
                        # 构建验证表格
                        verify_rows = []
                        for symbol, record in tracking_data.items():
                            direction = record.get('signal_direction', '')
                            signal_type_text = record.get('signal_type', '')
                            is_new = record.get('is_new', False)
                            
                            # 如果没有signal_direction字段（旧记录），从signal_type文本推断
                            # 【更新】添加新的信号关键词
                            if not direction or direction == 'neutral':
                                # 做多信号关键词
                                bullish_keywords = [
                                    '做多', '反弹', '偏多', 'bullish', 
                                    'Squeeze Up', '突破CW', 'PW支撑区', '弹簧蓄势', '正Gamma轧空'
                                ]
                                # 做空信号关键词
                                bearish_keywords = [
                                    '做空', '压力', '偏空', '破位', 'bearish',
                                    'Squeeze Down', '跌破PW', 'CW阻力区', '负Gamma螺旋'
                                ]
                                
                                if any(x in signal_type_text for x in bullish_keywords):
                                    direction = 'bullish'
                                elif any(x in signal_type_text for x in bearish_keywords):
                                    direction = 'bearish'
                                else:
                                    direction = 'neutral'
                            
                            # 获取当前价格和涨跌幅
                            daily_prices = record.get('daily_prices', {})
                            entry_price = record.get('entry_price', 0)
                            if daily_prices:
                                latest_date = max(daily_prices.keys())
                                current_price = daily_prices[latest_date]
                            else:
                                current_price = entry_price
                            
                            # 实时计算涨跌幅
                            if entry_price > 0 and current_price > 0:
                                current_return = ((current_price - entry_price) / entry_price) * 100
                            else:
                                current_return = 0
                            
                            # 实时判断方向正确性（不需要等到追踪到期）
                            if direction == 'bullish':
                                direction_correct = current_return > 0
                            elif direction == 'bearish':
                                direction_correct = current_return < 0
                            else:
                                direction_correct = None
                            
                            # 方向图标
                            if direction == 'bullish':
                                dir_icon = "🟢 多头"
                            elif direction == 'bearish':
                                dir_icon = "🔴 空头"
                            else:
                                dir_icon = "⚪ 中性"
                            
                            # 正确性图标
                            if direction_correct is True:
                                correct_icon = "✅ 正确"
                            elif direction_correct is False:
                                correct_icon = "❌ 错误"
                            else:
                                correct_icon = "⚪ 不判定"
                            
                            # 新标的标注
                            symbol_display = f"🆕 {symbol}" if is_new else symbol
                            
                            verify_rows.append({
                                '标的': symbol_display,
                                '信号方向': dir_icon,
                                '信号类型': record.get('signal_type', '')[:20],
                                'D0价格': entry_price,
                                '当前价格': current_price,
                                '涨跌幅%': current_return,
                                '方向正确': correct_icon,
                                '到期日': record.get('top_gamma_exp', ''),
                                '追踪结束': record.get('track_end_date', ''),
                                '状态': record.get('status', 'tracking')
                            })
                        
                        verify_df = pd.DataFrame(verify_rows)
                        
                        # 按正确性排序：正确 > 待定 > 错误
                        verify_df['sort_key'] = verify_df['方向正确'].apply(
                            lambda x: 0 if '正确' in x else (1 if '待定' in x else 2)
                        )
                        verify_df = verify_df.sort_values(['sort_key', '涨跌幅%'], ascending=[True, False])
                        verify_df = verify_df.drop('sort_key', axis=1)
                        
                        # 样式化
                        def color_direction(val):
                            if '正确' in str(val):
                                return 'background-color: #90EE90'
                            elif '错误' in str(val):
                                return 'background-color: #FFB6C1'
                            return ''
                        
                        def color_return_direction(val):
                            if isinstance(val, (int, float)):
                                if val > 0:
                                    return 'color: green'
                                elif val < 0:
                                    return 'color: red'
                            return ''
                        
                        styled_verify = verify_df.style.applymap(
                            color_direction, subset=['方向正确']
                        ).applymap(
                            color_return_direction, subset=['涨跌幅%']
                        ).format({
                            'D0价格': '${:.2f}',
                            '当前价格': '${:.2f}',
                            '涨跌幅%': '{:+.2f}%'
                        })
                        
                        st.dataframe(styled_verify, use_container_width=True, hide_index=True)
                        
                        # 按信号类型分组统计
                        with st.expander("📋 按信号类型分组统计"):
                            signal_type_stats = {}
                            for symbol, record in tracking_data.items():
                                sig_type = record.get('signal_type', '未知')[:20]
                                direction = record.get('signal_direction', '')
                                
                                # 如果没有signal_direction字段（旧记录），从signal_type文本推断
                                # 【更新】添加新的信号关键词
                                if not direction or direction == 'neutral':
                                    # 做多信号关键词
                                    bullish_keywords = [
                                        '做多', '反弹', '偏多', 'bullish', 
                                        'Squeeze Up', '突破CW', 'PW支撑区', '弹簧蓄势', '正Gamma轧空'
                                    ]
                                    # 做空信号关键词
                                    bearish_keywords = [
                                        '做空', '压力', '偏空', '破位', 'bearish',
                                        'Squeeze Down', '跌破PW', 'CW阻力区', '负Gamma螺旋'
                                    ]
                                    
                                    if any(x in sig_type for x in bullish_keywords):
                                        direction = 'bullish'
                                    elif any(x in sig_type for x in bearish_keywords):
                                        direction = 'bearish'
                                    else:
                                        direction = 'neutral'
                                
                                # 实时计算涨跌幅
                                entry_price = record.get('entry_price', 0)
                                daily_prices = record.get('daily_prices', {})
                                if daily_prices and entry_price > 0:
                                    latest_date = max(daily_prices.keys())
                                    current_price = daily_prices[latest_date]
                                    current_return = ((current_price - entry_price) / entry_price) * 100
                                else:
                                    current_return = 0
                                
                                # 实时判断方向正确性
                                if direction == 'bullish':
                                    direction_correct = current_return > 0
                                elif direction == 'bearish':
                                    direction_correct = current_return < 0
                                else:
                                    direction_correct = None
                                
                                if sig_type not in signal_type_stats:
                                    signal_type_stats[sig_type] = {'total': 0, 'correct': 0, 'wrong': 0, 'neutral': 0}
                                
                                signal_type_stats[sig_type]['total'] += 1
                                if direction_correct is True:
                                    signal_type_stats[sig_type]['correct'] += 1
                                elif direction_correct is False:
                                    signal_type_stats[sig_type]['wrong'] += 1
                                else:
                                    signal_type_stats[sig_type]['neutral'] += 1
                            
                            type_rows = []
                            for sig_type, data in signal_type_stats.items():
                                judged = data['correct'] + data['wrong']
                                accuracy = (data['correct'] / judged * 100) if judged > 0 else 0
                                type_rows.append({
                                    '信号类型': sig_type,
                                    '总数': data['total'],
                                    '正确': data['correct'],
                                    '错误': data['wrong'],
                                    '中性(不判定)': data['neutral'],
                                    '正确率': f"{accuracy:.1f}%"
                                })
                            
                            type_df = pd.DataFrame(type_rows)
                            type_df = type_df.sort_values('总数', ascending=False)
                            st.dataframe(type_df, use_container_width=True, hide_index=True)
                        
                        # ===== 已完成追踪历史 =====
                        st.subheader("📚 已完成追踪历史")
                        st.caption("追踪周期结束的标的，用于分析历史准确率")
                        
                        # 分离追踪中和已完成的记录
                        completed_records = []
                        for symbol, record in tracking_data.items():
                            status = record.get('status', 'tracking')
                            track_end = record.get('track_end_date', '')
                            
                            # 检查是否已过追踪结束日期
                            is_completed = status == 'completed'
                            if track_end:
                                try:
                                    end_date = datetime.strptime(track_end, '%Y-%m-%d')
                                    if datetime.now() > end_date:
                                        is_completed = True
                                except:
                                    pass
                            
                            if is_completed:
                                # 计算最终收益
                                entry_price = record.get('entry_price', 0)
                                daily_prices = record.get('daily_prices', {})
                                if daily_prices and entry_price > 0:
                                    latest_date = max(daily_prices.keys())
                                    final_price = daily_prices[latest_date]
                                    final_return = ((final_price - entry_price) / entry_price) * 100
                                else:
                                    final_return = 0
                                
                                # 判断最终结果
                                direction = record.get('signal_direction', 'neutral')
                                if direction == 'bullish':
                                    final_correct = final_return > 0
                                elif direction == 'bearish':
                                    final_correct = final_return < 0
                                else:
                                    final_correct = None
                                
                                completed_records.append({
                                    '标的': symbol,
                                    '信号类型': record.get('signal_type', '')[:25],
                                    '信号方向': '🟢多' if direction == 'bullish' else ('🔴空' if direction == 'bearish' else '⚪中'),
                                    '入场价': entry_price,
                                    '最终收益': final_return,
                                    '结果': '✅正确' if final_correct == True else ('❌错误' if final_correct == False else '⚪不判定'),
                                    '到期日': record.get('top_gamma_exp', ''),
                                    '追踪结束': track_end,
                                    'CW上移': '✓' if record.get('cw_increase') else ''
                                })
                        
                        if completed_records:
                            completed_df = pd.DataFrame(completed_records)
                            
                            # 统计历史准确率
                            total_judged = len([r for r in completed_records if r['结果'] != '⚪不判定'])
                            total_correct = len([r for r in completed_records if r['结果'] == '✅正确'])
                            history_accuracy = (total_correct / total_judged * 100) if total_judged > 0 else 0
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("历史准确率", f"{history_accuracy:.1f}%", f"{total_correct}/{total_judged}")
                            with col2:
                                st.metric("已完成追踪", len(completed_records))
                            with col3:
                                cw_correct = len([r for r in completed_records if r['CW上移'] == '✓' and r['结果'] == '✅正确'])
                                cw_total = len([r for r in completed_records if r['CW上移'] == '✓' and r['结果'] != '⚪不判定'])
                                cw_acc = (cw_correct / cw_total * 100) if cw_total > 0 else 0
                                st.metric("CW上移准确率", f"{cw_acc:.1f}%", f"{cw_correct}/{cw_total}")
                            
                            st.dataframe(
                                completed_df.style.format({'入场价': '${:.2f}', '最终收益': '{:+.2f}%'}),
                                use_container_width=True, 
                                hide_index=True
                            )
                        else:
                            st.info("暂无已完成的追踪记录")
                    else:
                        st.info("暂无追踪记录")
                        
            except Exception as e:
                st.error(f"读取文件失败: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # ========== 侧边栏 ==========
    with st.sidebar:
        st.header("📖 使用说明")
        st.markdown("""
        **筛选流程:**
        1. **板块资金流** → 扫描ETF，获取板块状态
        2. **个股筛选** → 输入股票池，技术筛选
        3. **综合名单** → 查看多空分类 + 顺逆风
        4. **SpotGamma** → 上传CSV交叉验证
        
        ---
        
        **技术信号说明:**
        - 🟢 多头回调买点
        - 🔵 超卖 / 反转
        - 🔴 空头反弹做空
        - 🔥 Squeeze向上突破
        - 💥 Squeeze向下突破
        - ⏳ Squeeze蓄势
        
        ---
        
        **顺风/逆风:**
        - 🌬️ 顺风 = 信号方向与板块资金流一致
        - 🌪️ 逆风 = 信号方向与板块资金流相反
        """)
        
        with st.expander("📊 SpotGamma 官方定义"):
            st.markdown("""
            **关键行权价:**
            - **Call Wall**: 最大Call Gamma行权价，市场"天花板"阻力
            - **Put Wall**: 最大Put Gamma行权价，市场"地板"支撑
            - **Hedge Wall**: MM风险暴露变化位，价格>HW=均值回归，<HW=趋势
            - **Key Gamma Strike**: 最大总Gamma行权价，磁吸效应中心
            
            ---
            
            **比率指标:**
            - **Delta Ratio** = Put Delta ÷ Call Delta（方向性敞口）
            - **Gamma Ratio** = Put Gamma ÷ Call Gamma（加速效应）
            - **Volume Ratio** = ATM Put/Call Delta成交量比（反弹潜力）
            - **P/C OI Ratio** = Put/Call持仓量比（情绪参考）
            
            ---
            
            **到期风险:**
            - **Next Exp Gamma**: >25%集中（官方警戒线），到期前后剧烈波动
            - **Options Impact**: 期权对股价的驱动程度，>50%=期权主导
            """)
        
        with st.expander("🎯 交易信号矩阵"):
            st.markdown("""
            **位置×结构矩阵:**
            
            | 位置 | Call主导 | Put主导 |
            |------|----------|---------|
            | 近CW | 🟢突破做多 | 🔴压力做空 |
            | 近PW | 🟢反弹做多 | 🔴破位做空 |
            | 中间 | 观察 | 观察 |
            
            ---
            
            **期权结构判断:**
            - **Call主导**: DR > -1 且 GR < 1
            - **Put主导**: DR < -3 且 GR > 2
            
            ---
            
            **MM对冲机制:**
            - CW是天花板，MM卖Call→突破后被迫买股→squeeze↑
            - PW是地板，MM卖Put→跌破后被迫卖股→squeeze↓
            
            ---
            
            **波动环境修正:**
            - 价格 > Hedge Wall → 均值回归，突破难度大
            - 价格 < Hedge Wall → 趋势环境，顺势信号更可靠
            """)
        
        with st.expander("⚡ 特殊信号说明"):
            st.markdown("""
            **💀 Gamma陷阱（最高优先级警告）:**
            - 已跌破Put Wall + Next Exp Gamma > 25%
            - MM正在连环抛售，**千万不要抄底！**
            
            ---
            
            **⚡ 到期反弹潜力（4条件 + 环境修正）:**
            1. Volume Ratio > 1.2（ATM Put活跃）
            2. Delta Ratio < -3（Put Delta占优）
            3. Next Exp Gamma > 25%（临近到期）
            4. 价格高于Put Wall 2%以上
            
            **环境修正（基于Hedge Wall）:**
            - 正Gamma区（价格>HW）：均值回归，反弹更稳健
            - 负Gamma区（价格<HW）：高波动，反弹剧烈但风险更高
            
            **逻辑链条:**
            ```
            MM Short Put → 正Delta
                ↓
            卖股票对冲（持有空头）
                ↓
            到期Put无价值(OTM)
                ↓
            买回股票平仓 → 反弹
            ```
            
            ---
            
            **MM对冲速查:**
            | MM持仓 | Delta | 对冲 | 到期平仓 |
            |--------|-------|------|---------|
            | Short Call | 负 | 买股 | 卖股↓ |
            | Short Put | 正 | 卖股 | 买股↑ |
            
            ---
            
            **其他信号:**
            - 🔴 Gamma极度集中: >50%待释放
            - 🟠 Gamma集中警告: >25%待释放
            - ⚠️ 空头挤压: DR<-5 + 低VR + 近PW
            - ⚠️ 多头踩踏: DR>-1 + 高VR + 近CW
            - ⚠️ 深度负Gamma区: 价格远低于HW
            """)


if __name__ == "__main__":
    main()
