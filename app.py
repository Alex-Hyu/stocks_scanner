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

def load_tracking_data():
    """加载追踪数据"""
    if os.path.exists(TRACKING_FILE):
        try:
            with open(TRACKING_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_tracking_data(data):
    """保存追踪数据"""
    with open(TRACKING_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

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
            # 到期日+5个交易日（约7个自然日）
            track_end = (exp_date + timedelta(days=7)).strftime('%Y-%m-%d')
        else:
            # 默认10天后结束追踪
            track_end = (datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d')
    except:
        track_end = (datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d')
    
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
        'signal_direction': signal_direction,  # 新增：信号方向
        'vol_regime': row.get('Vol_Regime', '未知'),
        'delta_ratio': float(row.get('Delta Ratio', 0)),
        'gamma_ratio': float(row.get('Gamma Ratio', 0)),
        'volume_ratio': float(row.get('Volume Ratio', 0)) if pd.notna(row.get('Volume Ratio')) else 0,
        'next_exp_gamma': float(row.get('Next Exp Gamma', 0)) if pd.notna(row.get('Next Exp Gamma')) else 0,
        'options_impact': float(row.get('Options Impact', 0)) if pd.notna(row.get('Options Impact')) else 0,
        'put_wall': float(row.get('Put Wall', 0)),
        'call_wall': float(row.get('Call Wall', 0)),
        'hedge_wall': float(row.get('Hedge Wall', 0)) if pd.notna(row.get('Hedge Wall')) else 0,
        'daily_prices': {today_str: float(row['Current Price'])},
        'current_return': 0,
        'max_gain': 0,
        'max_drawdown': 0,
        'squeeze_confirmed': False,
        'direction_correct': None,  # 新增：方向是否正确
        'status': 'tracking',
        'is_new': True  # 标记为新增
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
        if not direction or direction == 'neutral':
            if any(x in signal_type_text for x in ['做多', '反弹', '偏多', 'bullish']):
                direction = 'bullish'
            elif any(x in signal_type_text for x in ['做空', '压力', '偏空', '破位', 'bearish']):
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
# Streamlit 界面
# ============================================================

def main():
    st.title("🎯 股票波段期权筛选系统")
    st.caption(f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 板块资金流", "🔍 个股筛选", "🎯 综合名单", "📋 SpotGamma验证"])
    
    # ========== Tab 1: 板块资金流 ==========
    with tab1:
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
    with tab2:
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
    with tab3:
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
    
    # ========== Tab 4: SpotGamma验证 ==========
    with tab4:
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
        
        uploaded_file = st.file_uploader("上传SpotGamma CSV文件", type=['csv'])
        
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
                        判断期权结构（放宽条件，使用OR逻辑）
                        - Delta Ratio = Put Delta ÷ Call Delta（方向性敞口）
                        - Gamma Ratio = Put Gamma ÷ Call Gamma（加速效应）
                        
                        【修正逻辑】
                        - Put主导: DR < -2 OR GR > 1.5（满足其一即可）
                        - Call主导: DR > -1 OR GR < 1（满足其一即可）
                        - 中性: 两者都在中间范围
                        """
                        dr = row['Delta Ratio']
                        gr = row['Gamma Ratio']
                        if pd.isna(dr) or pd.isna(gr):
                            return "数据缺失", "unknown"
                        
                        # Put主导判断（放宽，OR逻辑）
                        is_put_by_dr = dr < -2
                        is_put_by_gr = gr > 1.5
                        
                        # Call主导判断（放宽，OR逻辑）
                        is_call_by_dr = dr > -1
                        is_call_by_gr = gr < 1
                        
                        # 优先判断明确的主导方向
                        if dr < -3 or gr > 2:
                            # 强Put主导
                            return "Put主导", "put_dominant"
                        elif dr > -0.5 or gr < 0.8:
                            # 强Call主导
                            return "Call主导", "call_dominant"
                        elif is_put_by_dr or is_put_by_gr:
                            # 弱Put主导
                            return "Put偏多", "put_leaning"
                        elif is_call_by_dr or is_call_by_gr:
                            # 弱Call主导
                            return "Call偏多", "call_leaning"
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
                    
                    def get_trade_signal(position, structure, vol_regime, options_impact, high_oi_thresh, dist_pw, dist_cw, next_gamma):
                        """
                        生成交易信号 - 基于做市商盈利逻辑
                        
                        【核心原理】
                        - 做市商卖期权收权利金，希望期权到期作废
                        - Put主导/Put偏多 = MM大量Short Put → MM希望价格涨（杀Put）→ Squeeze Up潜力
                        - Call主导/Call偏多 = MM大量Short Call → MM希望价格跌（杀Call）→ Squeeze Down风险
                        - Options Impact高 + Next Exp Gamma集中 = Squeeze力度大
                        """
                        # 判断Squeeze条件
                        has_squeeze_potential = (options_impact >= 20 and 
                                                next_gamma is not None and not pd.isna(next_gamma) and next_gamma >= 0.25)
                        
                        if options_impact > high_oi_thresh:
                            confidence = "⭐⭐⭐"
                        elif options_impact > high_oi_thresh * 0.6:
                            confidence = "⭐⭐"
                        else:
                            confidence = "⭐"
                        
                        # 归类期权结构
                        is_put_side = structure in ["Put主导", "Put偏多"]
                        is_call_side = structure in ["Call主导", "Call偏多"]
                        is_strong_put = structure == "Put主导"
                        is_strong_call = structure == "Call主导"
                        
                        # ===== 新的信号矩阵（基于做市商盈利逻辑）=====
                        
                        # 1. Put主导/偏多情况：MM希望价格涨（杀Put）
                        if is_put_side:
                            if dist_pw > 10 and dist_cw < 20:
                                # Put侧 + 远离PW + 接近CW = Squeeze Up潜力
                                if has_squeeze_potential:
                                    strength = "⭐⭐⭐" if is_strong_put else "⭐⭐"
                                    return (f"🟢 Squeeze Up潜力 {strength}", 
                                           f"{structure}+OI={options_impact:.0f}%+NEG={next_gamma*100:.0f}%，MM有动机推涨杀Put", 
                                           "bullish")
                                else:
                                    return (f"🟢 偏多观察 {confidence}", 
                                           f"{structure}+远离PW，但Squeeze条件不完整", 
                                           "bullish_watch")
                            elif dist_cw < 0:
                                # 已突破CW = Squeeze Up进行中
                                return (f"🔥 Squeeze Up进行中 {confidence}", 
                                       f"{structure}+突破CW，MM被迫买股对冲，加速上涨", 
                                       "bullish")
                            elif dist_cw < 5:
                                # 接近CW = 突破前
                                if has_squeeze_potential:
                                    return (f"🟢 接近突破 {confidence}", 
                                           f"{structure}+近CW，突破后可能Squeeze Up", 
                                           "bullish")
                                else:
                                    return (f"⏳ 观察突破 {confidence}", f"{structure}+近CW，等待突破确认", "neutral")
                            elif dist_pw < 5:
                                # Put侧 + 近PW = 风险
                                if has_squeeze_potential:
                                    return (f"⚠️ 方向待定 {confidence}", 
                                           f"{structure}+近PW，需观察是否守住支撑", 
                                           "neutral")
                                else:
                                    return ("⚠️ 支撑观察", f"{structure}+近PW，观察是否守住", "neutral")
                            else:
                                # 中间区域
                                if has_squeeze_potential:
                                    return (f"🟢 Squeeze Up蓄势 {confidence}", 
                                           f"{structure}+OI高+NEG高，等待接近CW触发", 
                                           "bullish_watch")
                                else:
                                    return (f"⏳ 偏多蓄势 {confidence}", f"{structure}但条件不完整，等待时机", "bullish_watch")
                        
                        # 2. Call主导/偏多情况：MM希望价格跌（杀Call）
                        elif is_call_side:
                            if dist_cw > 10 and dist_pw < 20:
                                # Call侧 + 远离CW + 接近PW = Squeeze Down风险
                                if has_squeeze_potential:
                                    strength = "⭐⭐⭐" if is_strong_call else "⭐⭐"
                                    return (f"🔴 Squeeze Down风险 {strength}", 
                                           f"{structure}+OI={options_impact:.0f}%+NEG={next_gamma*100:.0f}%，MM有动机压价杀Call", 
                                           "bearish")
                                else:
                                    return (f"🔴 偏空观察 {confidence}", 
                                           f"{structure}+远离CW，但Squeeze条件不完整", 
                                           "bearish_watch")
                            elif dist_pw < 0:
                                # 已跌破PW = Squeeze Down进行中
                                return (f"💥 Squeeze Down进行中 {confidence}", 
                                       f"{structure}+跌破PW，MM被迫卖股对冲，加速下跌", 
                                       "bearish")
                            elif dist_pw < 5:
                                # 接近PW = 破位前
                                if has_squeeze_potential:
                                    return (f"🔴 接近破位 {confidence}", 
                                           f"{structure}+近PW，跌破后可能Squeeze Down", 
                                           "bearish")
                                else:
                                    return (f"⏳ 观察破位 {confidence}", f"{structure}+近PW，等待破位确认", "neutral")
                            elif dist_cw < 5:
                                # Call侧 + 近CW = 可能受阻
                                return (f"⚠️ 阻力观察 {confidence}", f"{structure}+近CW，MM有动机压价", "bearish_watch")
                            else:
                                # 中间区域
                                if has_squeeze_potential:
                                    return (f"🔴 Squeeze Down蓄势 {confidence}", 
                                           f"{structure}+OI高+NEG高，等待接近PW触发", 
                                           "bearish_watch")
                                else:
                                    return (f"⏳ 偏空蓄势 {confidence}", f"{structure}但条件不完整，等待时机", "bearish_watch")
                        
                        # 3. 中性情况
                        else:
                            if position == "近Call Wall":
                                return ("⚪ CW观望", "结构中性+近CW，等待方向确认", "neutral")
                            elif position == "近Put Wall":
                                return ("⚪ PW观望", "结构中性+近PW，等待方向确认", "neutral")
                            else:
                                return ("⚪ 中性", "结构中性+位置中性，无明确方向", "neutral")
                        
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
                            row.get('Next Exp Gamma', None)  # Next Exp Gamma
                        ), axis=1)
                    sg_df['Trade_Signal'] = signal_results.apply(lambda x: x[0])
                    sg_df['Signal_Logic'] = signal_results.apply(lambda x: x[1])
                    sg_df['Signal_Type'] = signal_results.apply(lambda x: x[2])
                    
                    # 特殊信号检测
                    sg_df['Special_Signals'] = sg_df.apply(
                        lambda row: detect_special_signals(row, row['Dist_PW_Calc'], row['Dist_CW_Calc']), axis=1)
                    
                    # 过滤低OI标的
                    sg_filtered = sg_df[sg_df['Options Impact'] >= min_options_impact].copy()
                    
                    # ===== 显示统计 =====
                    st.subheader("📊 分析概览")
                    
                    # 统计各类信号
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    bullish_count = len(sg_filtered[sg_filtered['Signal_Type'] == 'bullish'])
                    bearish_count = len(sg_filtered[sg_filtered['Signal_Type'] == 'bearish'])
                    watch_bull = len(sg_filtered[sg_filtered['Signal_Type'] == 'bullish_watch'])
                    watch_bear = len(sg_filtered[sg_filtered['Signal_Type'] == 'bearish_watch'])
                    
                    # 统计波动环境
                    mean_rev_count = len(sg_filtered[sg_filtered['Vol_Regime_Type'] == 'mean_reversion'])
                    trending_count = len(sg_filtered[sg_filtered['Vol_Regime_Type'] == 'trending'])
                    
                    with col1:
                        st.metric("🟢 高确信做多", bullish_count)
                    with col2:
                        st.metric("🔴 高确信做空", bearish_count)
                    with col3:
                        st.metric("🟢 偏多观察", watch_bull)
                    with col4:
                        st.metric("🔴 偏空观察", watch_bear)
                    with col5:
                        st.metric("📈 趋势环境", trending_count, help="价格<Hedge Wall，高波动")
                    
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
                    st.caption("核心逻辑: Put主导+高OI+高NEG+远离PW = MM有动机推涨杀Put → Squeeze Up潜力")
                    
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
                    st.caption("核心逻辑: Call主导+高OI+高NEG+远离CW = MM有动机压价杀Call → Squeeze Down风险")
                    
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
                    with st.expander("📋 查看完整分析表"):
                        full_cols = ['Symbol', 'Current Price', 'Trade_Signal', 'Price_Position', 
                                    'Option_Structure', 'Vol_Regime', 'Gamma_Magnet', 'Delta Ratio', 'Gamma Ratio',
                                    'Put Wall', 'Call Wall', 'Hedge Wall', 'Dist_to_PW_%', 'Dist_to_CW_%', 
                                    'Options Impact', 'Volume Ratio', 'Next Exp Gamma']
                        available_cols = [c for c in full_cols if c in sg_filtered.columns]
                        df_sorted = sg_filtered.sort_values('Options Impact', ascending=False)
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
                    st.caption(f"追踪文件: {os.path.abspath(TRACKING_FILE)} | Squeeze标准: ≥{SQUEEZE_THRESHOLD}%涨幅")
                    
                    # 加载追踪数据
                    tracking_data = load_tracking_data()
                    today_str = datetime.now().strftime('%Y-%m-%d')
                    
                    # 识别新标的并添加到追踪
                    new_symbols = []
                    for _, row in sg_filtered.iterrows():
                        symbol = row['Symbol']
                        signal_type = row.get('Trade_Signal', '未知信号')
                        
                        if symbol not in tracking_data:
                            # 新标的
                            tracking_data[symbol] = add_new_tracking(symbol, row, signal_type, today_str)
                            new_symbols.append(symbol)
                        else:
                            # 已存在的标的，检查是否需要更新信号（如果信号变化）
                            tracking_data[symbol]['is_new'] = False
                    
                    # 保存更新
                    if new_symbols:
                        save_tracking_data(tracking_data)
                        st.success(f"🆕 新增追踪: {', '.join(new_symbols)}")
                    
                    # 刷新价格按钮
                    col1, col2, col3 = st.columns([1, 1, 2])
                    with col1:
                        refresh_btn = st.button("🔄 刷新价格", type="primary")
                    with col2:
                        clear_completed = st.button("🗑️ 清除已完成")
                    with col3:
                        if st.button("🗑️ 清空所有追踪记录"):
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
                    
                    stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
                    with stat_col1:
                        st.metric("⏳ 追踪中", stats['tracking'])
                    with stat_col2:
                        st.metric("✅ 已完成", stats['completed'])
                    with stat_col3:
                        st.metric("🎯 确认Squeeze", stats['squeeze'])
                    with stat_col4:
                        st.metric("❌ 失败", stats['failed'])
                    with stat_col5:
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
                        
                        # 显示统计
                        st.markdown("#### 📊 信号正确率统计")
                        
                        sig_col1, sig_col2, sig_col3, sig_col4 = st.columns(4)
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
                            st.metric(
                                "⚪ 中性信号", 
                                f"{signal_stats['neutral']['total']}个",
                                "不计入正确率"
                            )
                        with sig_col4:
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
                            if not direction or direction == 'neutral':
                                if any(x in signal_type_text for x in ['做多', '反弹', '偏多', 'bullish']):
                                    direction = 'bullish'
                                elif any(x in signal_type_text for x in ['做空', '压力', '偏空', '破位', 'bearish']):
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
                                if not direction or direction == 'neutral':
                                    if any(x in sig_type for x in ['做多', '反弹', '偏多', 'bullish']):
                                        direction = 'bullish'
                                    elif any(x in sig_type for x in ['做空', '压力', '偏空', '破位', 'bearish']):
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
