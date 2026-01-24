"""
SpotGamma 期权分析系统 V2.0
板块一：QQQ/NQ盘前分析
板块二：Equity Hub分析（Squeeze + CW上移）
板块三：周五到期Gamma分析

运行方式: streamlit run app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import re
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
    page_title="SpotGamma期权分析系统",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# 配置常量
# ============================================================
TRACKING_FILE = "./spotgamma_tracking.json"
SQUEEZE_THRESHOLD = 5.0  # 5%涨幅算squeeze确认

# Google Sheets配置
GSHEETS_CREDENTIALS_FILE = "./google_credentials.json"
GSHEETS_SPREADSHEET_NAME = "SpotGamma_Tracking"

# Worksheet名称
WS_QQQ_DAILY = "QQQ_Daily"
WS_EQUITY_HUB = "Equity_Hub"
WS_FRIDAY_EXPIRY = "Friday_Expiry"

# ============================================================
# Google Sheets 集成函数
# ============================================================

def get_gsheets_client():
    """获取Google Sheets客户端"""
    if not GSHEETS_AVAILABLE:
        return None
    
    try:
        if hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
            creds = Credentials.from_service_account_info(
                st.secrets['gcp_service_account'],
                scopes=[
                    'https://www.googleapis.com/auth/spreadsheets',
                    'https://www.googleapis.com/auth/drive'
                ]
            )
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

def get_or_create_worksheet(spreadsheet, ws_name):
    """获取或创建worksheet"""
    try:
        return spreadsheet.worksheet(ws_name)
    except gspread.exceptions.WorksheetNotFound:
        return spreadsheet.add_worksheet(title=ws_name, rows=1000, cols=50)

def load_worksheet_data(ws_name):
    """从指定worksheet加载数据"""
    client = get_gsheets_client()
    if not client:
        return None
    
    try:
        spreadsheet = client.open(GSHEETS_SPREADSHEET_NAME)
        worksheet = get_or_create_worksheet(spreadsheet, ws_name)
        all_values = worksheet.get_all_values()
        
        if len(all_values) <= 1:
            return {}
        
        headers = all_values[0]
        data = {}
        
        # 查找key和data_json列
        try:
            key_idx = headers.index('key')
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
        st.warning(f"加载{ws_name}失败: {e}")
        return None

def save_worksheet_data(ws_name, data):
    """保存数据到指定worksheet"""
    client = get_gsheets_client()
    if not client:
        return False
    
    try:
        spreadsheet = client.open(GSHEETS_SPREADSHEET_NAME)
        worksheet = get_or_create_worksheet(spreadsheet, ws_name)
        
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
        st.warning(f"保存{ws_name}失败: {e}")
        return False

# ============================================================
# 通用工具函数
# ============================================================

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

def parse_number(value):
    """解析数字（处理带单位的情况如1.2M, -2.1B）"""
    if pd.isna(value) or value == '':
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

# ============================================================
# CSV自动识别
# ============================================================

def identify_csv_type(df):
    """自动识别CSV类型"""
    columns = set(df.columns.str.lower())
    
    # QQQ历史数据：有Date列
    if 'date' in columns:
        return 'qqq_history'
    
    # 检查是否有Options Impact列
    has_options_impact = 'options impact' in columns
    
    # 检查是否有Call Gamma列
    has_call_gamma = 'call gamma' in columns
    
    if has_call_gamma:
        if has_options_impact:
            return 'call_wall_increase'
        else:
            return 'friday_expiry'
    else:
        if has_options_impact:
            return 'squeeze'
    
    return 'unknown'

# ============================================================
# 板块一：QQQ/NQ盘前分析
# ============================================================

def parse_qqq_premarket_text(text):
    """解析QQQ盘前粘贴数据"""
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
        if 'QQQ' in line.upper() and '盘前' in line:
            current_section = 'qqq'
            # 解析价格：QQQ盘前现价：__619.14__，昨收__620.78__
            prices = re.findall(r'[\d.]+', line)
            if len(prices) >= 2:
                result['qqq']['current'] = float(prices[0])
                result['qqq']['prev_close'] = float(prices[1])
            continue
        
        # 检测NQ部分
        if 'NQ' in line.upper() and '盘前' in line:
            current_section = 'nq'
            # 解析价格
            prices = re.findall(r'[\d.]+', line)
            if len(prices) >= 2:
                result['nq']['current'] = float(prices[0])
                result['nq']['prev_close'] = float(prices[1])
            continue
        
        # 解析关键位
        if current_section:
            # QQQ格式：630 Call Wall
            # NQ格式：25901 26020 Combo 4 (第一列NDX忽略，第二列NQ)
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
                    # 第二列是NQ值
                    price = float(parts[1])
                    level_name = ' '.join(parts[2:])
                    result['nq']['levels'][level_name] = price
                except:
                    pass
    
    return result

def get_gamma_environment(current_price, zero_gamma):
    """判断Gamma环境"""
    if current_price is None or zero_gamma is None:
        return "未知", "unknown"
    
    if current_price > zero_gamma:
        return "正Gamma", "positive"
    else:
        return "负Gamma", "negative"

def get_volatility_regime(current_price, hedge_wall):
    """判断波动环境"""
    if current_price is None or hedge_wall is None:
        return "未知", "unknown"
    
    if current_price > hedge_wall:
        return "均值回归", "mean_reversion"
    else:
        return "高波动/趋势", "trending"

def analyze_qqq_nq(premarket_data, history_data=None):
    """分析QQQ/NQ数据"""
    analysis = {
        'qqq': {},
        'nq': {},
        'cross_validation': {},
        'prediction': {}
    }
    
    # 提取关键位
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
            env, env_type = get_gamma_environment(current, zero_gamma)
            analysis[market]['gamma_env'] = env
            analysis[market]['gamma_env_type'] = env_type
        
        # 波动环境
        if current and vol_trigger:
            vol_env, vol_type = get_volatility_regime(current, vol_trigger)
            analysis[market]['vol_regime'] = vol_env
            analysis[market]['vol_regime_type'] = vol_type
    
    # 交叉验证
    qqq_env = analysis['qqq'].get('gamma_env_type')
    nq_env = analysis['nq'].get('gamma_env_type')
    
    if qqq_env and nq_env:
        if qqq_env == nq_env:
            analysis['cross_validation'] = {
                'status': '一致',
                'message': f"QQQ和NQ均处于{analysis['qqq'].get('gamma_env')}环境",
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
    
    # 日内预测
    if qqq.get('call_wall') and qqq.get('put_wall'):
        analysis['prediction'] = {
            'resistance': f"{qqq.get('call_wall', 'N/A')}",
            'support': f"{qqq.get('put_wall', 'N/A')}",
            'gamma_env': dominant_data.get('gamma_env', '未知'),
            'vol_regime': dominant_data.get('vol_regime', '未知')
        }
        
        # 收盘预测（向Zero Gamma靠拢）
        if qqq.get('zero_gamma') and qqq.get('current'):
            zg = qqq['zero_gamma']
            current = qqq['current']
            # 窄幅区间：±0.5%
            close_low = zg * 0.995
            close_high = zg * 1.005
            analysis['prediction']['close_range'] = f"{close_low:.2f} - {close_high:.2f}"
            analysis['prediction']['close_target'] = zg
    
    return analysis

def predict_next_day_open(history_row, analysis):
    """预测明日开盘方向"""
    factors = []
    bullish_score = 0
    bearish_score = 0
    
    if history_row is not None:
        # NEG强度
        neg = history_row.get('Next Exp Gamma', 0)
        if isinstance(neg, str):
            neg = parse_number(neg.replace('%', '')) 
        if neg:
            neg = neg * 100 if neg < 1 else neg  # 转换为百分比
        
        # Gamma Ratio
        gr = history_row.get('Gamma Ratio', 1)
        if isinstance(gr, str):
            gr = parse_number(gr)
        
        # DPI
        dpi = history_row.get('DPI', 0)
        if isinstance(dpi, str):
            dpi = parse_number(dpi.replace('%', ''))
        
        # 判断因素
        # 1. Gamma Ratio
        if gr and gr < 1:
            factors.append(f"GR={gr:.2f} (Call主导) → 轻微看多")
            bullish_score += 1
        elif gr and gr > 1.2:
            factors.append(f"GR={gr:.2f} (Put主导) → 轻微看空")
            bearish_score += 1
        
        # 2. DPI
        if dpi and dpi > 50:
            factors.append(f"DPI={dpi:.1f}% → 机构买入支撑")
            bullish_score += 1
        elif dpi and dpi < 45:
            factors.append(f"DPI={dpi:.1f}% → 机构买盘减弱")
            bearish_score += 1
        
        # 3. NEG强度
        neg_strength = "弱"
        if neg and neg > 25:
            neg_strength = "中等" if neg < 40 else "强"
        factors.append(f"NEG={neg:.1f}% ({neg_strength}强度)")
    
    # 位置因素
    qqq = analysis.get('qqq', {})
    if qqq.get('current') and qqq.get('call_wall'):
        if qqq['current'] > qqq['call_wall']:
            factors.append("价格>CW → 空头回补压力")
            bullish_score += 1
        elif qqq.get('put_wall') and qqq['current'] < qqq['put_wall']:
            factors.append("价格<PW → 多头止损压力")
            bearish_score += 1
    
    # 综合判断
    if bullish_score > bearish_score:
        direction = "平开或轻微高开"
        direction_type = "bullish"
    elif bearish_score > bullish_score:
        direction = "平开或轻微低开"
        direction_type = "bearish"
    else:
        direction = "方向不明，观望"
        direction_type = "neutral"
    
    return {
        'direction': direction,
        'direction_type': direction_type,
        'factors': factors,
        'bullish_score': bullish_score,
        'bearish_score': bearish_score
    }


# ============================================================
# 板块二：Equity Hub分析
# ============================================================

def get_option_structure(row):
    """判断期权结构"""
    dr = row.get('Delta Ratio')
    gr = row.get('Gamma Ratio')
    
    if pd.isna(dr) or pd.isna(gr):
        return "数据缺失", "unknown"
    
    dr = parse_number(dr) if isinstance(dr, str) else dr
    gr = parse_number(gr) if isinstance(gr, str) else gr
    
    if dr is None or gr is None:
        return "数据缺失", "unknown"
    
    # Put主导（强）: GR > 1.5 AND DR < -2
    if gr > 1.5 and dr < -2:
        return "Put主导", "put_dominant"
    
    # Put偏多: GR > 1.3
    elif gr > 1.3:
        return "Put偏多", "put_leaning"
    
    # Put轻微: GR > 1
    elif gr > 1:
        return "Put轻微", "put_slight"
    
    # Call主导（强）: GR < 0.8 AND DR > -0.5
    elif gr < 0.8 and dr > -0.5:
        return "Call主导", "call_dominant"
    
    # Call偏多: GR < 1
    elif gr < 1:
        return "Call偏多", "call_leaning"
    
    else:
        return "中性", "neutral"

def get_position_zone(row, threshold=5):
    """
    判断价格位置（7个细分区域）
    """
    price = row.get('Current Price')
    cw = row.get('Call Wall')
    pw = row.get('Put Wall')
    
    if pd.isna(price) or pd.isna(cw) or pd.isna(pw):
        return "数据缺失", 0, 0
    
    price = parse_number(price) if isinstance(price, str) else price
    cw = parse_number(cw) if isinstance(cw, str) else cw
    pw = parse_number(pw) if isinstance(pw, str) else pw
    
    if not all([price, cw, pw]):
        return "数据缺失", 0, 0
    
    dist_to_cw = (cw - price) / price * 100
    dist_to_pw = (price - pw) / price * 100
    
    threshold_critical = 1  # 临界区1%
    threshold_observe = threshold  # 观察区5%
    
    if dist_to_cw < 0:
        return "已突破CW", dist_to_cw, dist_to_pw
    elif dist_to_pw < 0:
        return "已跌破PW", dist_to_cw, dist_to_pw
    elif dist_to_cw < threshold_critical:
        return "临界CW", dist_to_cw, dist_to_pw
    elif dist_to_pw < threshold_critical:
        return "临界PW", dist_to_cw, dist_to_pw
    elif dist_to_cw < threshold_observe:
        return "观察区CW", dist_to_cw, dist_to_pw
    elif dist_to_pw < threshold_observe:
        return "观察区PW", dist_to_cw, dist_to_pw
    else:
        return "中间区域", dist_to_cw, dist_to_pw

def get_trade_signal(position, structure, neg, dist_cw, dist_pw, has_cw_increase=False):
    """
    生成交易信号
    
    核心逻辑：
    - 正Gamma (Call主导): 墙是「盾」→ CW阻力, PW支撑 → 均值回归
    - 负Gamma (Put主导): 墙是「弹簧」→ CW突破加速, PW跌破加速 → 趋势跟随
    """
    # 判断Gamma环境
    is_put_side = structure in ["Put主导", "Put偏多", "Put轻微"]
    is_call_side = structure in ["Call主导", "Call偏多"]
    
    # NEG强度
    neg_val = parse_number(str(neg).replace('%', '')) if neg else 0
    if neg_val and neg_val < 1:
        neg_val = neg_val * 100
    
    if neg_val and neg_val > 40:
        confidence = "⭐⭐⭐"
    elif neg_val and neg_val > 25:
        confidence = "⭐⭐"
    else:
        confidence = "⭐"
    
    neg_str = f"NEG={neg_val:.0f}%" if neg_val else "NEG=N/A"
    
    # 周五Pinning检测
    is_friday = datetime.now().weekday() == 4
    pinning_warning = ""
    if is_friday and neg_val and neg_val > 35:
        pinning_warning = " | ⚠️周五+高NEG→Pinning效应"
    
    signal = ""
    logic = ""
    signal_type = "neutral"
    
    # 已突破CW
    if position == "已突破CW":
        if is_call_side:
            signal = f"🟡 谨慎追多 {confidence}"
            logic = f"正Gamma+已突破CW→需CW上移确认{pinning_warning}"
            signal_type = "bullish_cautious"
        elif is_put_side:
            signal = f"🚀 强势做多 {confidence}"
            logic = f"负Gamma+已突破CW→MM被迫买入{pinning_warning}"
            signal_type = "strong_bullish"
        else:
            signal = f"🟢 偏多观察 {confidence}"
            logic = f"已突破阻力，结构中性{pinning_warning}"
            signal_type = "bullish_watch"
    
    # 已跌破PW
    elif position == "已跌破PW":
        if is_call_side:
            signal = f"🟡 谨慎抄底 {confidence}"
            logic = f"正Gamma+已跌破PW→可能假跌破{pinning_warning}"
            signal_type = "bullish_cautious"
        elif is_put_side:
            signal = f"💀 强势做空 {confidence}"
            logic = f"负Gamma+已跌破PW→Gamma坍塌{pinning_warning}"
            signal_type = "strong_bearish"
        else:
            signal = f"🔴 偏空观察 {confidence}"
            logic = f"已跌破支撑，结构中性{pinning_warning}"
            signal_type = "bearish_watch"
    
    # 临界CW
    elif position == "临界CW":
        if is_call_side:
            signal = f"🔴 阻力减仓 {confidence}"
            logic = f"正Gamma→CW是盾→卖压最大{pinning_warning}"
            signal_type = "bearish"
        elif is_put_side:
            signal = f"🟢 突破做多 {confidence}"
            logic = f"负Gamma→CW是弹簧→突破后加速{pinning_warning}"
            signal_type = "bullish"
        else:
            signal = f"⚪ CW观望 {confidence}"
            logic = f"临界阻力，结构中性{pinning_warning}"
            signal_type = "neutral"
    
    # 临界PW
    elif position == "临界PW":
        if is_call_side:
            signal = f"🟢 支撑做多 {confidence}"
            logic = f"正Gamma→PW是盾→买盘支撑{pinning_warning}"
            signal_type = "bullish"
        elif is_put_side:
            signal = f"🔴 破位做空 {confidence}"
            logic = f"负Gamma→PW是薄冰→跌破后加速{pinning_warning}"
            signal_type = "bearish"
        else:
            signal = f"⚪ PW观望 {confidence}"
            logic = f"临界支撑，结构中性{pinning_warning}"
            signal_type = "neutral"
    
    # 观察区CW
    elif position == "观察区CW":
        if is_call_side:
            signal = f"🟡 接近阻力 {confidence}"
            logic = f"正Gamma→接近CW→准备减仓{pinning_warning}"
            signal_type = "bearish_watch"
        elif is_put_side:
            signal = f"🟢 突破潜力 {confidence}"
            logic = f"负Gamma→接近CW→有突破潜力{pinning_warning}"
            signal_type = "bullish_watch"
        else:
            signal = f"⚪ 接近CW {confidence}"
            logic = f"接近阻力，观望{pinning_warning}"
            signal_type = "neutral"
    
    # 观察区PW
    elif position == "观察区PW":
        if is_call_side:
            signal = f"🟢 接近支撑 {confidence}"
            logic = f"正Gamma→接近PW→准备做多{pinning_warning}"
            signal_type = "bullish_watch"
        elif is_put_side:
            signal = f"🟡 破位风险 {confidence}"
            logic = f"负Gamma→接近PW→有破位风险{pinning_warning}"
            signal_type = "bearish_watch"
        else:
            signal = f"⚪ 接近PW {confidence}"
            logic = f"接近支撑，观望{pinning_warning}"
            signal_type = "neutral"
    
    # 中间区域
    else:
        if is_call_side:
            signal = f"⚖️ 均值回归 {confidence}"
            logic = f"正Gamma+中间区域→高抛低吸"
            signal_type = "mean_reversion"
        elif is_put_side:
            signal = f"⚡ 趋势跟随 {confidence}"
            logic = f"负Gamma+中间区域→顺势交易"
            signal_type = "trend_follow"
        else:
            signal = f"⚪ 中性观望 {confidence}"
            logic = f"中间区域+结构中性→等待方向"
            signal_type = "neutral"
    
    # CW上移叠加
    if has_cw_increase:
        if signal_type in ['bullish', 'strong_bullish', 'bullish_watch', 'bullish_cautious']:
            # 做多信号+CW上移 → 增强
            confidence = confidence.replace("⭐⭐", "⭐⭐⭐").replace("⭐", "⭐⭐")
            signal = signal + " 🚀CW↑"
            logic = logic + " | CW上移确认，上方空间打开"
        elif signal_type == 'bearish':
            # 阻力减仓不叠加CW上移
            pass
        elif signal_type in ['bearish_watch', 'strong_bearish']:
            # 做空信号+CW上移 → 冲突，降级
            signal = f"⚠️ 信号冲突 {confidence}"
            logic = logic + " | ⚠️CW上移与做空信号冲突"
            signal_type = "neutral"
        else:
            # 中性信号+CW上移 → 升级为偏多
            signal = f"📈 CW上移偏多 {confidence}"
            logic = logic + " | CW上移，偏多观察"
            signal_type = "bullish_watch"
    
    return signal, logic, signal_type

def analyze_equity_hub(squeeze_df, cw_increase_df=None):
    """分析Equity Hub数据"""
    results = []
    
    # CW上移股票列表
    cw_increase_symbols = set()
    if cw_increase_df is not None and not cw_increase_df.empty:
        cw_increase_symbols = set(cw_increase_df['Symbol'].tolist())
    
    for _, row in squeeze_df.iterrows():
        symbol = row.get('Symbol', '')
        if not symbol:
            continue
        
        # 期权结构
        structure, structure_type = get_option_structure(row)
        
        # 价格位置
        position, dist_cw, dist_pw = get_position_zone(row)
        
        # NEG
        neg = row.get('Next Exp Gamma', 0)
        
        # 是否有CW上移
        has_cw_increase = symbol in cw_increase_symbols
        
        # 生成信号
        signal, logic, signal_type = get_trade_signal(
            position, structure, neg, dist_cw, dist_pw, has_cw_increase
        )
        
        results.append({
            'Symbol': symbol,
            'Current Price': row.get('Current Price'),
            'Call Wall': row.get('Call Wall'),
            'Put Wall': row.get('Put Wall'),
            'Position': position,
            'Dist_CW': dist_cw,
            'Dist_PW': dist_pw,
            'Structure': structure,
            'Structure_Type': structure_type,
            'Delta Ratio': row.get('Delta Ratio'),
            'Gamma Ratio': row.get('Gamma Ratio'),
            'NEG': neg,
            'Options Impact': row.get('Options Impact'),
            'Signal': signal,
            'Logic': logic,
            'Signal_Type': signal_type,
            'CW_Increase': has_cw_increase
        })
    
    # 处理仅在CW上移中出现的股票（独立弱信号）
    if cw_increase_df is not None:
        squeeze_symbols = set(squeeze_df['Symbol'].tolist()) if not squeeze_df.empty else set()
        
        for _, row in cw_increase_df.iterrows():
            symbol = row.get('Symbol', '')
            if symbol and symbol not in squeeze_symbols:
                results.append({
                    'Symbol': symbol,
                    'Current Price': row.get('Current Price'),
                    'Call Wall': row.get('Call Wall'),
                    'Put Wall': row.get('Put Wall'),
                    'Position': 'N/A',
                    'Dist_CW': 0,
                    'Dist_PW': 0,
                    'Structure': 'N/A',
                    'Structure_Type': 'unknown',
                    'Delta Ratio': row.get('Delta Ratio'),
                    'Gamma Ratio': row.get('Gamma Ratio'),
                    'NEG': row.get('Next Exp Gamma'),
                    'Options Impact': row.get('Options Impact'),
                    'Signal': '📈 CW上移观察 ⭐',
                    'Logic': 'CW上移独立信号，上方空间打开，但无Squeeze配合',
                    'Signal_Type': 'cw_increase_watch',
                    'CW_Increase': True
                })
    
    return pd.DataFrame(results)


# ============================================================
# 板块三：周五到期Gamma分析
# ============================================================

def calculate_gamma_direction(call_gamma, put_gamma):
    """
    计算Gamma Direction（到期方向指标）
    
    = |Put Gamma| / (|Put Gamma| + |Call Gamma|)
    > 0.6 = Put主导到期 → 高开倾向
    < 0.4 = Call主导到期 → 低开倾向
    """
    call_g = abs(parse_number(call_gamma)) if call_gamma else 0
    put_g = abs(parse_number(put_gamma)) if put_gamma else 0
    
    if call_g + put_g == 0:
        return 0.5, "均衡", "neutral"
    
    direction = put_g / (put_g + call_g)
    
    if direction > 0.6:
        return direction, "Put主导到期", "put_dominant"
    elif direction < 0.4:
        return direction, "Call主导到期", "call_dominant"
    else:
        return direction, "均衡", "neutral"

def get_neg_strength(neg):
    """获取NEG强度"""
    neg_val = parse_number(str(neg).replace('%', '')) if neg else 0
    if neg_val and neg_val < 1:
        neg_val = neg_val * 100
    
    if neg_val and neg_val > 40:
        return neg_val, "⭐⭐⭐", "strong"
    elif neg_val and neg_val > 25:
        return neg_val, "⭐⭐", "medium"
    else:
        return neg_val, "⭐", "weak"

def predict_monday_gap(friday_close_position, gamma_direction_type, neg_strength):
    """
    预测下周一跳空方向
    
    | 周五收盘位置 | Gamma Direction | 预测 |
    |--------------|-----------------|------|
    | > CW | Put主导 | 🚀 强势高开 |
    | > CW | Call主导 | ⚠️ 冲突观望 |
    | < PW | Call主导 | 💀 强势低开 |
    | < PW | Put主导 | ⚠️ 冲突观望 |
    | 区间内 | Put主导 | 📈 轻微高开 |
    | 区间内 | Call主导 | 📉 轻微低开 |
    """
    confidence = "⭐⭐⭐" if neg_strength == "strong" else ("⭐⭐" if neg_strength == "medium" else "⭐")
    
    if friday_close_position == "above_cw":
        if gamma_direction_type == "put_dominant":
            return f"🚀 强势高开 {confidence}", "strong_bullish", "位置>CW + Put Gamma到期 → MM买入平仓"
        elif gamma_direction_type == "call_dominant":
            return f"⚠️ 冲突观望 {confidence}", "neutral", "位置>CW(看多) vs Call Gamma到期(看空) → 方向不明"
        else:
            return f"📈 轻微高开 {confidence}", "bullish", "位置>CW + Gamma均衡"
    
    elif friday_close_position == "below_pw":
        if gamma_direction_type == "call_dominant":
            return f"💀 强势低开 {confidence}", "strong_bearish", "位置<PW + Call Gamma到期 → MM卖出平仓"
        elif gamma_direction_type == "put_dominant":
            return f"⚠️ 冲突观望 {confidence}", "neutral", "位置<PW(看空) vs Put Gamma到期(看多) → 方向不明"
        else:
            return f"📉 轻微低开 {confidence}", "bearish", "位置<PW + Gamma均衡"
    
    else:  # 区间内
        if gamma_direction_type == "put_dominant":
            return f"📈 轻微高开 {confidence}", "bullish", "区间内 + Put Gamma到期 → MM买入平仓倾向"
        elif gamma_direction_type == "call_dominant":
            return f"📉 轻微低开 {confidence}", "bearish", "区间内 + Call Gamma到期 → MM卖出平仓倾向"
        else:
            return f"⚖️ 方向不明 {confidence}", "neutral", "区间内 + Gamma均衡 → 观望"

def analyze_friday_expiry(df):
    """分析周五到期Gamma数据"""
    results = []
    
    for _, row in df.iterrows():
        symbol = row.get('Symbol', '')
        if not symbol:
            continue
        
        # 基础数据
        current_price = parse_number(row.get('Current Price'))
        call_wall = parse_number(row.get('Call Wall'))
        put_wall = parse_number(row.get('Put Wall'))
        call_gamma = row.get('Call Gamma')
        put_gamma = row.get('Put Gamma')
        neg = row.get('Next Exp Gamma')
        
        # Gamma Direction
        gd_value, gd_desc, gd_type = calculate_gamma_direction(call_gamma, put_gamma)
        
        # NEG强度
        neg_val, neg_stars, neg_strength = get_neg_strength(neg)
        
        # 当前位置（用于参考，实际预测在周五收盘后）
        current_position = "in_range"
        if current_price and call_wall and current_price > call_wall:
            current_position = "above_cw"
        elif current_price and put_wall and current_price < put_wall:
            current_position = "below_pw"
        
        # 预测（基于当前位置的参考预测）
        prediction, pred_type, pred_logic = predict_monday_gap(current_position, gd_type, neg_strength)
        
        results.append({
            'Symbol': symbol,
            'Current Price': current_price,
            'Call Wall': call_wall,
            'Put Wall': put_wall,
            'Call Gamma': call_gamma,
            'Put Gamma': put_gamma,
            'Gamma Direction': gd_value,
            'GD Desc': gd_desc,
            'GD Type': gd_type,
            'NEG': neg_val,
            'NEG Stars': neg_stars,
            'NEG Strength': neg_strength,
            'Current Position': current_position,
            'Prediction': prediction,
            'Pred Type': pred_type,
            'Pred Logic': pred_logic,
            'Pinning Range': f"{put_wall} - {call_wall}" if put_wall and call_wall else "N/A"
        })
    
    return pd.DataFrame(results)

# ============================================================
# 追踪管理函数
# ============================================================

def add_equity_hub_tracking(symbol, row, signal, signal_type, today_str):
    """添加Equity Hub追踪记录"""
    neg = row.get('NEG', 0)
    neg_val = parse_number(str(neg).replace('%', '')) if neg else 0
    
    # 到期日
    top_gamma_exp = row.get('Top Gamma Exp', '')
    try:
        if isinstance(top_gamma_exp, str) and top_gamma_exp:
            exp_date = datetime.strptime(top_gamma_exp, '%Y-%m-%d')
            track_end = (exp_date + timedelta(days=7)).strftime('%Y-%m-%d')
        else:
            track_end = (datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d')
    except:
        track_end = (datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d')
    
    # 信号方向
    bullish_types = ['bullish', 'strong_bullish', 'bullish_cautious', 'bullish_watch', 
                     'mean_reversion', 'cw_increase_watch']
    bearish_types = ['bearish', 'strong_bearish', 'bearish_watch', 'trend_follow']
    
    if signal_type in bullish_types:
        signal_direction = 'bullish'
    elif signal_type in bearish_types:
        signal_direction = 'bearish'
    else:
        signal_direction = 'neutral'
    
    return {
        'signal_date': today_str,
        'entry_price': float(row.get('Current Price', 0)),
        'signal': signal,
        'signal_type': signal_type,
        'signal_direction': signal_direction,
        'structure': row.get('Structure', ''),
        'position': row.get('Position', ''),
        'neg': neg_val,
        'call_wall': float(row.get('Call Wall', 0)) if row.get('Call Wall') else 0,
        'put_wall': float(row.get('Put Wall', 0)) if row.get('Put Wall') else 0,
        'cw_increase': row.get('CW_Increase', False),
        'track_end_date': track_end,
        'daily_prices': {today_str: float(row.get('Current Price', 0))},
        'current_return': 0,
        'direction_correct': None,
        'status': 'tracking'
    }

def add_friday_expiry_tracking(symbol, row, today_str):
    """添加周五到期追踪记录"""
    # 追踪到下周二
    days_to_tuesday = (8 - datetime.now().weekday()) % 7
    if days_to_tuesday == 0:
        days_to_tuesday = 7
    track_end = (datetime.now() + timedelta(days=days_to_tuesday + 1)).strftime('%Y-%m-%d')
    
    return {
        'week_start': today_str,
        'symbol': symbol,
        'entry_price': float(row.get('Current Price', 0)),
        'call_wall': float(row.get('Call Wall', 0)) if row.get('Call Wall') else 0,
        'put_wall': float(row.get('Put Wall', 0)) if row.get('Put Wall') else 0,
        'gamma_direction': row.get('Gamma Direction', 0.5),
        'gd_type': row.get('GD Type', 'neutral'),
        'neg_initial': row.get('NEG', 0),
        'neg_history': {today_str: row.get('NEG', 0)},
        'prediction': row.get('Prediction', ''),
        'pred_type': row.get('Pred Type', 'neutral'),
        'daily_prices': {today_str: float(row.get('Current Price', 0))},
        'friday_close': None,
        'friday_position': None,
        'monday_open': None,
        'monday_gap_pct': None,
        'gap_direction_correct': None,
        'track_end_date': track_end,
        'status': 'tracking',
        'neg_trend': 'stable'  # stable, rising, falling, dropped_off
    }

def update_tracking_prices(tracking_data, today_str):
    """更新追踪记录的价格"""
    updated = 0
    for symbol, record in tracking_data.items():
        if record.get('status') == 'tracking':
            price = get_current_price(symbol)
            if price:
                if 'daily_prices' not in record:
                    record['daily_prices'] = {}
                record['daily_prices'][today_str] = price
                
                entry_price = record.get('entry_price', 0)
                if entry_price > 0:
                    record['current_return'] = ((price - entry_price) / entry_price) * 100
                    
                    # 判断方向正确性
                    direction = record.get('signal_direction', 'neutral')
                    if direction == 'bullish':
                        record['direction_correct'] = record['current_return'] > 0
                    elif direction == 'bearish':
                        record['direction_correct'] = record['current_return'] < 0
                
                updated += 1
    
    return updated


# ============================================================
# Streamlit UI 主函数
# ============================================================

def main():
    st.title("📊 SpotGamma 期权分析系统 V2.0")
    st.caption(f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # ===== 统一文件上传入口 =====
    st.header("📁 数据上传")
    
    col_upload1, col_upload2 = st.columns([2, 1])
    
    with col_upload1:
        uploaded_files = st.file_uploader(
            "上传CSV文件（支持多文件，自动识别类型）",
            type=['csv'],
            accept_multiple_files=True,
            help="支持: QQQ历史数据, Squeeze, Call Wall Increase, Friday Expiry"
        )
    
    with col_upload2:
        st.markdown("**CSV类型自动识别:**")
        st.caption("• 有Date列 → QQQ历史")
        st.caption("• 有Call Gamma + Options Impact → CW上移")
        st.caption("• 有Call Gamma 无Options Impact → 周五到期")
        st.caption("• 其他 → Squeeze")
    
    # QQQ/NQ盘前数据粘贴框
    st.subheader("📋 QQQ/NQ 盘前数据")
    premarket_text = st.text_area(
        "粘贴盘前数据（QQQ和NQ）",
        height=200,
        placeholder="""QQQ盘前现价：__619.14__，昨收__620.78__ 
630 Call Wall 
625 Large Gamma 3 
...

NQ盘前现价__25587__，昨收__25646__，第二列为NQ的数值 
25901 26020 Combo 4 
..."""
    )
    
    # 解析上传的文件
    csv_data = {
        'qqq_history': None,
        'squeeze': None,
        'call_wall_increase': None,
        'friday_expiry': None
    }
    
    if uploaded_files:
        for file in uploaded_files:
            try:
                df = pd.read_csv(file)
                csv_type = identify_csv_type(df)
                
                if csv_type == 'qqq_history':
                    csv_data['qqq_history'] = df
                    st.success(f"✅ {file.name} → QQQ历史数据 ({len(df)}行)")
                elif csv_type == 'squeeze':
                    csv_data['squeeze'] = df
                    st.success(f"✅ {file.name} → Squeeze数据 ({len(df)}行)")
                elif csv_type == 'call_wall_increase':
                    csv_data['call_wall_increase'] = df
                    st.success(f"✅ {file.name} → Call Wall上移数据 ({len(df)}行)")
                elif csv_type == 'friday_expiry':
                    csv_data['friday_expiry'] = df
                    st.success(f"✅ {file.name} → 周五到期Gamma数据 ({len(df)}行)")
                else:
                    st.warning(f"⚠️ {file.name} → 无法识别类型")
            except Exception as e:
                st.error(f"❌ {file.name} 读取失败: {e}")
    
    st.divider()
    
    # ================================================================
    # 板块一：QQQ/NQ盘前分析
    # ================================================================
    st.header("📈 板块一：QQQ/NQ 盘前分析")
    
    if premarket_text.strip():
        premarket_data = parse_qqq_premarket_text(premarket_text)
        
        # 获取历史数据（如果有）
        history_row = None
        if csv_data['qqq_history'] is not None:
            today_str = datetime.now().strftime('%Y/%m/%d')
            hist_df = csv_data['qqq_history']
            if 'Date' in hist_df.columns:
                # 尝试匹配今天或最近的日期
                hist_df['Date'] = hist_df['Date'].astype(str)
                matched = hist_df[hist_df['Date'] == today_str]
                if matched.empty:
                    # 取最后一行
                    history_row = hist_df.iloc[-1].to_dict() if not hist_df.empty else None
                else:
                    history_row = matched.iloc[0].to_dict()
        
        # 分析
        analysis = analyze_qqq_nq(premarket_data, history_row)
        
        # 显示分析结果
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("QQQ")
            qqq = analysis.get('qqq', {})
            if qqq.get('current'):
                change_pct = qqq.get('change_pct', 0)
                change_color = "🟢" if change_pct >= 0 else "🔴"
                st.metric(
                    "盘前价",
                    f"${qqq['current']:.2f}",
                    f"{change_pct:+.2f}% vs 昨收"
                )
                st.write(f"**Gamma环境**: {qqq.get('gamma_env', 'N/A')}")
                st.write(f"**波动环境**: {qqq.get('vol_regime', 'N/A')}")
                
                # 关键位
                with st.expander("📊 关键位置"):
                    for name, price in qqq.get('levels', {}).items():
                        st.write(f"• {name}: {price}")
        
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
                st.write(f"**Gamma环境**: {nq.get('gamma_env', 'N/A')}")
                st.write(f"**波动环境**: {nq.get('vol_regime', 'N/A')}")
                
                with st.expander("📊 关键位置"):
                    for name, price in nq.get('levels', {}).items():
                        st.write(f"• {name}: {price}")
        
        # 交叉验证
        cv = analysis.get('cross_validation', {})
        if cv:
            if cv.get('status') == '矛盾':
                st.warning(cv.get('message', ''))
            else:
                st.success(cv.get('message', ''))
        
        # 预测
        st.subheader("🔮 日内预测")
        pred = analysis.get('prediction', {})
        if pred:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("阻力区", pred.get('resistance', 'N/A'))
            with col2:
                st.metric("支撑区", pred.get('support', 'N/A'))
            with col3:
                st.metric("收盘预测", pred.get('close_range', 'N/A'))
        
        # 明日开盘预测
        if history_row:
            st.subheader("🌅 明日开盘预测")
            next_day = predict_next_day_open(history_row, analysis)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                direction_emoji = "📈" if next_day['direction_type'] == 'bullish' else (
                    "📉" if next_day['direction_type'] == 'bearish' else "⚖️"
                )
                st.metric("预测方向", f"{direction_emoji} {next_day['direction']}")
            
            with col2:
                st.write("**判断依据:**")
                for factor in next_day['factors']:
                    st.write(f"• {factor}")
            
            # DPI信息
            if history_row:
                dpi = history_row.get('DPI')
                dpi_5d = history_row.get('5Day DPI')
                if dpi or dpi_5d:
                    st.subheader("🏦 机构动向 (DPI)")
                    col1, col2 = st.columns(2)
                    with col1:
                        if dpi:
                            dpi_val = parse_number(str(dpi).replace('%', ''))
                            if dpi_val:
                                dpi_status = "机构积极买入" if dpi_val > 50 else (
                                    "机构中性" if dpi_val > 45 else "机构买盘减弱"
                                )
                                st.metric("当日DPI", f"{dpi_val:.1f}%", dpi_status)
                    with col2:
                        if dpi_5d:
                            dpi_5d_val = parse_number(str(dpi_5d).replace('%', ''))
                            if dpi_5d_val:
                                dpi_5d_status = "🏦 持续强力买入" if dpi_5d_val > 52 else (
                                    "🏦 积极买入" if dpi_5d_val > 50 else "中性"
                                )
                                st.metric("5日DPI", f"{dpi_5d_val:.1f}%", dpi_5d_status)
    else:
        st.info("👆 请在上方粘贴QQQ/NQ盘前数据")
    
    st.divider()
    
    # ================================================================
    # 板块二：Equity Hub分析
    # ================================================================
    st.header("📊 板块二：Equity Hub 分析")
    
    if csv_data['squeeze'] is not None or csv_data['call_wall_increase'] is not None:
        squeeze_df = csv_data['squeeze'] if csv_data['squeeze'] is not None else pd.DataFrame()
        cw_df = csv_data['call_wall_increase']
        
        # 分析
        results_df = analyze_equity_hub(squeeze_df, cw_df)
        
        if not results_df.empty:
            # 统计
            st.subheader("📊 信号概览")
            
            bullish_types = ['bullish', 'strong_bullish', 'bullish_cautious', 'bullish_watch', 'cw_increase_watch']
            bearish_types = ['bearish', 'strong_bearish', 'bearish_watch']
            
            bullish_count = len(results_df[results_df['Signal_Type'].isin(bullish_types)])
            bearish_count = len(results_df[results_df['Signal_Type'].isin(bearish_types)])
            cw_increase_count = len(results_df[results_df['CW_Increase'] == True])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🟢 做多信号", bullish_count)
            with col2:
                st.metric("🔴 做空信号", bearish_count)
            with col3:
                st.metric("🚀 CW上移", cw_increase_count)
            with col4:
                st.metric("📊 总计", len(results_df))
            
            # 做多信号列表
            st.subheader("🟢 做多信号")
            bullish_df = results_df[results_df['Signal_Type'].isin(bullish_types)].copy()
            if not bullish_df.empty:
                for _, row in bullish_df.iterrows():
                    cw_tag = " 🚀CW↑" if row['CW_Increase'] else ""
                    with st.container():
                        st.markdown(f"**{row['Symbol']}** ${row['Current Price']:.2f}{cw_tag}")
                        st.caption(f"{row['Signal']}")
                        st.write(f"位置: {row['Position']} | 结构: {row['Structure']} | NEG: {row['NEG']}")
                        st.write(f"逻辑: {row['Logic']}")
                        st.divider()
            else:
                st.info("暂无做多信号")
            
            # 做空信号列表
            st.subheader("🔴 做空信号")
            bearish_df = results_df[results_df['Signal_Type'].isin(bearish_types)].copy()
            if not bearish_df.empty:
                for _, row in bearish_df.iterrows():
                    with st.container():
                        st.markdown(f"**{row['Symbol']}** ${row['Current Price']:.2f}")
                        st.caption(f"{row['Signal']}")
                        st.write(f"位置: {row['Position']} | 结构: {row['Structure']} | NEG: {row['NEG']}")
                        st.write(f"逻辑: {row['Logic']}")
                        st.divider()
            else:
                st.info("暂无做空信号")
            
            # 完整表格
            with st.expander("📋 查看完整分析表"):
                display_cols = ['Symbol', 'Current Price', 'Signal', 'Position', 'Structure', 
                               'Delta Ratio', 'Gamma Ratio', 'NEG', 'CW_Increase']
                st.dataframe(results_df[display_cols], use_container_width=True, hide_index=True)
            
            # 追踪功能
            st.subheader("📈 Equity Hub 追踪")
            
            # 加载追踪数据
            eh_tracking = load_worksheet_data(WS_EQUITY_HUB) or {}
            today_str = datetime.now().strftime('%Y-%m-%d')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("➕ 添加信号到追踪", key="add_eh_tracking"):
                    new_count = 0
                    reset_count = 0
                    for _, row in results_df.iterrows():
                        symbol = row['Symbol']
                        signal_type = row['Signal_Type']
                        if signal_type != 'neutral':
                            if symbol not in eh_tracking:
                                eh_tracking[symbol] = add_equity_hub_tracking(
                                    symbol, row, row['Signal'], signal_type, today_str
                                )
                                new_count += 1
                            else:
                                # 信号变化则重置
                                old_type = eh_tracking[symbol].get('signal_type', '')
                                if old_type != signal_type:
                                    eh_tracking[symbol] = add_equity_hub_tracking(
                                        symbol, row, row['Signal'], signal_type, today_str
                                    )
                                    reset_count += 1
                    
                    if new_count > 0 or reset_count > 0:
                        save_worksheet_data(WS_EQUITY_HUB, eh_tracking)
                        st.success(f"✅ 新增{new_count}个，重置{reset_count}个")
                    else:
                        st.info("无新信号需要添加")
            
            with col2:
                if st.button("🔄 刷新价格", key="refresh_eh_prices"):
                    updated = update_tracking_prices(eh_tracking, today_str)
                    save_worksheet_data(WS_EQUITY_HUB, eh_tracking)
                    st.success(f"✅ 更新了{updated}个标的价格")
            
            with col3:
                if st.button("🗑️ 清空追踪", key="clear_eh_tracking"):
                    save_worksheet_data(WS_EQUITY_HUB, {})
                    eh_tracking = {}
                    st.success("✅ 已清空")
            
            # 显示追踪记录
            if eh_tracking:
                st.write(f"**追踪中: {len(eh_tracking)}个标的**")
                
                # 统计正确率
                correct = sum(1 for r in eh_tracking.values() if r.get('direction_correct') == True)
                wrong = sum(1 for r in eh_tracking.values() if r.get('direction_correct') == False)
                total = correct + wrong
                accuracy = (correct / total * 100) if total > 0 else 0
                
                st.metric("信号准确率", f"{accuracy:.1f}%", f"{correct}/{total} 正确")
                
                # 追踪表格
                tracking_rows = []
                for symbol, record in eh_tracking.items():
                    direction = "🟢多" if record.get('signal_direction') == 'bullish' else (
                        "🔴空" if record.get('signal_direction') == 'bearish' else "⚪中"
                    )
                    correct_status = "✅" if record.get('direction_correct') == True else (
                        "❌" if record.get('direction_correct') == False else "⏳"
                    )
                    tracking_rows.append({
                        '标的': symbol,
                        '方向': direction,
                        '入场价': record.get('entry_price', 0),
                        '当前收益': f"{record.get('current_return', 0):.2f}%",
                        '正确': correct_status,
                        'CW↑': "✓" if record.get('cw_increase') else "",
                        '状态': record.get('status', 'tracking')
                    })
                
                st.dataframe(pd.DataFrame(tracking_rows), use_container_width=True, hide_index=True)
    else:
        st.info("👆 请上传 Squeeze 或 Call Wall Increase CSV文件")
    
    st.divider()

    
    # ================================================================
    # 板块三：周五到期Gamma分析
    # ================================================================
    st.header("📅 板块三：周五到期 Gamma 分析")
    
    if csv_data['friday_expiry'] is not None:
        friday_df = csv_data['friday_expiry']
        
        # 分析
        friday_results = analyze_friday_expiry(friday_df)
        
        if not friday_results.empty:
            # 统计
            st.subheader("📊 Gamma Direction 概览")
            
            put_dominant = len(friday_results[friday_results['GD Type'] == 'put_dominant'])
            call_dominant = len(friday_results[friday_results['GD Type'] == 'call_dominant'])
            neutral = len(friday_results[friday_results['GD Type'] == 'neutral'])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📈 Put主导到期", put_dominant, "高开倾向")
            with col2:
                st.metric("📉 Call主导到期", call_dominant, "低开倾向")
            with col3:
                st.metric("⚖️ 均衡", neutral)
            with col4:
                st.metric("📊 总计", len(friday_results))
            
            # 高NEG标的
            st.subheader("🎯 高NEG标的 (>25%)")
            high_neg = friday_results[friday_results['NEG'] > 25].sort_values('NEG', ascending=False)
            
            if not high_neg.empty:
                for _, row in high_neg.iterrows():
                    gd_emoji = "📈" if row['GD Type'] == 'put_dominant' else (
                        "📉" if row['GD Type'] == 'call_dominant' else "⚖️"
                    )
                    
                    with st.container():
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.markdown(f"**{row['Symbol']}** ${row['Current Price']:.2f}")
                            st.caption(f"NEG: {row['NEG']:.1f}% {row['NEG Stars']}")
                        with col2:
                            st.write(f"{gd_emoji} **{row['GD Desc']}** (GD={row['Gamma Direction']:.2f})")
                            st.write(f"Pinning区间: {row['Pinning Range']}")
                            st.write(f"**预测**: {row['Prediction']}")
                            st.caption(f"逻辑: {row['Pred Logic']}")
                        st.divider()
            else:
                st.info("暂无高NEG标的")
            
            # 完整表格
            with st.expander("📋 查看完整分析表"):
                display_cols = ['Symbol', 'Current Price', 'Call Wall', 'Put Wall', 
                               'Gamma Direction', 'GD Desc', 'NEG', 'NEG Stars', 'Prediction']
                st.dataframe(friday_results[display_cols].round(3), use_container_width=True, hide_index=True)
            
            # 追踪功能
            st.subheader("📈 周五到期 追踪")
            
            # 加载追踪数据
            fe_tracking = load_worksheet_data(WS_FRIDAY_EXPIRY) or {}
            today_str = datetime.now().strftime('%Y-%m-%d')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("➕ 添加到追踪", key="add_fe_tracking"):
                    new_count = 0
                    updated_count = 0
                    
                    for _, row in friday_results.iterrows():
                        symbol = row['Symbol']
                        row_dict = row.to_dict()
                        
                        if symbol not in fe_tracking:
                            fe_tracking[symbol] = add_friday_expiry_tracking(symbol, row_dict, today_str)
                            new_count += 1
                        else:
                            # 更新NEG
                            old_neg = fe_tracking[symbol].get('neg_initial', 0)
                            new_neg = row['NEG']
                            
                            if 'neg_history' not in fe_tracking[symbol]:
                                fe_tracking[symbol]['neg_history'] = {}
                            fe_tracking[symbol]['neg_history'][today_str] = new_neg
                            
                            # NEG趋势
                            if new_neg > old_neg * 1.1:
                                fe_tracking[symbol]['neg_trend'] = 'rising'
                            elif new_neg < old_neg * 0.9:
                                fe_tracking[symbol]['neg_trend'] = 'falling'
                            else:
                                fe_tracking[symbol]['neg_trend'] = 'stable'
                            
                            # 更新预测
                            fe_tracking[symbol]['prediction'] = row['Prediction']
                            fe_tracking[symbol]['pred_type'] = row['Pred Type']
                            fe_tracking[symbol]['gamma_direction'] = row['Gamma Direction']
                            fe_tracking[symbol]['gd_type'] = row['GD Type']
                            
                            updated_count += 1
                    
                    # 检查消失的标的
                    current_symbols = set(friday_results['Symbol'].tolist())
                    for symbol in fe_tracking:
                        if symbol not in current_symbols and fe_tracking[symbol].get('status') == 'tracking':
                            fe_tracking[symbol]['neg_trend'] = 'dropped_off'
                    
                    if new_count > 0 or updated_count > 0:
                        save_worksheet_data(WS_FRIDAY_EXPIRY, fe_tracking)
                        st.success(f"✅ 新增{new_count}个，更新{updated_count}个")
                    else:
                        st.info("无新标的需要添加")
            
            with col2:
                if st.button("🔄 刷新价格", key="refresh_fe_prices"):
                    updated = update_tracking_prices(fe_tracking, today_str)
                    save_worksheet_data(WS_FRIDAY_EXPIRY, fe_tracking)
                    st.success(f"✅ 更新了{updated}个标的价格")
            
            with col3:
                if st.button("🗑️ 清空追踪", key="clear_fe_tracking"):
                    save_worksheet_data(WS_FRIDAY_EXPIRY, {})
                    fe_tracking = {}
                    st.success("✅ 已清空")
            
            # 显示追踪记录
            if fe_tracking:
                st.write(f"**追踪中: {len(fe_tracking)}个标的**")
                
                # 统计
                correct = sum(1 for r in fe_tracking.values() if r.get('gap_direction_correct') == True)
                wrong = sum(1 for r in fe_tracking.values() if r.get('gap_direction_correct') == False)
                total = correct + wrong
                accuracy = (correct / total * 100) if total > 0 else 0
                
                if total > 0:
                    st.metric("跳空预测准确率", f"{accuracy:.1f}%", f"{correct}/{total} 正确")
                
                # 追踪表格
                tracking_rows = []
                for symbol, record in fe_tracking.items():
                    neg_trend_emoji = "📈" if record.get('neg_trend') == 'rising' else (
                        "📉" if record.get('neg_trend') == 'falling' else (
                            "⚠️" if record.get('neg_trend') == 'dropped_off' else "➡️"
                        )
                    )
                    
                    gap_status = "✅" if record.get('gap_direction_correct') == True else (
                        "❌" if record.get('gap_direction_correct') == False else "⏳"
                    )
                    
                    tracking_rows.append({
                        '标的': symbol,
                        '入场价': record.get('entry_price', 0),
                        '当前收益': f"{record.get('current_return', 0):.2f}%",
                        'NEG初始': f"{record.get('neg_initial', 0):.1f}%",
                        'NEG趋势': neg_trend_emoji,
                        '预测': record.get('prediction', '')[:20],
                        '跳空验证': gap_status,
                        '状态': record.get('status', 'tracking')
                    })
                
                st.dataframe(pd.DataFrame(tracking_rows), use_container_width=True, hide_index=True)
                
                # 周五收盘记录（手动输入）
                with st.expander("📝 记录周五收盘/周一开盘"):
                    st.write("选择标的并记录周五收盘价或周一开盘价")
                    
                    symbol_to_update = st.selectbox(
                        "选择标的",
                        options=list(fe_tracking.keys()),
                        key="fe_symbol_update"
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        friday_close = st.number_input("周五收盘价", min_value=0.0, key="friday_close_input")
                        if st.button("记录周五收盘", key="record_friday"):
                            if symbol_to_update and friday_close > 0:
                                record = fe_tracking[symbol_to_update]
                                record['friday_close'] = friday_close
                                
                                # 判断位置
                                cw = record.get('call_wall', 0)
                                pw = record.get('put_wall', 0)
                                if friday_close > cw:
                                    record['friday_position'] = 'above_cw'
                                elif friday_close < pw:
                                    record['friday_position'] = 'below_pw'
                                else:
                                    record['friday_position'] = 'in_range'
                                
                                save_worksheet_data(WS_FRIDAY_EXPIRY, fe_tracking)
                                st.success(f"✅ 已记录{symbol_to_update}周五收盘: ${friday_close}")
                    
                    with col2:
                        monday_open = st.number_input("周一开盘价", min_value=0.0, key="monday_open_input")
                        if st.button("记录周一开盘", key="record_monday"):
                            if symbol_to_update and monday_open > 0:
                                record = fe_tracking[symbol_to_update]
                                record['monday_open'] = monday_open
                                
                                # 计算跳空
                                friday_close = record.get('friday_close')
                                if friday_close:
                                    gap_pct = ((monday_open - friday_close) / friday_close) * 100
                                    record['monday_gap_pct'] = gap_pct
                                    
                                    # 验证预测
                                    pred_type = record.get('pred_type', 'neutral')
                                    if pred_type in ['bullish', 'strong_bullish'] and gap_pct > 0:
                                        record['gap_direction_correct'] = True
                                    elif pred_type in ['bearish', 'strong_bearish'] and gap_pct < 0:
                                        record['gap_direction_correct'] = True
                                    elif pred_type == 'neutral':
                                        record['gap_direction_correct'] = None
                                    else:
                                        record['gap_direction_correct'] = False
                                
                                save_worksheet_data(WS_FRIDAY_EXPIRY, fe_tracking)
                                st.success(f"✅ 已记录{symbol_to_update}周一开盘: ${monday_open}")
    else:
        st.info("👆 请上传 Top Gamma Expiring This Friday CSV文件")
    
    st.divider()
    
    # ================================================================
    # 使用说明
    # ================================================================
    with st.expander("📖 使用说明"):
        st.markdown("""
        ## 系统概述
        
        本系统整合SpotGamma期权数据，提供三大分析板块：
        
        ### 板块一：QQQ/NQ盘前分析
        - 粘贴盘前数据，自动解析关键位置
        - 判断Gamma环境（正/负Gamma）
        - 判断波动环境（均值回归/趋势）
        - NQ与QQQ交叉验证，矛盾时以NQ为主
        - 日内预测：阻力、支撑、收盘区间
        - 明日开盘预测：综合NEG、GR、DPI判断
        
        ### 板块二：Equity Hub分析
        - 支持Squeeze和Call Wall Increase两种CSV
        - 7区域位置判定：已突破CW/已跌破PW/临界CW/临界PW/观察区CW/观察区PW/中间区域
        - 核心逻辑：
          - 正Gamma(Call主导)：墙是「盾」→ CW阻力, PW支撑 → 均值回归
          - 负Gamma(Put主导)：墙是「弹簧」→ CW突破加速, PW跌破加速 → 趋势跟随
        - CW上移叠加：做多信号增强，做空信号冲突
        - 追踪验证：涨就对(做多)，跌就对(做空)
        
        ### 板块三：周五到期Gamma分析
        - Gamma Direction = |Put Gamma| / (|Put Gamma| + |Call Gamma|)
          - > 0.6 = Put主导到期 → 高开倾向（MM买入平仓）
          - < 0.4 = Call主导到期 → 低开倾向（MM卖出平仓）
        - NEG强度：>40%强，25-40%中，<25%弱
        - 追踪周期：周一上传 → 下周二验证
        - NEG趋势追踪：上升/下降/消失
        
        ### 数据追踪
        - 所有追踪数据保存到Google Sheets
        - 支持一键刷新价格
        - 自动计算信号准确率
        """)

if __name__ == "__main__":
    main()
