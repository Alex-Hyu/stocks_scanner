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
from datetime import datetime
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
    
    bb = ta.bbands(df['Close'], length=20, std=2.0)
    if bb is not None:
        df['BB_Upper'] = bb['BBU_20_2.0']
        df['BB_Lower'] = bb['BBL_20_2.0']
        df['BB_Mid'] = bb['BBM_20_2.0']
    
    kc = ta.kc(df['High'], df['Low'], df['Close'], length=20, scalar=1.5)
    if kc is not None:
        df['KC_Upper'] = kc['KCUe_20_1.5']
        df['KC_Lower'] = kc['KCLe_20_1.5']
    
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
        
        default_tickers = "AAPL,MSFT,NVDA,TSLA,AMD,META,GOOGL,AMZN,NFLX,CRM"
        ticker_input = st.text_area(
            "输入股票代码（逗号分隔）",
            value=default_tickers,
            height=100
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            min_score = st.slider("最低评分", 0, 5, 2)
        with col2:
            direction_filter = st.selectbox("信号方向", ["全部", "看多", "看空"])
        with col3:
            wind_filter = st.selectbox("顺风/逆风", ["全部", "顺风", "逆风"])
        
        if st.button("🔍 开始筛选", key="stock_scan"):
            tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
            
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
        st.header("SpotGamma Squeeze验证")
        
        uploaded_file = st.file_uploader("上传SpotGamma CSV文件", type=['csv'])
        
        if uploaded_file is not None:
            try:
                sg_df = pd.read_csv(uploaded_file, header=1)
                sg_df = sg_df.dropna(subset=['Symbol'])
                
                st.subheader("Squeeze名单分析")
                
                analysis_results = []
                
                for _, row in sg_df.iterrows():
                    ticker = row['Symbol']
                    
                    try:
                        price = float(row.get('Current Price', 0))
                        gamma_strike = float(row.get('Key Gamma Strike', 0))
                        call_wall = float(row.get('Call Wall', 0))
                        put_wall = float(row.get('Put Wall', 0))
                        delta_ratio_raw = row.get('Delta Ratio', 0)
                        delta_ratio = float(str(delta_ratio_raw).replace("'", "").replace(",", "")) if delta_ratio_raw else 0
                        options_impact = float(row.get('Options Impact', 0)) if row.get('Options Impact') else 0
                    except:
                        continue
                    
                    # 方向判断
                    if price > gamma_strike:
                        gamma_direction = "↗️ 偏多"
                    else:
                        gamma_direction = "↘️ 偏空"
                    
                    if delta_ratio < -5:
                        gamma_direction += " (强)"
                    elif delta_ratio > 5:
                        gamma_direction = "↗️ 偏多 (强)"
                    
                    # 风险等级
                    if options_impact > 50:
                        risk = "🔴 极高"
                    elif options_impact > 30:
                        risk = "🟠 高"
                    else:
                        risk = "🟢 中"
                    
                    # 检查是否在筛选名单中
                    in_watchlist = "❌"
                    if 'stock_results' in st.session_state:
                        watchlist = st.session_state['stock_results']
                        passed_tickers = watchlist[watchlist['passed'] == True]['ticker'].tolist()
                        if ticker in passed_tickers:
                            in_watchlist = "✅"
                    
                    analysis_results.append({
                        '代码': ticker,
                        '价格': f"${price:.2f}",
                        'Gamma Strike': gamma_strike,
                        'Call Wall': call_wall,
                        'Put Wall': put_wall,
                        'Gamma方向': gamma_direction,
                        'Options Impact': f"{options_impact:.1f}%",
                        '风险': risk,
                        '在筛选名单': in_watchlist,
                    })
                
                if analysis_results:
                    analysis_df = pd.DataFrame(analysis_results)
                    st.dataframe(analysis_df, use_container_width=True, hide_index=True)
                    
                    # 交叉验证
                    st.subheader("🎯 交叉验证")
                    overlap = [r['代码'] for r in analysis_results if r['在筛选名单'] == '✅']
                    
                    if overlap:
                        st.success(f"同时出现在两个名单: **{', '.join(overlap)}**")
                        
                        for ticker in overlap:
                            sg_row = next((r for r in analysis_results if r['代码'] == ticker), None)
                            if sg_row and 'stock_results' in st.session_state:
                                stock_row = st.session_state['stock_results']
                                stock_row = stock_row[stock_row['ticker'] == ticker].iloc[0]
                                
                                st.markdown(f"""
                                ---
                                **{ticker}** 双重验证:  
                                - 技术信号: {stock_row['direction']} | {' '.join(stock_row['signals'])}  
                                - Gamma信号: {sg_row['Gamma方向']}  
                                - 风险等级: {sg_row['风险']}
                                """)
                    else:
                        st.info("无重叠股票")
                        
            except Exception as e:
                st.error(f"读取文件失败: {e}")
    
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
        
        **信号说明:**
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
        
        顺风置信度更高，逆风需谨慎。
        """)


if __name__ == "__main__":
    main()
