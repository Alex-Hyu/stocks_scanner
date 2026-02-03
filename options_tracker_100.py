"""
100只期权追踪系统
- 每日从CSV提取数据
- 存储到Google Sheets
- 生成交易信号
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# ============================================================
# 100只股票清单
# ============================================================

WATCHLIST_100 = {
    # === MAG7 七巨头 (7只) ===
    'MAG7': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'],
    
    # === AI/芯片 (15只) ===
    'AI_SEMI': ['AMD', 'INTC', 'MU', 'AVGO', 'TSM', 'MRVL', 'LRCX', 'SMCI', 'ARM', 
                'PLTR', 'AI', 'APP', 'IONQ', 'RGTI', 'SOUN'],
    
    # === 加密货币相关 (10只) ===
    'CRYPTO': ['IBIT', 'MSTR', 'MARA', 'COIN', 'RIOT', 'CLSK', 'WULF', 'IREN', 'CORZ', 'GLXY'],
    
    # === Meme/热门成长 (15只) ===
    'MEME_GROWTH': ['GME', 'AMC', 'SOFI', 'HOOD', 'RIVN', 'NIO', 'RKLB', 'JOBY', 'LUNR', 
                   'ACHR', 'CVNA', 'DKNG', 'HIMS', 'ASTS', 'OKLO'],
    
    # === 软件/云/SaaS (10只) ===
    'SOFTWARE': ['CRM', 'NOW', 'ORCL', 'SHOP', 'SNOW', 'NET', 'DDOG', 'CRWD', 'ZS', 'MDB'],
    
    # === 消费/媒体 (8只) ===
    'CONSUMER': ['NFLX', 'DIS', 'UBER', 'SNAP', 'SPOT', 'ABNB', 'BA', 'LUV'],
    
    # === 金融 (5只) ===
    'FINANCE': ['JPM', 'BAC', 'C', 'PYPL', 'SQ'],
    
    # === 能源/材料 (8只) ===
    'ENERGY': ['XOM', 'CVX', 'OXY', 'FCX', 'AA', 'MP', 'SMR', 'BE'],
    
    # === 黄金白银 (8只) ===
    'GOLD_SILVER': ['GLD', 'SLV', 'GDX', 'AG', 'NEM', 'KGC', 'AEM', 'GOLD'],
    
    # === 中概股 (8只) ===
    'CHINA': ['BABA', 'JD', 'PDD', 'XPEV', 'LI', 'BILI', 'TAL', 'KWEB'],
    
    # === 核心ETF/指数 (6只) ===
    'ETF_INDEX': ['SPY', 'SPX', 'QQQ', 'IWM', 'SOXL', 'VIX'],
}

def get_watchlist_flat():
    """获取扁平化的股票列表"""
    all_stocks = []
    for sector, stocks in WATCHLIST_100.items():
        for s in stocks:
            if s not in all_stocks:
                all_stocks.append(s)
    return all_stocks

def get_sector(symbol):
    """获取股票所属板块"""
    for sector, stocks in WATCHLIST_100.items():
        if symbol in stocks:
            return sector
    return 'OTHER'

# ============================================================
# 数据解析函数
# ============================================================

def parse_number(value):
    """安全解析数字"""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).strip().replace(',', '').replace("'", "-")
    try:
        return float(value.replace('%', '').replace('$', ''))
    except:
        return None

def parse_dpi(value):
    """解析DPI值（0.91 -> 91%）"""
    val = parse_number(value)
    if val is None:
        return None
    if 0 <= val <= 1:
        return val * 100
    return val

# ============================================================
# 提取100只股票数据
# ============================================================

def extract_watchlist_data(df, date_str=None):
    """
    从完整CSV中提取100只股票的数据
    
    参数:
        df: 完整的DataFrame
        date_str: 日期字符串，如 '2026-02-03'，如果为None则使用今天
    
    返回:
        DataFrame: 100只股票的关键数据
    """
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    watchlist = get_watchlist_flat()
    
    # 筛选100只股票
    df_filtered = df[df['Symbol'].isin(watchlist)].copy()
    
    # 提取关键字段
    result = []
    for _, row in df_filtered.iterrows():
        symbol = row.get('Symbol', '')
        
        record = {
            'Date': date_str,
            'Symbol': symbol,
            'Sector': get_sector(symbol),
            
            # 价格信息
            'Price': parse_number(row.get('Current Price')),
            'Prev_Close': parse_number(row.get('Previous Close')),
            
            # Gamma位置
            'Call_Wall': parse_number(row.get('Call Wall')),
            'Put_Wall': parse_number(row.get('Put Wall')),
            'Hedge_Wall': parse_number(row.get('Hedge Wall')),
            'Key_Gamma': parse_number(row.get('Key Gamma Strike')),
            
            # 核心指标
            'Delta_Ratio': parse_number(row.get('Delta Ratio')),
            'Gamma_Ratio': parse_number(row.get('Gamma Ratio')),
            'Volume_Ratio': parse_number(row.get('Volume Ratio')),
            
            # DPI
            'DPI_Pct': parse_dpi(row.get('% DPI Volume')),
            'DPI_5d_Pct': parse_dpi(row.get('5d % DPI Volume') or row.get('5 day DPI')),
            
            # Gamma到期
            'Next_Exp_Gamma': parse_number(row.get('Next Exp Gamma')),
            'Next_Exp_Call_Vol': parse_number(row.get('Next Exp Call Vol')),
            'Next_Exp_Put_Vol': parse_number(row.get('Next Exp Put Vol')),
            
            # IV
            'IV_Rank': parse_number(row.get('IV Rank')),
            'Implied_Move': parse_number(row.get('Options Implied Move')),
            'IV_1M': parse_number(row.get('1 M IV')),
            
            # 其他
            'Options_Impact': parse_number(row.get('Options Impact')),
            'Call_Volume': parse_number(row.get('Call Volume')),
            'Put_Volume': parse_number(row.get('Put Volume')),
        }
        
        # 计算衍生指标
        price = record['Price']
        cw = record['Call_Wall']
        pw = record['Put_Wall']
        
        if price and cw and price > 0:
            record['Dist_CW_Pct'] = round((cw - price) / price * 100, 2)
        else:
            record['Dist_CW_Pct'] = None
            
        if price and pw and price > 0:
            record['Dist_PW_Pct'] = round((price - pw) / price * 100, 2)
        else:
            record['Dist_PW_Pct'] = None
        
        result.append(record)
    
    result_df = pd.DataFrame(result)
    
    # 检查缺失
    found_symbols = result_df['Symbol'].tolist()
    missing = [s for s in watchlist if s not in found_symbols]
    
    return result_df, missing

# ============================================================
# 信号生成
# ============================================================

def generate_signals(today_df, history_df=None):
    """
    生成交易信号
    
    参数:
        today_df: 今日数据
        history_df: 历史数据（用于计算趋势）
    
    返回:
        signals: 信号列表
    """
    signals = []
    
    for _, row in today_df.iterrows():
        symbol = row['Symbol']
        sector = row['Sector']
        price = row['Price']
        
        # 提取指标
        dpi = row.get('DPI_Pct')
        dpi_5d = row.get('DPI_5d_Pct')
        delta_ratio = row.get('Delta_Ratio')
        gamma_ratio = row.get('Gamma_Ratio')
        volume_ratio = row.get('Volume_Ratio')
        dist_cw = row.get('Dist_CW_Pct')
        dist_pw = row.get('Dist_PW_Pct')
        next_exp_gamma = row.get('Next_Exp_Gamma')
        iv_rank = row.get('IV_Rank')
        options_impact = row.get('Options_Impact')
        
        stock_signals = []
        
        # === 做多信号 ===
        
        # 1. 机构建仓
        if dpi and dpi > 55:
            if dpi_5d and dpi_5d > 52:
                stock_signals.append({
                    'type': 'BULLISH',
                    'name': '机构持续买入',
                    'reason': f'DPI {dpi:.1f}%, 5日DPI {dpi_5d:.1f}%',
                    'style': 'Swing'
                })
            else:
                stock_signals.append({
                    'type': 'BULLISH',
                    'name': '机构当日买入',
                    'reason': f'DPI {dpi:.1f}%',
                    'style': 'Day/Swing'
                })
        
        # 2. 支撑位抄底
        if dist_pw is not None and dist_pw < 3 and dist_pw > 0:
            if dpi and dpi > 50:
                stock_signals.append({
                    'type': 'BULLISH',
                    'name': '支撑位抄底',
                    'reason': f'距Put Wall {dist_pw:.1f}%, DPI {dpi:.1f}%',
                    'style': 'Day'
                })
        
        # 3. Gamma Squeeze预警
        if next_exp_gamma and next_exp_gamma > 25:
            if gamma_ratio and gamma_ratio < 0.8:
                if dist_cw is not None and dist_cw < 5 and dist_cw > 0:
                    stock_signals.append({
                        'type': 'SQUEEZE',
                        'name': 'Gamma Squeeze预警',
                        'reason': f'NEG {next_exp_gamma:.1f}%, Gamma Ratio {gamma_ratio:.2f}, 距CW {dist_cw:.1f}%',
                        'style': 'Day'
                    })
        
        # 4. 超卖反弹
        if delta_ratio and delta_ratio < -5:
            if dpi and dpi > 48:
                stock_signals.append({
                    'type': 'BULLISH',
                    'name': '超卖反弹',
                    'reason': f'Delta Ratio {delta_ratio:.2f} 极度偏空, DPI {dpi:.1f}%',
                    'style': 'Swing'
                })
        
        # === 做空信号 ===
        
        # 5. 机构出货
        if dpi and dpi < 45:
            if dpi_5d and dpi_5d < 48:
                stock_signals.append({
                    'type': 'BEARISH',
                    'name': '机构持续卖出',
                    'reason': f'DPI {dpi:.1f}%, 5日DPI {dpi_5d:.1f}%',
                    'style': 'Swing'
                })
        
        # 6. 阻力位压制
        if dist_cw is not None and dist_cw < 3 and dist_cw > 0:
            if dpi and dpi < 50:
                if volume_ratio and volume_ratio > 1.2:
                    stock_signals.append({
                        'type': 'BEARISH',
                        'name': '阻力位压制',
                        'reason': f'距Call Wall {dist_cw:.1f}%, DPI {dpi:.1f}%, Vol Ratio {volume_ratio:.2f}',
                        'style': 'Day'
                    })
        
        # === 特殊信号 ===
        
        # 7. IV Crush机会
        if iv_rank and iv_rank > 80:
            stock_signals.append({
                'type': 'IV_HIGH',
                'name': 'IV高位',
                'reason': f'IV Rank {iv_rank:.1f}%',
                'style': '卖方策略'
            })
        
        # 8. 大波动预警
        if next_exp_gamma and next_exp_gamma > 30:
            implied_move = row.get('Implied_Move')
            if implied_move and implied_move > 5:
                stock_signals.append({
                    'type': 'VOLATILE',
                    'name': '大波动预警',
                    'reason': f'NEG {next_exp_gamma:.1f}%, Implied Move ${implied_move:.2f}',
                    'style': '跨式策略'
                })
        
        # 添加到总信号列表
        for sig in stock_signals:
            sig['symbol'] = symbol
            sig['sector'] = sector
            sig['price'] = price
            sig['options_impact'] = options_impact
            signals.append(sig)
    
    return signals

# ============================================================
# 汇总报告
# ============================================================

def generate_daily_report(today_df, signals):
    """
    生成每日报告
    """
    report = {
        'date': today_df['Date'].iloc[0] if not today_df.empty else datetime.now().strftime('%Y-%m-%d'),
        'total_stocks': len(today_df),
        'signals_count': len(signals),
        
        # 按类型分类
        'bullish_signals': [s for s in signals if s['type'] == 'BULLISH'],
        'bearish_signals': [s for s in signals if s['type'] == 'BEARISH'],
        'squeeze_signals': [s for s in signals if s['type'] == 'SQUEEZE'],
        'iv_signals': [s for s in signals if s['type'] == 'IV_HIGH'],
        'volatile_signals': [s for s in signals if s['type'] == 'VOLATILE'],
        
        # 统计
        'sector_summary': today_df.groupby('Sector').agg({
            'DPI_Pct': 'mean',
            'Delta_Ratio': 'mean',
            'Volume_Ratio': 'mean'
        }).round(2).to_dict(),
    }
    
    return report

def format_report_text(report):
    """
    格式化报告为文本
    """
    lines = []
    lines.append(f"=" * 60)
    lines.append(f"📊 100只期权追踪日报 - {report['date']}")
    lines.append(f"=" * 60)
    lines.append(f"追踪股票: {report['total_stocks']} 只 | 信号数: {report['signals_count']}")
    lines.append("")
    
    # 做多信号
    if report['bullish_signals']:
        lines.append("🚀 【做多信号】")
        for s in report['bullish_signals'][:10]:
            oi_str = f" [OI:{s['options_impact']:.1f}%]" if s['options_impact'] else ""
            lines.append(f"  • {s['symbol']} ({s['sector']}): {s['name']} - {s['reason']} [{s['style']}]{oi_str}")
        lines.append("")
    
    # 做空信号
    if report['bearish_signals']:
        lines.append("💀 【做空信号】")
        for s in report['bearish_signals'][:10]:
            oi_str = f" [OI:{s['options_impact']:.1f}%]" if s['options_impact'] else ""
            lines.append(f"  • {s['symbol']} ({s['sector']}): {s['name']} - {s['reason']} [{s['style']}]{oi_str}")
        lines.append("")
    
    # Squeeze信号
    if report['squeeze_signals']:
        lines.append("⚡ 【Gamma Squeeze预警】")
        for s in report['squeeze_signals'][:5]:
            lines.append(f"  • {s['symbol']} ({s['sector']}): {s['reason']}")
        lines.append("")
    
    # IV高位
    if report['iv_signals']:
        lines.append("📈 【IV高位 - 卖方机会】")
        for s in report['iv_signals'][:5]:
            lines.append(f"  • {s['symbol']}: {s['reason']}")
        lines.append("")
    
    # 大波动
    if report['volatile_signals']:
        lines.append("🌊 【大波动预警】")
        for s in report['volatile_signals'][:5]:
            lines.append(f"  • {s['symbol']}: {s['reason']}")
        lines.append("")
    
    return "\n".join(lines)

# ============================================================
# 测试
# ============================================================

if __name__ == '__main__':
    # 测试数据提取
    import os
    
    test_file = '/mnt/user-data/uploads/Top200_2026-02-03.csv'
    if os.path.exists(test_file):
        df = pd.read_csv(test_file)
        
        # 提取数据
        result_df, missing = extract_watchlist_data(df, '2026-02-03')
        print(f"提取到 {len(result_df)} 只股票")
        print(f"缺失 {len(missing)} 只: {missing[:10]}...")
        
        # 生成信号
        signals = generate_signals(result_df)
        print(f"\n生成 {len(signals)} 个信号")
        
        # 生成报告
        report = generate_daily_report(result_df, signals)
        print(format_report_text(report))
    else:
        print("测试文件不存在")
