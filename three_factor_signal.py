"""
========================================================================
  Three-Factor Pre-Market Signal Module (三因素早盘方向信号)
  
  自动读取 QQQ 手动数据 CSV，取最新两行 (t-1, t-2) 计算信号
  
  逻辑来源：33 样本回测 (2026-01-15 ~ 2026-03-06)
  - 三因素同降 → t 日 09:30-10:25 上涨概率 80% (N=10, 平均 +0.13%)
  - 三因素同升 → t 日 09:30-10:25 下跌概率 67% (N=9, 平均 -0.17%)
  - 时段：t 日 09:30 开盘价 → 10:25 收盘 (60min)
  - 入场建议：09:40 之后（避开开盘混乱）
========================================================================
"""

from dataclasses import dataclass
from typing import Optional, Union
import pandas as pd
import numpy as np


# ==================== 1. 数据读取与清洗 ====================

def _pct_to_float(x):
    """把百分比字符串转换为数值（'12.94%' -> 12.94, '$7.27 ' -> 7.27）"""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace('%', '').replace('$', '').replace(',', '')
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan


def load_spotgamma_csv(source) -> pd.DataFrame:
    """
    读取 QQQ 手动数据 CSV 并清洗。
    
    参数：
        source: 文件路径 (str) 或 Streamlit UploadedFile 对象 或 DataFrame
    
    返回：
        清洗后的 DataFrame，按日期升序排列。
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.read_csv(source)
    
    # 列名标准化：替换不间断空格 \xa0，去除首尾空白
    df.columns = df.columns.str.replace('\xa0', ' ', regex=False).str.strip()
    
    # 日期解析（兼容 2026/1/15 和 2026-01-15 两种格式）
    df['Date'] = pd.to_datetime(df['Date'].astype(str).str.strip(), errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    
    # 关键字段清洗
    pct_cols = ['NE Skew', 'Skew', '1 M RV', '1 M IV', 'IV Rank',
                'Garch Rank', 'Skew Rank', 'Options Implied Move',
                'Next Exp Gamma', 'Next Exp Delta',
                'DPI', '%DPI Volume', '5Day DPI', '5D% DPI Volume']
    for c in pct_cols:
        if c in df.columns:
            df[c] = df[c].apply(_pct_to_float)
    
    num_cols = ['Volume Ratio', 'Gamma Ratio', 'Delta Ratio', 'Put/Call OI Ratio',
                'previous close', 'Current Price(盘前价)',
                'Key Gamma Strike', 'Key Delta Strike',
                'Hedge Wall', 'Call Wall', 'Put Wall']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    
    return df


# ==================== 2. 信号数据结构 ====================

@dataclass
class ThreeFactorSignal:
    """三因素早盘信号结果"""
    signal: str
    direction_zh: str
    confidence: str
    
    date_t1: pd.Timestamp
    date_t2: pd.Timestamp
    ne_skew_t1: float
    ne_skew_t2: float
    iv_rank_t1: float
    iv_rank_t2: float
    delta_ratio_t1: float
    delta_ratio_t2: float
    
    d_ne_skew: float
    d_iv_rank: float
    d_delta_ratio: float
    
    bearish_count: int
    bullish_count: int
    
    hist_win_rate: float
    hist_avg_return: float
    hist_sample_n: int
    
    entry_window: str
    target_window: str
    note: str


# ==================== 3. 核心信号计算 ====================

def compute_three_factor_signal_from_df(df: pd.DataFrame) -> ThreeFactorSignal:
    """
    从清洗过的 SpotGamma DataFrame 中自动取最新两行计算信号。
    最后一行 = t-1（昨日）；倒数第二行 = t-2（前日）。
    """
    if len(df) < 2:
        raise ValueError(f"数据至少需要 2 行才能计算变化量，当前只有 {len(df)} 行")
    
    required = ['NE Skew', 'IV Rank', 'Delta Ratio']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必需字段: {missing}")
    
    t2 = df.iloc[-2]
    t1 = df.iloc[-1]
    
    for col in required:
        if pd.isna(t1[col]) or pd.isna(t2[col]):
            raise ValueError(
                f"最新两行的 {col} 字段含空值（t-1={t1[col]}, t-2={t2[col]}），"
                f"请检查 CSV 数据完整性"
            )
    
    return _build_signal(
        date_t1=t1['Date'], date_t2=t2['Date'],
        ne_skew_t1=float(t1['NE Skew']), ne_skew_t2=float(t2['NE Skew']),
        iv_rank_t1=float(t1['IV Rank']), iv_rank_t2=float(t2['IV Rank']),
        delta_ratio_t1=float(t1['Delta Ratio']), delta_ratio_t2=float(t2['Delta Ratio']),
    )


def _build_signal(
    date_t1, date_t2,
    ne_skew_t1, ne_skew_t2,
    iv_rank_t1, iv_rank_t2,
    delta_ratio_t1, delta_ratio_t2,
) -> ThreeFactorSignal:
    """根据 t-1/t-2 数值构建信号对象"""
    d_ne = ne_skew_t1 - ne_skew_t2
    d_ivr = iv_rank_t1 - iv_rank_t2
    d_dr = delta_ratio_t1 - delta_ratio_t2
    
    bullish_count = sum([d_ne < 0, d_ivr < 0, d_dr < 0])
    bearish_count = sum([d_ne > 0, d_ivr > 0, d_dr > 0])
    
    if bullish_count == 3:
        signal = "LONG"
        direction_zh = "🟢 看多 (三因素同降)"
        confidence = "HIGH"
        hist_win_rate, hist_avg_return, hist_sample_n = 80.0, 0.13, 10
        note = (
            "t-1 收盘后 NE Skew、IV Rank、Delta Ratio 三者同时下降，"
            "对应『恐慌指标全面退潮』模式 → t 日早盘倾向反弹。"
            "⚠️ 若宏观状态为『恶化中/高危震荡』则此信号失效。"
        )
    elif bearish_count == 3:
        signal = "SHORT"
        direction_zh = "🔴 看空 (三因素同升)"
        confidence = "HIGH"
        hist_win_rate, hist_avg_return, hist_sample_n = 66.7, -0.17, 9
        note = (
            "t-1 收盘后 NE Skew、IV Rank、Delta Ratio 三者同时上升，"
            "对应『风险溢价全面抬升』模式 → t 日早盘倾向走弱。"
        )
    elif bullish_count == 2:
        signal = "LONG_WEAK"
        direction_zh = "🟡 弱看多 (2/3 同降)"
        confidence = "LOW"
        hist_win_rate, hist_avg_return, hist_sample_n = 0.0, 0.0, 0
        note = "三因素未同时满足，信号强度不足，建议观望或等待开盘后 30 分钟方向确认。"
    elif bearish_count == 2:
        signal = "SHORT_WEAK"
        direction_zh = "🟡 弱看空 (2/3 同升)"
        confidence = "LOW"
        hist_win_rate, hist_avg_return, hist_sample_n = 0.0, 0.0, 0
        note = "三因素未同时满足，信号强度不足，建议观望或等待开盘后 30 分钟方向确认。"
    else:
        signal = "NEUTRAL"
        direction_zh = "⚪ 中性 (无明确信号)"
        confidence = "LOW"
        hist_win_rate, hist_avg_return, hist_sample_n = 57.6, 0.05, 33
        note = "三因素方向分歧，无明确预期。早盘策略以 SpotGamma 关键位 (PW/CW/ZG) 为主导。"
    
    return ThreeFactorSignal(
        signal=signal,
        direction_zh=direction_zh,
        confidence=confidence,
        date_t1=date_t1, date_t2=date_t2,
        ne_skew_t1=ne_skew_t1, ne_skew_t2=ne_skew_t2,
        iv_rank_t1=iv_rank_t1, iv_rank_t2=iv_rank_t2,
        delta_ratio_t1=delta_ratio_t1, delta_ratio_t2=delta_ratio_t2,
        d_ne_skew=d_ne, d_iv_rank=d_ivr, d_delta_ratio=d_dr,
        bearish_count=bearish_count, bullish_count=bullish_count,
        hist_win_rate=hist_win_rate,
        hist_avg_return=hist_avg_return,
        hist_sample_n=hist_sample_n,
        entry_window="t 日 09:40 之后（避开开盘混乱）",
        target_window="t 日 09:30 → 10:25 (60min 早盘窗口)",
        note=note,
    )


# ==================== 4. Streamlit 显示模块 ====================

def render_three_factor_signal(
    st,
    csv_source=None,
    container=None,
):
    """
    在 Streamlit 中渲染三因素早盘信号模块。
    
    用法：
    
        from three_factor_signal import render_three_factor_signal
        
        # 方式 A：让模块自己显示上传按钮
        render_three_factor_signal(st)
        
        # 方式 B：传入已经上传好的文件（推荐 — 复用主页面的上传器）
        uploaded = st.file_uploader("上传 QQQ 手动数据", type='csv')
        if uploaded:
            render_three_factor_signal(st, csv_source=uploaded)
        
        # 方式 C：传入固定路径
        render_three_factor_signal(st, csv_source="data/QQQ数据手动.csv")
    
    参数：
        st: streamlit 模块对象
        csv_source: CSV 路径 / UploadedFile / DataFrame，None 时显示内置上传器
        container: 可选的 st.container() 或 st.expander()
    """
    target = container if container is not None else st
    
    target.markdown("### 🎯 三因素早盘方向信号")
    target.caption("回测样本 N=33 (2026-01-15 ~ 2026-03-06)　|　目标窗口：t 日 09:30 → 10:25")
    
    # ---------- 数据来源 ----------
    if csv_source is None:
        csv_source = target.file_uploader(
            "上传 QQQ 手动数据 CSV (取最新两行作为 t-1 / t-2)",
            type='csv',
            key="three_factor_csv"
        )
        if csv_source is None:
            target.info("👆 请上传 QQQ 手动数据 CSV 文件以查看信号")
            return None
    
    # ---------- 加载数据 ----------
    try:
        df = load_spotgamma_csv(csv_source)
    except Exception as e:
        target.error(f"❌ CSV 读取失败: {e}")
        return None
    
    if len(df) < 2:
        target.warning(f"⚠️ CSV 中只有 {len(df)} 行数据，至少需要 2 行才能计算变化量")
        return None
    
    # ---------- 计算信号 ----------
    try:
        sig = compute_three_factor_signal_from_df(df)
    except ValueError as e:
        target.error(f"❌ 信号计算失败: {e}")
        return None
    
    # ---------- 数据来源提示 ----------
    target.caption(
        f"📅 数据来源：t-1 = **{sig.date_t1.strftime('%Y-%m-%d')}** "
        f"| t-2 = **{sig.date_t2.strftime('%Y-%m-%d')}** "
        f"（CSV 共 {len(df)} 行，自动取最新两行）"
    )
    
    # ---------- 信号显示 ----------
    if sig.signal == "LONG":
        target.success(f"### {sig.direction_zh}")
    elif sig.signal == "SHORT":
        target.error(f"### {sig.direction_zh}")
    elif sig.signal in ("LONG_WEAK", "SHORT_WEAK"):
        target.warning(f"### {sig.direction_zh}")
    else:
        target.info(f"### {sig.direction_zh}")
    
    # ---------- 三因素当前值 + 变化量 ----------
    target.markdown("**三因素数值与变化量**")
    
    factor_data = pd.DataFrame({
        '指标': ['NE Skew (%)', 'IV Rank (%)', 'Delta Ratio'],
        f't-2 ({sig.date_t2.strftime("%m-%d")})': [
            f"{sig.ne_skew_t2:.2f}",
            f"{sig.iv_rank_t2:.2f}",
            f"{sig.delta_ratio_t2:.2f}",
        ],
        f't-1 ({sig.date_t1.strftime("%m-%d")})': [
            f"{sig.ne_skew_t1:.2f}",
            f"{sig.iv_rank_t1:.2f}",
            f"{sig.delta_ratio_t1:.2f}",
        ],
        '变化量 Δ': [
            f"{sig.d_ne_skew:+.2f}",
            f"{sig.d_iv_rank:+.2f}",
            f"{sig.d_delta_ratio:+.2f}",
        ],
        '方向': [
            "🔻 降" if sig.d_ne_skew < 0 else ("🔺 升" if sig.d_ne_skew > 0 else "⏸ 平"),
            "🔻 降" if sig.d_iv_rank < 0 else ("🔺 升" if sig.d_iv_rank > 0 else "⏸ 平"),
            "🔻 降" if sig.d_delta_ratio < 0 else ("🔺 升" if sig.d_delta_ratio > 0 else "⏸ 平"),
        ],
    })
    target.dataframe(factor_data, hide_index=True, use_container_width=True)
    
    # ---------- 历史参考（仅高置信度时显示）----------
    if sig.confidence == "HIGH":
        target.markdown("**历史回测参考**")
        m1, m2, m3 = target.columns(3)
        m1.metric("历史胜率", f"{sig.hist_win_rate:.1f}%")
        m2.metric("平均收益", f"{sig.hist_avg_return:+.2f}%")
        m3.metric("样本量", f"N = {sig.hist_sample_n}")
    
    # ---------- 操作指引 ----------
    target.markdown("**操作指引**")
    target.markdown(f"- 入场时段：{sig.entry_window}")
    target.markdown(f"- 目标窗口：{sig.target_window}")
    target.markdown(f"- {sig.note}")
    
    return sig


# ==================== 5. 命令行测试入口 ====================

if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'QQQ数据手动.csv'
    
    df = load_spotgamma_csv(csv_path)
    print(f"✅ 加载 {len(df)} 行数据")
    print(f"   日期范围：{df['Date'].min().date()} ~ {df['Date'].max().date()}")
    
    sig = compute_three_factor_signal_from_df(df)
    print(f"\n📅 t-1 = {sig.date_t1.date()}, t-2 = {sig.date_t2.date()}")
    print(f"\n🎯 信号: {sig.direction_zh}")
    print(f"   置信度: {sig.confidence}")
    print(f"\n   NE Skew    : {sig.ne_skew_t2:+.2f} → {sig.ne_skew_t1:+.2f}  Δ={sig.d_ne_skew:+.2f}")
    print(f"   IV Rank    : {sig.iv_rank_t2:+.2f} → {sig.iv_rank_t1:+.2f}  Δ={sig.d_iv_rank:+.2f}")
    print(f"   Delta Ratio: {sig.delta_ratio_t2:+.2f} → {sig.delta_ratio_t1:+.2f}  Δ={sig.d_delta_ratio:+.2f}")
    
    if sig.confidence == "HIGH":
        print(f"\n   历史胜率: {sig.hist_win_rate}%")
        print(f"   平均收益: {sig.hist_avg_return:+.2f}%")
        print(f"   样本量  : N = {sig.hist_sample_n}")
    
    print(f"\n💡 {sig.note}")
