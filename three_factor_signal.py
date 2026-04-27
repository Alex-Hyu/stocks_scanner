"""
========================================================================
  Three-Factor Pre-Market Signal Module (三因素早盘方向信号)
  
  逻辑来源：33 样本回测 (2026-01-15 ~ 2026-03-06)
  - 三因素同降 → t 日 09:30-10:25 上涨概率 80% (N=10, 平均 +0.13%)
  - 三因素同升 → t 日 09:30-10:25 下跌概率 67% (N=9, 平均 -0.17%)
  - 时段：t 日 09:30 开盘价 → 10:25 收盘 (60min)
  - 入场建议：09:40 之后（避开开盘混乱）
  
  使用方法：
  1. 把 t-1 和 t-2 两天的 SpotGamma 数据传入
  2. 函数返回信号、变化量、统计参考
  3. Streamlit 部分直接调用 render_three_factor_signal()
========================================================================
"""

from dataclasses import dataclass
from typing import Optional, Dict


# ==================== 1. 核心信号计算（纯函数，无依赖）====================

@dataclass
class ThreeFactorSignal:
    """三因素早盘信号结果"""
    signal: str              # "LONG" / "SHORT" / "NEUTRAL"
    direction_zh: str        # 中文方向
    confidence: str          # "HIGH" / "MEDIUM" / "LOW"
    
    # 三因素变化量
    d_ne_skew: float
    d_iv_rank: float
    d_delta_ratio: float
    
    # 命中条件数（0-3）
    bearish_count: int       # 几个因素是上升（看跌方向）
    bullish_count: int       # 几个因素是下降（看多方向）
    
    # 历史统计参考
    hist_win_rate: float     # 该信号历史胜率
    hist_avg_return: float   # 该信号历史平均收益
    hist_sample_n: int       # 历史样本量
    
    # 操作指引
    entry_window: str
    target_window: str
    note: str


def compute_three_factor_signal(
    ne_skew_t1: float, ne_skew_t2: float,
    iv_rank_t1: float, iv_rank_t2: float,
    delta_ratio_t1: float, delta_ratio_t2: float,
) -> ThreeFactorSignal:
    """
    计算三因素早盘方向信号。
    
    参数：
        ne_skew_t1, ne_skew_t2:           t-1 和 t-2 的 NE Skew (% 数值，例如 -24.39)
        iv_rank_t1, iv_rank_t2:           t-1 和 t-2 的 IV Rank (% 数值，例如 11.14)
        delta_ratio_t1, delta_ratio_t2:   t-1 和 t-2 的 Delta Ratio (例如 -1.13)
    
    返回：
        ThreeFactorSignal 对象
    
    回测样本（2026-01-15 ~ 2026-03-06, N=33）：
        三因素同降 → 60min 上涨概率 80% (N=10, 平均 +0.13%)
        三因素同升 → 60min 下跌概率 67% (N=9, 平均 -0.17%)
    """
    # 计算变化量
    d_ne = ne_skew_t1 - ne_skew_t2
    d_ivr = iv_rank_t1 - iv_rank_t2
    d_dr = delta_ratio_t1 - delta_ratio_t2
    
    # 严格按回测：<0 / >0
    bullish_count = sum([d_ne < 0, d_ivr < 0, d_dr < 0])
    bearish_count = sum([d_ne > 0, d_ivr > 0, d_dr > 0])
    
    # 信号判定
    if bullish_count == 3:
        signal = "LONG"
        direction_zh = "🟢 看多 (三因素同降)"
        confidence = "HIGH"
        hist_win_rate = 80.0
        hist_avg_return = 0.13
        hist_sample_n = 10
        note = (
            "t-1 收盘后 NE Skew、IV Rank、Delta Ratio 三者同时下降，"
            "对应『恐慌指标全面退潮』模式 → t 日早盘倾向反弹。"
            "⚠️ 若宏观状态为『恶化中/高危震荡』则此信号失效。"
        )
    elif bearish_count == 3:
        signal = "SHORT"
        direction_zh = "🔴 看空 (三因素同升)"
        confidence = "HIGH"
        hist_win_rate = 66.7
        hist_avg_return = -0.17
        hist_sample_n = 9
        note = (
            "t-1 收盘后 NE Skew、IV Rank、Delta Ratio 三者同时上升，"
            "对应『风险溢价全面抬升』模式 → t 日早盘倾向走弱。"
        )
    elif bullish_count == 2:
        signal = "LONG_WEAK"
        direction_zh = "🟡 弱看多 (2/3 同降)"
        confidence = "LOW"
        hist_win_rate = 0.0
        hist_avg_return = 0.0
        hist_sample_n = 0
        note = "三因素未同时满足，信号强度不足，建议观望或等待开盘后 30 分钟方向确认。"
    elif bearish_count == 2:
        signal = "SHORT_WEAK"
        direction_zh = "🟡 弱看空 (2/3 同升)"
        confidence = "LOW"
        hist_win_rate = 0.0
        hist_avg_return = 0.0
        hist_sample_n = 0
        note = "三因素未同时满足，信号强度不足，建议观望或等待开盘后 30 分钟方向确认。"
    else:
        signal = "NEUTRAL"
        direction_zh = "⚪ 中性 (无明确信号)"
        confidence = "LOW"
        hist_win_rate = 57.6  # 基线
        hist_avg_return = 0.05
        hist_sample_n = 33
        note = "三因素方向分歧，无明确预期。早盘策略以 SpotGamma 关键位 (PW/CW/ZG) 为主导。"
    
    return ThreeFactorSignal(
        signal=signal,
        direction_zh=direction_zh,
        confidence=confidence,
        d_ne_skew=d_ne,
        d_iv_rank=d_ivr,
        d_delta_ratio=d_dr,
        bearish_count=bearish_count,
        bullish_count=bullish_count,
        hist_win_rate=hist_win_rate,
        hist_avg_return=hist_avg_return,
        hist_sample_n=hist_sample_n,
        entry_window="t 日 09:40 之后（避开开盘混乱）",
        target_window="t 日 09:30 → 10:25 (60min 早盘窗口)",
        note=note,
    )


# ==================== 2. Streamlit 显示模块 ====================

def render_three_factor_signal(st, container=None):
    """
    在 Streamlit 中渲染三因素早盘信号模块。
    
    用法（在你的盘前分析 tab 里）：
        from three_factor_signal import render_three_factor_signal
        render_three_factor_signal(st)
    
    参数：
        st: streamlit 模块对象（直接传 st 进来）
        container: 可选的 st.container() 或 st.expander()，不传则使用主区域
    """
    target = container if container is not None else st
    
    target.markdown("### 🎯 三因素早盘方向信号")
    target.caption("回测样本 N=33 (2026-01-15 ~ 2026-03-06)　|　目标窗口：t 日 09:30 → 10:25")
    
    # ---------- 输入区 ----------
    col1, col2 = target.columns(2)
    
    with col1:
        target.markdown("**t-1 数据 (昨日收盘)**")
        ne_t1 = st.number_input(
            "NE Skew (%) — t-1",
            value=-24.39, step=0.1, format="%.2f", key="ne_t1",
            help="例如 -24.39 输入 -24.39（不要输入 -0.2439）"
        )
        ivr_t1 = st.number_input(
            "IV Rank (%) — t-1",
            value=11.14, step=0.1, format="%.2f", key="ivr_t1"
        )
        dr_t1 = st.number_input(
            "Delta Ratio — t-1",
            value=-1.13, step=0.01, format="%.2f", key="dr_t1",
            help="负值代表 Put Delta 占优，例如 -1.13"
        )
    
    with col2:
        target.markdown("**t-2 数据 (前日收盘)**")
        ne_t2 = st.number_input(
            "NE Skew (%) — t-2",
            value=-12.63, step=0.1, format="%.2f", key="ne_t2"
        )
        ivr_t2 = st.number_input(
            "IV Rank (%) — t-2",
            value=8.57, step=0.1, format="%.2f", key="ivr_t2"
        )
        dr_t2 = st.number_input(
            "Delta Ratio — t-2",
            value=-1.24, step=0.01, format="%.2f", key="dr_t2"
        )
    
    # ---------- 信号计算 ----------
    sig = compute_three_factor_signal(
        ne_skew_t1=ne_t1, ne_skew_t2=ne_t2,
        iv_rank_t1=ivr_t1, iv_rank_t2=ivr_t2,
        delta_ratio_t1=dr_t1, delta_ratio_t2=dr_t2,
    )
    
    target.markdown("---")
    
    # ---------- 信号显示 ----------
    if sig.signal == "LONG":
        target.success(f"### {sig.direction_zh}")
    elif sig.signal == "SHORT":
        target.error(f"### {sig.direction_zh}")
    elif sig.signal in ("LONG_WEAK", "SHORT_WEAK"):
        target.warning(f"### {sig.direction_zh}")
    else:
        target.info(f"### {sig.direction_zh}")
    
    # ---------- 三因素变化量明细 ----------
    target.markdown("**三因素变化量 (t-1 − t-2)**")
    c1, c2, c3 = target.columns(3)
    
    def fmt_delta(value, label):
        arrow = "🔻" if value < 0 else ("🔺" if value > 0 else "⏸")
        sign = "+" if value > 0 else ""
        return f"{arrow} {label}\n\n**{sign}{value:.2f}**"
    
    c1.markdown(fmt_delta(sig.d_ne_skew, "ΔNE Skew"))
    c2.markdown(fmt_delta(sig.d_iv_rank, "ΔIV Rank"))
    c3.markdown(fmt_delta(sig.d_delta_ratio, "ΔDelta Ratio"))
    
    # ---------- 历史参考 ----------
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


# ==================== 3. 测试入口 ====================

if __name__ == "__main__":
    # 命令行快速测试（不启动 Streamlit）
    sig = compute_three_factor_signal(
        ne_skew_t1=-24.70, ne_skew_t2=-12.63,
        iv_rank_t1=11.14, iv_rank_t2=8.57,
        delta_ratio_t1=-1.24, delta_ratio_t2=-1.13,
    )
    print(f"信号: {sig.direction_zh}")
    print(f"置信度: {sig.confidence}")
    print(f"ΔNE Skew = {sig.d_ne_skew:+.2f}")
    print(f"ΔIV Rank = {sig.d_iv_rank:+.2f}")
    print(f"ΔDelta Ratio = {sig.d_delta_ratio:+.2f}")
    print(f"历史胜率: {sig.hist_win_rate}% (N={sig.hist_sample_n})")
    print(f"操作: {sig.note}")
