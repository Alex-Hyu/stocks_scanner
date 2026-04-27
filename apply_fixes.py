"""
一键修复脚本 - 自动应用三处修改到 app.py

用法：
    python apply_fixes.py app.py

或者直接双击运行（如果脚本和 app.py 在同一目录）：
    python apply_fixes.py

效果：
    1. 修复第 9 行的 import 语法错误
    2. 删除第 1188 行附近的冗余 import
    3. 把三因素信号显示位置移到 CSV 上传器之后，并添加 seek(0)

会自动备份原文件为 app.py.backup_YYYYMMDD_HHMMSS
"""

import sys
import os
import shutil
from datetime import datetime


def main():
    # 找到 app.py 路径
    if len(sys.argv) >= 2:
        app_path = sys.argv[1]
    else:
        app_path = "app.py"
    
    if not os.path.exists(app_path):
        print(f"❌ 找不到文件: {app_path}")
        print(f"   用法: python apply_fixes.py [app.py 路径]")
        return 1
    
    # 备份
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"{app_path}.backup_{timestamp}"
    shutil.copy2(app_path, backup_path)
    print(f"✅ 已备份原文件: {backup_path}")
    
    # 读取
    with open(app_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    fixes_applied = []
    
    # ============================================================
    # 修复 1: 第 9 行的 import 语法错误
    # ============================================================
    bad_import = "import warnings\nimport from three_factor_signal import render_three_factor_signal\nwarnings.filterwarnings('ignore')"
    good_import = "import warnings\nwarnings.filterwarnings('ignore')\n\nfrom three_factor_signal import render_three_factor_signal"
    
    if bad_import in content:
        content = content.replace(bad_import, good_import)
        fixes_applied.append("✅ 修复 1: 顶部的 import 语法错误已修复")
    else:
        # 已经被修过了？检查是否已经存在好的版本
        if "from three_factor_signal import render_three_factor_signal" in content:
            fixes_applied.append("⏭  修复 1: 已经修过了（跳过）")
        else:
            fixes_applied.append("⚠️  修复 1: 没找到目标代码，可能已经被修改过了")
    
    # ============================================================
    # 修复 2: 删除函数定义之间的冗余 import
    # ============================================================
    redundant_block = """# ============================================================
# QQQ/NQ 盘前分析函数
# ============================================================
from three_factor_signal import render_three_factor_signal

        
def parse_qqq_premarket_text(text):"""
    
    clean_block = """# ============================================================
# QQQ/NQ 盘前分析函数
# ============================================================

def parse_qqq_premarket_text(text):"""
    
    if redundant_block in content:
        content = content.replace(redundant_block, clean_block)
        fixes_applied.append("✅ 修复 2: 已删除冗余的 import")
    else:
        fixes_applied.append("⏭  修复 2: 没找到冗余 import（可能已经删了）")
    
    # ============================================================
    # 修复 3: 把三因素信号移到 CSV 上传器之后，并加 seek(0)
    # ============================================================
    old_tab1_start = """    # ========== Tab 1: QQQ/NQ盘前分析 ==========
    with tab1:
        st.header("📈 QQQ/NQ 盘前分析")
        render_three_factor_signal(st)
        
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
            st.info("💡 上传CSV可获得详细的方向性分析和情景分析")"""
    
    new_tab1_start = """    # ========== Tab 1: QQQ/NQ盘前分析 ==========
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
        
        # ============================================================
        # 🎯 三因素早盘方向信号（复用上方上传的 CSV）
        # ============================================================
        if qqq_csv_file is not None:
            st.markdown("---")
            render_three_factor_signal(st, csv_source=qqq_csv_file)
            qqq_csv_file.seek(0)  # 关键：重置文件指针，让下方代码还能读取
            st.markdown("---")
        # ============================================================"""
    
    if old_tab1_start in content:
        content = content.replace(old_tab1_start, new_tab1_start)
        fixes_applied.append("✅ 修复 3: 三因素信号位置已调整 + 添加 seek(0)")
    else:
        fixes_applied.append("⚠️  修复 3: 没找到 Tab 1 起始代码（可能已经被修改过）")
    
    # ============================================================
    # 写回文件
    # ============================================================
    if content == original_content:
        print("\n⚠️  没有任何修改被应用 — 文件可能已经修过了\n")
        # 删除备份（因为没改）
        os.remove(backup_path)
        print(f"   已删除多余的备份文件")
    else:
        with open(app_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\n✅ 文件已更新: {app_path}\n")
    
    print("=" * 60)
    print("修改详情:")
    print("=" * 60)
    for f in fixes_applied:
        print(f"  {f}")
    print("=" * 60)
    print(f"\n📁 备份位置: {backup_path}")
    print("\n下一步:")
    print("  1. 确认 three_factor_signal.py 也在同一目录")
    print("  2. git add app.py three_factor_signal.py")
    print("  3. git commit -m 'Add three-factor signal module'")
    print("  4. git push")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
