# -*- coding: utf-8 -*-
"""
build_docx.py —— 将 Markdown 论文正文转换为规范排版的 Word 文档
=================================================================
用法：
    python code/build_docx.py

输入：docs/论文正文.md
输出：docs/企业数字化转型的文本测度构建及其对全要素生产率的影响_毕业论文模拟稿.docx

排版规范：
    * 正文：宋体小四（12pt），1.5 倍行距，首行缩进 2 字符
    * 一级标题：黑体三号（16pt）居中
    * 二级标题：黑体四号（14pt）左对齐
    * 三级标题：黑体小四加粗
    * 表格：仿宋小五，三线表风格，表题居中加粗
    * 图形：居中，图题居中
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "论文正文.md"
OUT = ROOT / "docs" / "企业数字化转型的文本测度构建及其对全要素生产率的影响_毕业论文模拟稿.docx"

BODY_FONT = "宋体"
HEAD_FONT = "黑体"
EAST = qn("w:eastAsia")


# ------------------------------------------------------------------
# 基础工具
# ------------------------------------------------------------------
def set_run_font(run, name: str, size: float, bold: bool = False,
                 color: tuple[int, int, int] | None = None):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run._element.rPr.rFonts.set(EAST, name)
    if color:
        run.font.color.rgb = RGBColor(*color)


def add_para(doc, text="", *, font=BODY_FONT, size=12, bold=False,
             align=None, indent_chars=0, space_before=0, space_after=6,
             line_spacing=1.5, color=None):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = line_spacing
    if indent_chars:
        pf.first_line_indent = Pt(size * indent_chars)
    if text:
        for seg, is_bold in parse_bold(text):
            r = p.add_run(seg)
            set_run_font(r, font, size, bold or is_bold, color)
    return p


def parse_bold(text: str):
    """把 markdown 的 **加粗** 解析为 (片段, 是否加粗) 序列。"""
    out, pos = [], 0
    for m in re.finditer(r"\*\*(.+?)\*\*", text):
        if m.start() > pos:
            out.append((text[pos:m.start()], False))
        out.append((m.group(1), True))
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], False))
    return out or [(text, False)]


def set_cell_text(cell, text, *, bold=False, size=9, align=WD_ALIGN_PARAGRAPH.CENTER,
                  font="仿宋"):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.line_spacing = 1.0
    for seg, is_bold in parse_bold(text):
        r = p.add_run(seg)
        set_run_font(r, font, size, bold or is_bold)


def set_table_borders(table):
    """三线表：上下框线加粗，表头下加一条细线，其余无框线。"""
    tbl = table._tbl
    tblPr = tbl.tblPr
    for old in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(old)
    borders = OxmlElement("w:tblBorders")
    for tag, sz in (("top", 12), ("bottom", 12), ("left", 0), ("right", 0),
                    ("insideH", 0), ("insideV", 0)):
        el = OxmlElement(f"w:{tag}")
        if sz:
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), str(sz))
            el.set(qn("w:color"), "000000")
        else:
            el.set(qn("w:val"), "none")
            el.set(qn("w:sz"), "0")
        borders.append(el)
    tblPr.append(borders)

    # 表头行下加细线
    if len(table.rows) >= 1:
        for cell in table.rows[0].cells:
            tcPr = cell._tc.get_or_add_tcPr()
            tcB = OxmlElement("w:tcBorders")
            bot = OxmlElement("w:bottom")
            bot.set(qn("w:val"), "single")
            bot.set(qn("w:sz"), "6")
            bot.set(qn("w:color"), "000000")
            tcB.append(bot)
            tcPr.append(tcB)


def add_toc_field(doc):
    p = doc.add_paragraph()
    r = p.add_run()
    fld_begin = OxmlElement("w:fldChar"); fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve")
    instr.text = r'TOC \o "1-2" \h \z \u'
    fld_sep = OxmlElement("w:fldChar"); fld_sep.set(qn("w:fldCharType"), "separate")
    fld_end = OxmlElement("w:fldChar"); fld_end.set(qn("w:fldCharType"), "end")
    r._r.append(fld_begin); r._r.append(instr); r._r.append(fld_sep)
    tip = p.add_run("（在 Word / WPS 中右键选择“更新域”，或按 F9 键，即可自动生成带页码的目录）")
    set_run_font(tip, BODY_FONT, 10.5, False, (0x80, 0x80, 0x80))
    r2 = p.add_run()
    r2._r.append(fld_end)


def add_page_number_footer(section):
    footer = section.footer
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run()
    f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = "PAGE"
    f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "end")
    r._r.append(f1); r._r.append(it); r._r.append(f2)
    set_run_font(r, BODY_FONT, 10.5)


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def build():
    doc = Document()

    # 全局默认字体
    style = doc.styles["Normal"]
    style.font.name = BODY_FONT
    style.font.size = Pt(12)
    style.element.rPr.rFonts.set(EAST, BODY_FONT)

    # 页面设置：A4 + 常规页边距
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.top_margin = sec.bottom_margin = Cm(2.54)
    sec.left_margin = sec.right_margin = Cm(3.17)

    lines = SRC.read_text(encoding="utf-8").splitlines()

    # ---------- 封面 ----------
    add_para(doc, "", size=12, space_after=0)
    add_para(doc, "", size=12, space_after=0)
    add_para(doc, "", size=12, space_after=0)
    add_para(doc, "本科毕业论文（方法预演）", font=HEAD_FONT, size=16, bold=True,
             align=WD_ALIGN_PARAGRAPH.CENTER, space_after=24)
    add_para(doc, "", size=12, space_after=0)

    # 从 markdown 中提取主标题与副标题
    title = next((l[2:].strip() for l in lines if l.startswith("# ")), "毕业论文")
    subtitle = next((l.replace("**", "").strip("— ").strip()
                     for l in lines if l.strip().startswith("**——")), "")
    add_para(doc, title, font=HEAD_FONT, size=22, bold=True,
             align=WD_ALIGN_PARAGRAPH.CENTER, space_after=12, line_spacing=1.4)
    if subtitle:
        add_para(doc, "——" + subtitle, font=HEAD_FONT, size=13,
                 align=WD_ALIGN_PARAGRAPH.CENTER, space_after=36, line_spacing=1.4)

    for _ in range(4):
        add_para(doc, "", size=12, space_after=0)
    for line in lines:
        m = re.match(r"^(作者|学号|专业|指导教师|完成日期)[:：](.*)$", line.strip())
        if m:
            add_para(doc, f"{m.group(1)}：{'　' * 2}{m.group(2).strip()}",
                     size=13, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10)
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # ---------- 目录 ----------
    add_para(doc, "目　　录", font=HEAD_FONT, size=16, bold=True,
             align=WD_ALIGN_PARAGRAPH.CENTER, space_after=16)
    add_toc_field(doc)
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # ---------- 正文 ----------
    i = 0
    skip_cover = True
    first_h2 = True
    while i < len(lines):
        line = lines[i]
        s = line.strip()

        # 跳过封面区（直到第一次出现 "## "）
        if skip_cover:
            if s.startswith("## "):
                skip_cover = False
            else:
                i += 1
                continue

        # 水平线
        if s in ("---", "***", "___"):
            i += 1
            continue

        # 图片
        m = re.match(r"^!\[(.*?)\]\((.*?)\)\s*$", s)
        if m:
            cap, rel = m.group(1), m.group(2)
            img = (SRC.parent / rel).resolve()
            if img.exists():
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(8)
                p.paragraph_format.space_after = Pt(2)
                p.add_run().add_picture(str(img), width=Cm(14.5))
                add_para(doc, cap, size=10.5, align=WD_ALIGN_PARAGRAPH.CENTER,
                         space_after=12, line_spacing=1.2)
            i += 1
            continue

        # 表格
        if s.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            rows = [[c.strip() for c in r.strip("|").split("|")] for r in block]
            rows = [r for r in rows if not all(set(c) <= set("-: ") for c in r)]
            if rows:
                ncol = max(len(r) for r in rows)
                rows = [r + [""] * (ncol - len(r)) for r in rows]
                t = doc.add_table(rows=len(rows), cols=ncol)
                t.alignment = WD_TABLE_ALIGNMENT.CENTER
                for ri, row in enumerate(rows):
                    for ci, val in enumerate(row):
                        set_cell_text(t.cell(ri, ci), val, bold=(ri == 0))
                set_table_borders(t)
                doc.add_paragraph().paragraph_format.space_after = Pt(6)
            continue

        # 代码块
        if s.startswith("```"):
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            for cl in code:
                p = add_para(doc, cl if cl.strip() else " ",
                             font="Consolas", size=10,
                             space_after=0, line_spacing=1.15)
                p.paragraph_format.left_indent = Pt(24)
            continue

        # 一级标题
        if s.startswith("## "):
            txt = s[3:].strip()
            if txt.startswith("摘要"):
                doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
                first_h2 = False
            add_para(doc, txt, font=HEAD_FONT, size=16, bold=True,
                     align=WD_ALIGN_PARAGRAPH.CENTER, space_before=10,
                     space_after=14, line_spacing=1.5)
            i += 1
            continue

        # 二级标题
        if s.startswith("### "):
            add_para(doc, s[4:].strip(), font=HEAD_FONT, size=14, bold=True,
                     space_before=10, space_after=8)
            i += 1
            continue

        if s.startswith("#### "):
            add_para(doc, s[5:].strip(), font=HEAD_FONT, size=12, bold=True,
                     space_before=8, space_after=6)
            i += 1
            continue

        # 引用块（假设、模型式）
        if s.startswith("> "):
            add_para(doc, s[2:].strip(), size=12,
                     align=WD_ALIGN_PARAGRAPH.CENTER, space_before=6,
                     space_after=8, line_spacing=1.5)
            i += 1
            continue
        if s == ">":
            i += 1
            continue

        # 表题 / 图题（加粗行）
        if re.match(r"^\*\*(表|图)\s*\d", s):
            add_para(doc, s.replace("**", ""), font=HEAD_FONT, size=11,
                     align=WD_ALIGN_PARAGRAPH.CENTER, space_before=10,
                     space_after=4, line_spacing=1.3)
            i += 1
            continue

        # 普通段落
        if s:
            add_para(doc, s, size=12, indent_chars=2, line_spacing=1.5,
                     space_after=6)
        i += 1

    add_page_number_footer(doc.sections[0])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT))
    print(f"已生成：{OUT}")
    print(f"文件大小：{OUT.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    build()
