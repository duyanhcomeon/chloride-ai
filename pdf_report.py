"""Two-page, vector PDF report for the research MVP.

Uses system fonts when available; never ships or bundles font files.
Falls back to an English-language PDF when no Vietnamese-capable font exists.
"""
from __future__ import annotations
from io import BytesIO
from datetime import datetime
from pathlib import Path
import math
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

NAVY = colors.HexColor('#102a43')
TEAL = colors.HexColor('#087f8c')
GREEN = colors.HexColor('#137765')
MUTED = colors.HexColor('#60798d')
GRID = colors.HexColor('#e1eaf0')
PALE = colors.HexColor('#edf6f8')
AMBER = colors.HexColor('#a46a1d')
PAPER = colors.white


def available_fonts():
    candidates = [
        ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'),
        ('/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf', '/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf'),
        ('/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf', '/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf'),
    ]
    for reg, bold in candidates:
        if Path(reg).exists() and Path(bold).exists():
            if 'CLReg' not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont('CLReg', reg))
                pdfmetrics.registerFont(TTFont('CLBold', bold))
            return 'CLReg', 'CLBold', True
    return 'Helvetica', 'Helvetica-Bold', False


def build_pdf_report(parameters: dict, time_axis, chloride_axis,
                     comparison: pd.DataFrame | None = None,
                     validation: dict | None = None) -> bytes:
    """Generate a reproducible, self-contained PDF from displayed analysis inputs.

    Parameters expected: fly_ash, wb, cover, cs, c0, critical, horizon,
    d28, m, years, d_source, model_source; only report accepted data.
    """
    base, bold, utf = available_fonts()
    T = (lambda vn, en: vn if utf else en)
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4, pageCompression=1)
    c.setTitle('CloriLife AI v0.3 | Durability analysis report')
    c.setAuthor('CloriLife AI research application')
    W, H = A4
    L = 42
    R = W - L

    def txt(x, y, s, size=9, color=NAVY, weight=False):
        c.setFillColor(color)
        c.setFont(bold if weight else base, size)
        c.drawString(x, y, str(s))

    def right(x, y, s, size=9, color=NAVY, weight=False):
        c.setFillColor(color)
        c.setFont(bold if weight else base, size)
        c.drawRightString(x, y, str(s))

    def row(x, y, name, value, width=218, small=9):
        txt(x, y, name, 8, MUTED)
        right(x+width, y, value, small, NAVY, True)
        c.setStrokeColor(GRID)
        c.line(x, y-7, x+width, y-7)

    def title(y, code, title_text):
        txt(L, y, code, 9, TEAL, True)
        txt(L, y-22, title_text, 13, NAVY, True)

    def footer(page):
        c.setStrokeColor(GRID)
        c.line(L, 49, R, 49)
        txt(L, 35, 'CLORILIFE AI  /  RESEARCH PROTOTYPE v0.3', 7.3, MUTED)
        right(R, 35, f'{page} / 2', 7.3, MUTED)

    def scientific(v):
        if v is None: return '--'
        return f'{float(v) / 1e-12:.2f} x 10^-12 m2/s'

    def yeartext(v):
        return (f'{v:.1f} ' + T('năm','years')) if v is not None and np.isfinite(v) else (
            '> ' + str(parameters['horizon']) + ' ' + T('năm (chưa đạt)', 'years (not reached)'))

    # PAGE 1: overview and chloride curve
    c.setFillColor(NAVY)
    c.rect(0, H - 151, W, 151, stroke=0, fill=1)
    c.setFillColor(TEAL)
    c.rect(0, H-151, 6, 151, stroke=0, fill=1)
    txt(L, H-47, 'CLORILIFE AI', 23, PAPER, True)
    txt(L, H-70, 'CONCRETE DURABILITY INTELLIGENCE  |  v0.3', 9, colors.HexColor('#a8e7e3'))
    txt(L, H-100, T('BÁO CÁO PHÂN TÍCH XÂM NHẬP CLORUA', 'CHLORIDE INGRESS ANALYSIS REPORT'), 13, PAPER, True)
    txt(L, H-122, T('Thời điểm đạt ngưỡng clorua tại cốt thép', 'Threshold arrival time at reinforcement'), 9, colors.HexColor('#d5e4ed'))
    right(R, H-37, datetime.now().strftime('%d/%m/%Y'), 8.5, colors.HexColor('#c7d9e5'))

    card_y = H-267
    c.setFillColor(PALE)
    c.roundRect(L, card_y, R-L, 96, 12, stroke=0, fill=1)
    txt(L+16, card_y+68, T('THỜI GIAN ĐẠT NGƯỠNG', 'THRESHOLD ARRIVAL TIME'), 9, MUTED, True)
    txt(L+16, card_y+34, yeartext(parameters['years']), 21, GREEN, True)
    txt(L+310, card_y+66, 'D28', 9, MUTED, True)
    txt(L+310, card_y+42, scientific(parameters['d28']), 10, NAVY, True)
    txt(L+310, card_y+23, T('Cơ sở: ' + str(parameters['d_source']), 'Source: '+str(parameters['d_source_en'])), 7, MUTED)

    title(H-298, '01 / INPUTS', T('Thông số mô hình khuếch tán', 'Diffusion model parameters'))
    vals = [
        (T('Tro bay / chất kết dính', 'Fly ash / binder'), f"{parameters['fly_ash']:.1f} %"),
        (T('Nước / chất kết dính', 'Water / binder'), f"{parameters['wb']:.2f}"),
        (T('Lớp bảo vệ cốt thép', 'Concrete cover'), f"{parameters['cover']:.1f} mm"),
        (T('Số mũ già hóa', 'Aging exponent'), f"{parameters['m']:.2f}"),
        (T('Clorua bề mặt Cs', 'Surface chloride Cs'), f"{parameters['cs']:.2f} %"),
        (T('Clorua tới hạn Ccrit', 'Threshold Ccrit'), f"{parameters['critical']:.2f} %"),
        (T('Clorua ban đầu C0', 'Initial chloride C0'), f"{parameters['c0']:.2f} %"),
        (T('Giới hạn phân tích', 'Analysis horizon'), f"{parameters['horizon']} " + T('năm', 'years')),
    ]
    top = H - 349
    for ix,(k,v) in enumerate(vals):
        col = ix % 2
        line = ix // 2
        row(L + col*277, top-line*26, k, v, width=237)

    title(H-482, '02 / CHLORIDE PROFILE', T('Diễn biến clorua tại chiều sâu cốt thép', 'Chloride at reinforcement cover depth'))
    x0, y0, graphw, graphh = 65, 135, 475, 165
    c.setStrokeColor(GRID)
    c.setLineWidth(0.65)
    t = np.asarray(time_axis, dtype=float)
    cl = np.asarray(chloride_axis, dtype=float)
    if t.shape != cl.shape or t.size < 2 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(cl)):
        raise ValueError('Dữ liệu đường cong clorua không hợp lệ')
    ymax = max(float(max(cl)), float(parameters['critical']), 0.1)*1.16
    for i in range(5):
        y = y0 + graphh*i/4
        c.line(x0,y,x0+graphw,y)
        right(x0-5,y-3,f'{ymax*i/4:.2f}',7.5,MUTED)
    for i in range(6):
        x = x0 + graphw*i/5
        c.line(x,y0,x,y0+graphh)
        txt(x-3,y0-16,str(int(parameters['horizon']*i/5)),7.5,MUTED)
    crit_y = y0 + graphh * float(parameters['critical'])/ymax
    c.setStrokeColor(colors.HexColor('#e4a14e'))
    c.setDash(4,3)
    c.setLineWidth(1.4)
    c.line(x0,crit_y,x0+graphw,crit_y)
    c.setDash()
    c.setStrokeColor(TEAL)
    c.setLineWidth(2)
    p = c.beginPath()
    for i,(tt,cc) in enumerate(zip(t,cl)):
        x = x0 + graphw * max(0,min(float(tt)/parameters['horizon'],1))
        y = y0 + graphh * float(cc)/ymax
        if i == 0: p.moveTo(x,y)
        else: p.lineTo(x,y)
    c.drawPath(p)
    txt(x0, y0-37, T('Đường xanh: Cl- tại cốt thép  |  Đường nét đứt: Ccrit giả định',
                     'Teal: chloride at rebar  |  Dashed: assumed Ccrit'),8.5,MUTED)
    txt(L,75,T('Giới hạn: kết quả KHÔNG phải tuổi thọ chịu lực của kết cấu.',
                'LIMITATION: this is NOT structural service life.'),8.5,AMBER,True)
    footer(1)
    c.showPage()

    # PAGE 2: mix comparison and method traceability
    c.setFillColor(NAVY)
    c.rect(0,H-97,W,97,stroke=0,fill=1)
    txt(L,H-45,'CLORILIFE AI  |  v0.3',17,PAPER,True)
    txt(L,H-69,T('So sánh cấp phối và kiểm định mô hình', 'Mix comparison and AI holdout evaluation'),9,colors.HexColor('#b9d5dd'))
    top_y = H-129
    title(top_y,'03 / MIX COMPARISON',T('So sánh các hàm lượng tro bay', 'Fly-ash dosage comparison'))
    origin = str(parameters.get('comparison_origin', ''))
    if 'GIẢ ĐỊNH' in origin or 'GIA DINH' in origin:
        c_line = T('CẢNH BÁO: D28 so sánh là GIẢ ĐỊNH / CHƯA XÁC NHẬN THÍ NGHIỆM.',
                   'WARNING: Comparison D28 values are ASSUMED / UNVERIFIED.')
        c_color = AMBER
    elif 'MÔ PHỎNG' in origin:
        c_line = T('CẢNH BÁO: D28 so sánh do AI học từ bộ dữ liệu MÔ PHỎNG.',
                   'WARNING: D28 estimated using SYNTHETIC DEMO input.')
        c_color = AMBER
    else:
        c_line = T('So sánh dựa trên D28 của từng cấp phối; cần xác nhận dữ liệu.',
                   'Comparison uses per-mix D28; ensure measurements are verified.')
        c_color = MUTED
    txt(L, top_y-42, c_line, 7.5, c_color, True)
    if comparison is not None and not comparison.empty:
        ty = top_y-71
        widths = [96, 130, 164, 120]
        xs = [L]
        for width in widths: xs.append(xs[-1]+width)
        c.setFillColor(NAVY)
        c.roundRect(L,ty-12,R-L,28,5,stroke=0,fill=1)
        headers = [T('Tro bay','Fly ash'), 'D28 (x10^-12)', T('Đạt ngưỡng','Threshold time'), T('Cl- năm 10','Cl- year 10')]
        for i,hdr in enumerate(headers):
            txt(xs[i]+8,ty-1,hdr,8,PAPER,True)
        for i,(_,rec) in enumerate(comparison.head(9).iterrows()):
            y = ty-32-i*29
            c.setFillColor(PALE if i%2 == 0 else PAPER)
            c.rect(L,y-13,R-L,29,fill=1,stroke=0)
            numeric = rec.get('Thời gian đạt ngưỡng (năm)')
            fields = [
                f"{rec['Tro bay (%)']:.0f}%",
                f"{rec['D28 (x10^-12 m2/s)']:.2f}",
                f'{numeric:.1f} ' + T('năm','yr') if pd.notna(numeric) else '> '+str(parameters['horizon']) + ' '+T('năm','yr'),
                f"{rec['Cl- tại thép năm 10 (%)']:.3f}%",
            ]
            for j,value in enumerate(fields):
                txt(xs[j]+8,y, value,8,NAVY,weight=(j==2))
        end_table = ty-52-29*min(len(comparison),9)
    else:
        txt(L,top_y-82,T('Chưa có kịch bản so sánh hợp lệ.', 'No eligible comparison cases.'),9,MUTED)
        end_table = top_y-105

    title(end_table-9,'04 / MODEL VALIDATION',T('Kiểm định AI trên tập giữ lại', 'AI evaluation on held-out data'))
    vm_y = end_table-62
    if validation is not None:
        txt(L,vm_y, f"n = {validation.get('n','--')}   |   train = {validation.get('train_count','--')}   |   test = {validation.get('test_count','--')}",8.5,NAVY)
        vm_y -= 24
        txt(L,vm_y,'MAE',8.5,MUTED)
        txt(L+62,vm_y, scientific(validation.get('MAE_m2_s')),9,NAVY,True)
        txt(L+266,vm_y,'RMSE',8.5,MUTED)
        txt(L+336,vm_y,scientific(validation.get('RMSE_m2_s')),9,NAVY,True)
        vm_y -= 22
        txt(L,vm_y,'R2 (log10 D28)',8.5,MUTED)
        txt(L+118,vm_y,f"{validation.get('R2_log10',float('nan')):.3f}",9,NAVY,True)
        txt(L+266,vm_y,'R2 (D28)',8.5,MUTED)
        txt(L+360,vm_y,f"{validation.get('R2_linear',float('nan')):.3f}",9,NAVY,True)
        vm_y -= 25
        msg = T('Nguồn AI: ' + str(parameters['model_source']), 'AI data: '+ str(parameters['model_source_en']))
        txt(L,vm_y,msg[:94],7.5,AMBER if parameters.get('synthetic') else MUTED)
    else:
        txt(L,vm_y,T('Chưa nạp CSV nên chưa có kết quả kiểm định AI.', 'No CSV uploaded. AI validation is not available.'),9,MUTED)
        vm_y -= 25
    y_method = min(vm_y-44, 275)
    title(y_method,'05 / SCIENTIFIC BASIS',T('Giả định và tiêu chuẩn tham khảo', 'Assumptions and standards referenced'))
    notes = [
        T('Mô hình Fick 1D, bề mặt bán vô hạn, Cs không đổi; D(t) già hóa theo hàm lũy thừa.',
          '1D Fick model, semi-infinite medium, fixed surface Cs, power-law D(t) aging.'),
        T('TCVN 12041:2017; TCVN 9346:2012; TCVN 10302:2014; ASTM C1556.',
          'TCVN 12041:2017; TCVN 9346:2012; TCVN 10302:2014; ASTM C1556.'),
        T('Các tiêu chuẩn viện dẫn không có nghĩa phần mềm được chứng nhận phù hợp.',
          'Referenced standards do not imply standards compliance or certification.'),
        T('Chưa xét ăn mòn phát triển, nứt, tải trọng hoặc độ tin cậy kết cấu.',
          'Corrosion propagation, cracking, load capacity and reliability are not modeled.'),
        T('Các bộ dữ liệu DEMO là MÔ PHỎNG, không phải số liệu thí nghiệm.',
          'DEMO datasets are SYNTHETIC, not experimentally measured.'),
    ]
    for ix,line in enumerate(notes):
        txt(L+1, y_method-47-ix*19, '\u2022 '+line, 8.2, MUTED)
    footer(2)
    c.save()
    return buf.getvalue()
