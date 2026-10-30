"""Reproducible PDF report with conditional corrosion-scenario page.

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

from corrosion import COL_REMAIN, COL_TIME, remaining_bar_history

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
        ('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'),
        ('/usr/share/fonts/dejavu/DejaVuSans.ttf', '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf'),
        ('/usr/share/fonts/TTF/DejaVuSans.ttf', '/usr/share/fonts/TTF/DejaVuSans-Bold.ttf'),
        ('C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/arialbd.ttf'),
        ('/System/Library/Fonts/Supplemental/Arial.ttf', '/System/Library/Fonts/Supplemental/Arial Bold.ttf'),
    ]
    for reg, bold in candidates:
        if Path(reg).exists() and Path(bold).exists():
            try:
                if 'CLReg' not in pdfmetrics.getRegisteredFontNames():
                    pdfmetrics.registerFont(TTFont('CLReg', reg))
                    pdfmetrics.registerFont(TTFont('CLBold', bold))
                return 'CLReg', 'CLBold', True
            except Exception:
                continue  # unreadable font file: try the next candidate
    return 'Helvetica', 'Helvetica-Bold', False


def build_pdf_report(parameters: dict, time_axis, chloride_axis,
                     comparison: pd.DataFrame | None = None,
                     validation: dict | None = None,
                     propagation: dict | None = None) -> bytes:
    """Generate a reproducible, self-contained PDF from displayed analysis inputs.

    Parameters expected: fly_ash, wb, cover, cs, c0, critical, horizon,
    d28, m, years, d_source, model_source; only report accepted data.
    """
    base, bold, utf = available_fonts()
    T = (lambda vn, en: vn if utf else en)
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4, pageCompression=1)
    c.setTitle('CloriLife AI v0.4.1 | Durability and uniform corrosion scenario')
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
        txt(L, 35, 'CLORILIFE AI  /  RESEARCH PROTOTYPE v0.4.1', 7.3, MUTED)
        right(R, 35, f'{page} / {3 if propagation is not None else 2}', 7.3, MUTED)

    def scientific(v):
        if v is None: return '--'
        return f'{float(v) / 1e-12:.2f} x 10^-12 m2/s'

    def rcm_text(v):
        if v is None or not np.isfinite(float(v)):
            return '--'
        return f'{float(v):.2f} x 10^-12 m2/s'

    def yeartext(v):
        return (f'{v:.1f} ' + T('năm','years')) if v is not None and np.isfinite(v) else (
            '> ' + str(parameters['horizon']) + ' ' + T('năm (chưa đạt)', 'years (not reached)'))

    # PAGE 1: overview and chloride curve
    c.setFillColor(NAVY)
    c.rect(0, H - 151, W, 151, stroke=0, fill=1)
    c.setFillColor(TEAL)
    c.rect(0, H-151, 6, 151, stroke=0, fill=1)
    txt(L, H-47, 'CLORILIFE AI', 23, PAPER, True)
    txt(L, H-70, 'CONCRETE DURABILITY INTELLIGENCE  |  v0.4.1', 9, colors.HexColor('#a8e7e3'))
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
    txt(L,H-45,'CLORILIFE AI  |  v0.4.1',17,PAPER,True)
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
        MAX_ROWS = 10
        shown = comparison.head(MAX_ROWS)
        row_h = 29 if len(shown) <= 6 else 22   # tighter rows keep notes clear of the footer
        if len(comparison) > MAX_ROWS:
            right(R, top_y-42, T(f'Hiển thị {MAX_ROWS}/{len(comparison)} dòng',
                                 f'Showing {MAX_ROWS} of {len(comparison)} rows'), 7.5, MUTED)
        for i,(_,rec) in enumerate(shown.iterrows()):
            y = ty-32-i*row_h
            c.setFillColor(PALE if i%2 == 0 else PAPER)
            c.rect(L,y-row_h/2+3,R-L,row_h,fill=1,stroke=0)
            numeric = rec.get('Thời gian đạt ngưỡng (năm)')
            fields = [
                f"{rec['Tro bay (%)']:.0f}%",
                f"{rec['D28 (x10^-12 m2/s)']:.2f}",
                f'{numeric:.1f} ' + T('năm','yr') if pd.notna(numeric) else '> '+str(parameters['horizon']) + ' '+T('năm','yr'),
                f"{rec['Cl- tại thép năm 10 (%)']:.3f}%",
            ]
            for j,value in enumerate(fields):
                txt(xs[j]+8,y, value,8,NAVY,weight=(j==2))
        end_table = ty-32-(len(shown)-1)*row_h-40
    else:
        txt(L,top_y-82,T('Chưa có kịch bản so sánh hợp lệ.', 'No eligible comparison cases.'),9,MUTED)
        end_table = top_y-105

    title(end_table-9,'04 / MODEL VALIDATION',T('Kiểm định AI (RCM) trên tập giữ lại', 'AI (RCM) evaluation on held-out data'))
    vm_y = end_table-62
    if validation is not None:
        txt(L,vm_y, f"n = {validation.get('n','--')}   |   train = {validation.get('train_count','--')}   |   test = {validation.get('test_count','--')}",8.5,NAVY)
        vm_y -= 24
        txt(L,vm_y,'MAE',8.5,MUTED)
        txt(L+62,vm_y, rcm_text(validation.get('MAE_1e12')),9,NAVY,True)
        txt(L+266,vm_y,'RMSE',8.5,MUTED)
        txt(L+336,vm_y,rcm_text(validation.get('RMSE_1e12')),9,NAVY,True)
        vm_y -= 22
        txt(L,vm_y,'R2 (log10 RCM)',8.5,MUTED)
        txt(L+118,vm_y,f"{validation.get('R2_log10',float('nan')):.3f}",9,NAVY,True)
        txt(L+266,vm_y,'R2 (linear)',8.5,MUTED)
        txt(L+360,vm_y,f"{validation.get('R2_linear',float('nan')):.3f}",9,NAVY,True)
        vm_y -= 25
        msg = T('Nguồn AI: ' + str(parameters['model_source']), 'AI data: '+ str(parameters['model_source_en']))
        txt(L,vm_y,msg[:94],7.5,AMBER if parameters.get('synthetic') else MUTED)
    else:
        txt(L,vm_y,T('Chưa nạp dữ liệu nên chưa có kết quả kiểm định AI.', 'No training data loaded. AI validation is not available.'),9,MUTED)
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
        T('Chỉ xét ăn mòn đều; chưa xét ăn mòn rỗ, nứt, tải trọng hoặc sức chịu lực.',
          'Uniform corrosion only; no pitting, cracking or load capacity analysis.'),
        T('AI dự báo hệ số RCM, KHÔNG phải D28; không đưa vào mô hình Fick.',
          'AI predicts RCM coefficients, NOT D28; not used in the Fick model.'),
    ]
    for ix,line in enumerate(notes):
        txt(L+1, y_method-47-ix*19, '\u2022 '+line, 8.2, MUTED)
    footer(2)
    if propagation is not None:
        c.showPage()
        c.setFillColor(NAVY)
        c.rect(0, H-123, W, 123, stroke=0, fill=1)
        c.setFillColor(TEAL)
        c.rect(0,H-123,6,123,stroke=0,fill=1)
        txt(L,H-46,'CLORILIFE AI  |  v0.4.1',19,PAPER,True)
        txt(L,H-74,T('KỊCH BẢN ĂN MÒN PHÁT TRIỂN', 'UNIFORM CORROSION SCENARIO'),12,PAPER,True)
        txt(L,H-99,T('Mốc hao hụt thép được người dùng tự chọn',
                       'User-selected steel area-loss limit'),8.5,colors.HexColor('#bad7e5'))
        title(H-156,'06 / TWO-STAGE INDICATOR',
              T('Hai giai đoạn của chỉ số tuổi thọ sơ bộ',
                'Two stages of the conditional serviceability indicator'))

        p1=propagation.get('initiation_years')
        p2=propagation['propagation_years']
        pt=propagation.get('total_years')
        crack_um=propagation.get('cracking_penetration_um')
        crack_total=propagation.get('cracking_total_years')
        crack_after=propagation.get('cracking_years')
        unknown=T('Chưa xác định','Unknown')
        cards=[
            (T('KHỞI PHÁT', 'INITIATION'),
             (f'{p1:.1f} '+T('năm','yr')) if p1 is not None else unknown),
            (T('MỐC NỨT LỚP BẢO VỆ', 'COVER CRACKING'),
             (f'{crack_total:.1f} '+T('năm','yr')) if crack_total is not None else unknown),
            (T('ĂN MÒN ĐẾN NGƯỠNG', 'PROPAGATION'),f'{p2:.1f} '+T('năm','yr')),
            (T('MỐC HAO HỤT THÉP', 'AREA LOSS LIMIT'),
             (f'{pt:.1f} '+T('năm','yr')) if pt is not None else unknown),
        ]
        for j,(label,value) in enumerate(cards):
            bx=L+j*131
            c.setFillColor(PALE)
            c.roundRect(bx,H-299,118,75,9,stroke=0,fill=1)
            txt(bx+8,H-248,label,7,MUTED,True)
            txt(bx+8,H-273,value,12.5,GREEN if j==3 else NAVY,True)

        title(H-331,'07 / ASSUMPTIONS',
              T('Thông số và quy tắc suy giảm tiết diện thép',
                'Inputs and uniform bar area-loss model'))
        inputs = [
            (T('Đường kính thanh thép', 'Bar diameter'),f"{propagation['diameter_mm']:.1f} mm"),
            (T('Mật độ dòng ăn mòn', 'Corrosion current density'), f"{propagation['current_density_uA_cm2']:.2f} µA/cm²" if utf else f"{propagation['current_density_uA_cm2']:.2f} uA/cm2"),
            (T('Ngưỡng hao hụt tiết diện', 'Area loss limit'),f"{propagation['area_loss_limit_pct']:.1f} %"),
            (T('Tốc độ xuyên sâu', 'Penetration rate'),f"{propagation['rate_mm_per_year']:.4f} mm/"+T('năm','yr')),
            (T('Độ xuyên sâu tại ngưỡng', 'Penetration at threshold'),f"{propagation['penetration_limit_mm']:.3f} mm"),
        ]
        if crack_um is not None:
            inputs.append((T('Độ xuyên sâu gây nứt (giả định)', 'Cracking penetration (assumed)'),
                           f"{crack_um:.0f} " + ('µm' if utf else 'um') +
                           (f" ({crack_after:.1f} " + T('năm sau khởi phát', 'yr after initiation') + ')'
                            if crack_after is not None else '')))
        for i,(label,value) in enumerate(inputs):
            row(L,H-381-i*24,label,value,width=510)

        title(H-532,'08 / BAR AREA RETENTION',
              T('Diễn biến tiết diện cốt thép còn lại',
                'Remaining reinforcing bar area over time'))
        if pt is not None:
            trajectory=remaining_bar_history(propagation, n=230)
            tt=np.asarray(trajectory[COL_TIME],dtype=float)
            rr=np.asarray(trajectory[COL_REMAIN],dtype=float)
            gx,gy,gw,gh=L+27,157,438,104
            c.setStrokeColor(GRID)
            for k in range(5):
                y=gy+gh*k/4
                c.line(gx,y,gx+gw,y)
                right(gx-7,y-3,f'{100*k/4:.0f}%',7,MUTED)
            tmax=max(float(tt[-1]),1.0)
            for k in range(5):
                x=gx+gw*k/4
                txt(x-3,gy-16,f'{tmax*k/4:.0f}',7,MUTED)
            lim=100-propagation['area_loss_limit_pct']
            c.setStrokeColor(colors.HexColor('#e4a14e'))
            c.setLineWidth(1.3)
            c.setDash(4,3)
            c.line(gx,gy+gh*lim/100,gx+gw,gy+gh*lim/100)
            c.setDash()
            c.setStrokeColor(TEAL)
            c.setLineWidth(1.85)
            curve=c.beginPath()
            for k,(t,r) in enumerate(zip(tt,rr)):
                x=gx+gw*t/tmax
                y=gy+gh*r/100
                if k==0: curve.moveTo(x,y)
                else: curve.lineTo(x,y)
            c.drawPath(curve)
            if crack_total is not None and 0 <= crack_total <= tmax:
                cx=gx+gw*crack_total/tmax
                c.setStrokeColor(colors.HexColor('#c2503f'))
                c.setLineWidth(1.2)
                c.setDash(2,3)
                c.line(cx,gy,cx,gy+gh)
                c.setDash()
                txt(min(cx+3,gx+gw-60),gy+gh-9,T('nứt','cracking'),7,colors.HexColor('#c2503f'),True)
            txt(gx,gy-34,T('Xanh: tiết diện còn lại | Cam đứt: ngưỡng hao hụt | Đỏ đứt: mốc nứt giả định',
                           'Teal: remaining area | Orange: area-loss limit | Red: assumed cracking'),8,MUTED)
        else:
            txt(L,218,T('Chưa đạt ngưỡng clorua trong khoảng phân tích; không có mốc cuối.',
                        'Chloride threshold not reached within horizon; endpoint unknown.'),9,AMBER)

        c.setFillColor(PALE)
        c.roundRect(L,72,R-L,45,8,stroke=0,fill=1)
        txt(L+10,100,T('CƠ SỞ', 'BASIS')+': Faraday; ASTM G102-23 (conversion of electrochemical corrosion rates).',8,NAVY,True)
        txt(L+10,84,T('KHÔNG xét nứt, rỗ, bám dính, nội lực, sức chịu tải hoặc an toàn công trình.',
                       'NO cracking, pitting, bond, loads, remaining strength or structural safety assessment.'),7.4,AMBER,True)
        footer(3)
    c.save()
    return buf.getvalue()
