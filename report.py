from io import BytesIO
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie

PRIMARY_BLUE = colors.HexColor('#2563eb')
GRID_COLOR = colors.HexColor('#bfdbfe')
LIGHT_BG = colors.HexColor('#eff6ff')
SENTIMENT_COLORS = {'positive': '#10b981', 'neutral': '#0ea5e9', 'negative': '#ef4444'}


def create_sentiment_pie_chart(counts):
    if sum(counts.values()) == 0:
        return None
    drawing = Drawing(200, 200)
    pie = Pie()
    pie.x, pie.y, pie.width, pie.height = 55, 50, 100, 100
    pie.data = [counts['positive'], counts['neutral'], counts['negative']]
    pie.labels = [f"Positive: {counts['positive']}", f"Neutral: {counts['neutral']}", f"Negative: {counts['negative']}"]
    for idx, key in enumerate(('positive', 'neutral', 'negative')):
        pie.slices[idx].fillColor = colors.HexColor(SENTIMENT_COLORS[key])
    pie.slices.strokeColor = colors.white
    pie.slices.strokeWidth = 1.5
    pie.slices.fontSize = 8
    pie.simpleLabels = 1
    drawing.add(pie)
    return drawing


def build_report(items, start=None, end=None):
    styles = getSampleStyleSheet()
    normal_style = ParagraphStyle('N', parent=styles['Normal'], fontSize=9, leading=12)
    heading_style = ParagraphStyle('H', parent=styles['Heading2'], fontSize=13, textColor=PRIMARY_BLUE, spaceBefore=4, spaceAfter=6)
    title_style = ParagraphStyle('T', parent=styles['Title'], fontSize=18, textColor=colors.HexColor('#1d4ed8'))
    sub_style = ParagraphStyle('S', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#475569'))
    th_style = ParagraphStyle('TH', parent=normal_style, fontName='Helvetica-Bold', textColor=colors.white)

    if start and end:
        period_text = f"{start} to {end}"
    elif start:
        period_text = f"From {start}"
    elif end:
        period_text = f"Until {end}"
    else:
        period_text = "All time"

    counts = {'positive': 0, 'neutral': 0, 'negative': 0}
    for item in items:
        s = (item.get('sentiment') or 'Neutral').lower()
        if s in counts:
            counts[s] += 1
        else:
            counts['neutral'] += 1
    total = len(items)

    def format_pct(num):
        return f"{(num / total * 100):.1f}%" if total > 0 else "0%"

    story = [
        Paragraph("AI Sentiment Analyzer - Prediction Report", title_style),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Period: {period_text}", sub_style),
        Spacer(1, 8),
        Paragraph("Summary", heading_style)
    ]

    summary_paragraphs = [Paragraph(f"<b>Total reviews: {total}</b>", normal_style)]
    for category in ('positive', 'neutral', 'negative'):
        hex_col = SENTIMENT_COLORS[category]
        cat_title = category.capitalize()
        cat_cnt = counts[category]
        summary_paragraphs.append(
            Paragraph(f'<font color="{hex_col}"><b>{cat_title}:</b></font> {cat_cnt} ({format_pct(cat_cnt)})', normal_style)
        )

    available_width = A4[0] - (2 * 14 * mm)
    chart = create_sentiment_pie_chart(counts) or Paragraph("No data available for the selected period.", normal_style)
    summary_box = Table([[summary_paragraphs, chart]], colWidths=[available_width * 0.5, available_width * 0.5])
    summary_box.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOX', (0, 0), (-1, -1), 0.5, GRID_COLOR),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8)
    ]))

    story.extend([summary_box, Spacer(1, 12), Paragraph("Prediction Records", heading_style)])

    if items:
        headers = ['#', 'Date', 'Review', 'Sentiment', 'Confidence']
        table_rows = [[Paragraph(h, th_style) for h in headers]]

        for idx, item in enumerate(items, 1):
            sent = (item.get('sentiment') or 'Neutral').capitalize()
            sent_lower = sent.lower()
            color = SENTIMENT_COLORS.get(sent_lower, '#334155')
            pos = float(item.get('positive') or 0.0)
            neu = float(item.get('neutral') or 0.0)
            neg = float(item.get('negative') or 0.0)
            max_confidence = max(pos, neu, neg)
            table_rows.append([
                Paragraph(str(idx), normal_style),
                Paragraph(str(item.get('created_at') or ''), normal_style),
                Paragraph(str(item.get('review') or ''), normal_style),
                Paragraph(f'<font color="{color}"><b>{sent}</b></font>', normal_style),
                Paragraph(f"{max_confidence:.2f}%", normal_style)
            ])

        col_widths = [28, 80, available_width - 28 - 80 - 70 - 70, 70, 70]
        records_table = Table(table_rows, colWidths=col_widths, repeatRows=1)
        records_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_BLUE),
            ('GRID', (0, 0), (-1, -1), 0.5, GRID_COLOR),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5)
        ]))
        story.append(records_table)
    else:
        story.append(Paragraph("No records found in the selected period.", normal_style))

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=14 * mm, leftMargin=14 * mm, topMargin=14 * mm, bottomMargin=14 * mm, title="Sentiment Report"
    )
    doc.build(story)
    return buffer.getvalue()

