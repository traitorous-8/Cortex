import os
from typing import List
from datetime import datetime
from fpdf import FPDF
from cortex.verify import ConfirmedViolation

def generate_report(violations: List[ConfirmedViolation], evidence_dir: str, output_path: str, site_name: str, period: str):
    """
    Generates a professional PDF report from confirmed violations using fpdf2.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Calculate statistics
    total_violations = len(violations)
    no_helmet_count = sum(1 for v in violations if v.kind == "no_helmet")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Load Cyrillic font
    font_path = os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf")
    pdf.add_font("DejaVu", "", font_path)

    # Title Page
    pdf.add_page()
    pdf.set_font("DejaVu", "", 24)
    pdf.set_text_color(26, 82, 118) # #1a5276
    pdf.ln(50)
    pdf.cell(0, 10, "Отчёт о нарушениях техники безопасности", align="C")
    pdf.ln(20)

    pdf.set_font("DejaVu", "", 16)
    pdf.set_text_color(85, 85, 85) # #555
    pdf.cell(0, 10, site_name, align="C")
    pdf.ln(30)

    pdf.set_font("DejaVu", "", 12)
    pdf.set_text_color(102, 102, 102) # #666
    pdf.cell(0, 8, f"Период: {period}", align="C")
    pdf.ln(8)
    pdf.cell(0, 8, f"Сгенерировано: {now_str}", align="C")
    pdf.ln(8)
    pdf.cell(0, 8, "Система: Cortex-Core", align="C")

    # Summary Section
    pdf.add_page()
    pdf.set_font("DejaVu", "", 18)
    pdf.set_text_color(26, 82, 118)
    pdf.cell(0, 10, "Сводка", border="B")
    pdf.ln(15)

    pdf.set_font("DejaVu", "", 14)
    pdf.set_text_color(51, 51, 51)

    pdf.set_fill_color(235, 245, 251) # #ebf5fb
    # Hack for borders and filling in fpdf
    pdf.cell(0, 15, f"    Всего подтверждённых нарушений: {total_violations}", fill=True)
    pdf.ln(15)
    pdf.cell(0, 15, f"    Из них \"Отсутствие каски\": {no_helmet_count}", fill=True)
    pdf.ln(20)

    # Detail Section
    pdf.set_font("DejaVu", "", 18)
    pdf.set_text_color(26, 82, 118)
    pdf.cell(0, 10, "Детализация нарушений", border="B")
    pdf.ln(15)

    if total_violations == 0:
        pdf.set_font("DejaVu", "", 12)
        pdf.set_text_color(51, 51, 51)
        pdf.cell(0, 10, "Нарушений не обнаружено.")
    else:
        for v in violations:
            pdf.add_page()

            # Draw a card-like border around the content if you wish,
            # but simpler just to layout the details.

            image_path = os.path.join(evidence_dir, f"evidence_track_{v.track_id}_frame_{v.evidence_frame}.jpg")
            type_str = "Отсутствие каски" if v.kind == "no_helmet" else v.kind

            minutes = int(v.start_time_s // 60)
            seconds = int(v.start_time_s % 60)
            timestamp_str = f"{minutes:02d}:{seconds:02d}"

            # Evidence Image
            if os.path.exists(image_path):
                # Try to place image, center it
                pdf.image(image_path, x=15, w=100) # w=100 leaves room on the right
                pdf.set_y(pdf.get_y() + 5)
            else:
                pdf.set_font("DejaVu", "", 12)
                pdf.set_text_color(153, 153, 153)
                pdf.cell(100, 50, "Нет изображения", border=1, align="C")
                pdf.set_y(pdf.get_y() + 55)

            # Details below or beside image. Placing below for simplicity,
            # though the prompt suggests "beside" like a card, we can use cell positioning.
            # But vertical flow is safer.

            pdf.set_font("DejaVu", "", 16)
            pdf.set_text_color(192, 57, 43) # #c0392b
            pdf.cell(0, 10, f"Нарушение #{v.track_id}")
            pdf.ln(10)

            pdf.set_font("DejaVu", "", 12)
            pdf.set_text_color(51, 51, 51)
            pdf.cell(0, 8, f"Тип: {type_str}")
            pdf.ln(8)
            pdf.cell(0, 8, f"Время (ММ:СС): {timestamp_str}")
            pdf.ln(8)
            pdf.cell(0, 8, f"Длительность: {v.duration_s:.1f} сек")
            pdf.ln(8)
            pdf.cell(0, 8, f"Уверенность (smoothed): {v.smoothed_confidence:.2f}")
            pdf.ln(15)

    pdf.output(output_path)
