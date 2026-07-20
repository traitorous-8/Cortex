import os
import base64
from typing import List
from datetime import datetime
from weasyprint import HTML
from cortex.verify import ConfirmedViolation

def generate_report(violations: List[ConfirmedViolation], evidence_dir: str, output_path: str, site_name: str, period: str):
    """
    Generates a professional PDF report from confirmed violations using WeasyPrint.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Calculate statistics
    total_violations = len(violations)
    no_helmet_count = sum(1 for v in violations if v.kind == "no_helmet")

    # Build HTML for detail section
    detail_html = ""
    for v in violations:
        # Construct evidence image path
        image_path = os.path.join(evidence_dir, f"evidence_track_{v.track_id}_frame_{v.evidence_frame}.jpg")

        # Determine violation type localized string
        type_str = "Отсутствие каски" if v.kind == "no_helmet" else v.kind

        # Format timestamp mm:ss
        minutes = int(v.start_time_s // 60)
        seconds = int(v.start_time_s % 60)
        timestamp_str = f"{minutes:02d}:{seconds:02d}"

        image_html = ""
        if os.path.exists(image_path):
            # Encode image to base64 for embedding
            with open(image_path, "rb") as f:
                encoded_string = base64.b64encode(f.read()).decode('utf-8')
            image_html = f'<img src="data:image/jpeg;base64,{encoded_string}" alt="Evidence" class="evidence-img"/>'
        else:
            image_html = '<div class="no-image">No image available</div>'

        detail_html += f"""
        <div class="violation-card">
            <div class="card-image">{image_html}</div>
            <div class="card-details">
                <h3>Нарушение #{v.track_id}</h3>
                <p><strong>Тип:</strong> {type_str}</p>
                <p><strong>Время (ММ:СС):</strong> {timestamp_str}</p>
                <p><strong>Длительность:</strong> {v.duration_s:.1f} сек</p>
                <p><strong>Уверенность (smoothed):</strong> {v.smoothed_confidence:.2f}</p>
            </div>
        </div>
        """

    # Assemble full HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>Отчёт о нарушениях техники безопасности</title>
        <style>
            body {{
                font-family: "DejaVu Sans", sans-serif;
                color: #333;
                background-color: #fff;
                margin: 0;
                padding: 20px;
            }}
            .title-page {{
                text-align: center;
                margin-top: 150px;
                page-break-after: always;
            }}
            h1 {{
                color: #1a5276;
                font-size: 32px;
                margin-bottom: 10px;
            }}
            .subtitle {{
                font-size: 20px;
                color: #555;
                margin-bottom: 30px;
            }}
            .metadata {{
                font-size: 16px;
                color: #666;
            }}
            .summary-section, .detail-section {{
                margin-bottom: 40px;
            }}
            h2 {{
                color: #1a5276;
                border-bottom: 2px solid #1a5276;
                padding-bottom: 5px;
            }}
            .stats-box {{
                background-color: #ebf5fb;
                padding: 15px;
                border-left: 5px solid #2980b9;
                border-radius: 4px;
            }}
            .violation-card {{
                display: flex;
                border: 1px solid #ddd;
                border-radius: 8px;
                margin-bottom: 20px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }}
            .card-image {{
                width: 300px;
                background-color: #f9f9f9;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .evidence-img {{
                max-width: 100%;
                max-height: 250px;
                object-fit: contain;
            }}
            .no-image {{
                color: #999;
                font-style: italic;
            }}
            .card-details {{
                padding: 20px;
                flex: 1;
            }}
            .card-details h3 {{
                margin-top: 0;
                color: #c0392b;
            }}
            .card-details p {{
                margin: 5px 0;
            }}
        </style>
    </head>
    <body>
        <div class="title-page">
            <h1>Отчёт о нарушениях техники безопасности</h1>
            <div class="subtitle">{site_name}</div>
            <div class="metadata">
                <p><strong>Период:</strong> {period}</p>
                <p><strong>Сгенерировано:</strong> {now_str}</p>
                <p><strong>Система:</strong> Cortex-Core</p>
            </div>
        </div>

        <div class="summary-section">
            <h2>Сводка</h2>
            <div class="stats-box">
                <p><strong>Всего подтверждённых нарушений:</strong> {total_violations}</p>
                <p><strong>Из них "Отсутствие каски":</strong> {no_helmet_count}</p>
            </div>
        </div>

        <div class="detail-section">
            <h2>Детализация нарушений</h2>
            {detail_html if total_violations > 0 else "<p>Нарушений не обнаружено.</p>"}
        </div>
    </body>
    </html>
    """

    HTML(string=html_content).write_pdf(output_path)
