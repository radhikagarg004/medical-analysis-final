
from io import BytesIO
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, Image
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.utils import ImageReader


def generate_pdf_report(
    filename,
    prediction,
    uncertainty,
    quality,
    review_required,
    threshold=2.0,
    gradcam_path=None
):
    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=(595, 842),
        leftMargin=45,
        rightMargin=45,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    elements = []

    def add_heading(text):
        elements.append(
            Paragraph(text, styles["Heading2"])
        )
        elements.append(Spacer(1, 8))

    def add_table(rows):
        data = [
            [
                Paragraph(escape(str(key)), styles["Normal"]),
                Paragraph(escape(str(value)), styles["Normal"])
            ]
            for key, value in rows
        ]

        table = Table(
            data,
            colWidths=[175, 325],
            hAlign="LEFT"
        )

        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1),
             colors.HexColor("#EAF1F8")),
            ("GRID", (0, 0), (-1, -1),
             0.4, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8)
        ]))

        elements.append(table)
        elements.append(Spacer(1, 16))

    elements.append(
        Paragraph("MEDVISION AI", styles["Title"])
    )

    elements.append(
        Paragraph(
            "Automated Chest X-ray Analysis Report",
            styles["Heading2"]
        )
    )
    elements.append(Spacer(1, 15))

    add_table([
        ("Image", filename),
        ("Date", datetime.now().strftime("%d-%m-%Y %H:%M")),
        ("Model", "EfficientNet-B0")
    ])

    add_heading("1. Classification Results")

    add_table([
        ("Predicted Category", prediction["predicted_class"]),
        ("Confidence", f'{prediction["confidence"]}%')
    ])

    add_heading("2. Uncertainty Assessment")

    add_table([
        ("Method", "Monte Carlo Dropout"),
        ("MC Passes", uncertainty["num_passes"]),
        ("Uncertainty Score", uncertainty["uncertainty_score"]),
        ("Review Threshold", threshold)
    ])

    add_heading("3. Image Quality Assessment")

    add_table([
        ("Quality Status", quality["quality_status"])
    ])

    add_heading("4. Human Review Recommendation")

    status = (
        "Human Review Recommended"
        if review_required
        else "No automatic review flag triggered"
    )

    elements.append(
        Paragraph(escape(status), styles["Normal"])
    )
    elements.append(Spacer(1, 15))

    if gradcam_path and Path(gradcam_path).exists():
        add_heading("5. Grad-CAM Visualization")

        image_reader = ImageReader(str(gradcam_path))
        width, height = image_reader.getSize()

        max_width = 330
        max_height = 230
        scale = min(
            max_width / width,
            max_height / height
        )

        elements.append(
            Image(
                str(gradcam_path),
                width=width * scale,
                height=height * scale
            )
        )

        elements.append(Spacer(1, 10))
        elements.append(
            Paragraph(
                "Highlighted regions show areas influencing "
                "the model prediction, not confirmed lesions.",
                styles["Normal"]
            )
        )

    elements.append(Spacer(1, 20))

    add_heading("Important Disclaimer")

    elements.append(
        Paragraph(
            "This is an AI-generated research analysis report. "
            "It is not a medical diagnosis and must not replace "
            "assessment by a qualified healthcare professional. "
            "Low uncertainty does not guarantee correctness.",
            styles["Normal"]
        )
    )

    document.build(elements)
    buffer.seek(0)

    return buffer
