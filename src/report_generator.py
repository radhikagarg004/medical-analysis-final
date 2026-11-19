
from io import BytesIO
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, Image, KeepTogether
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
    gradcam_path=None,
    similar_images=None
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
        image_reader = ImageReader(str(gradcam_path))
        width, height = image_reader.getSize()
        scale = min(330 / width, 230 / height)
        elements.append(KeepTogether([
            Paragraph("5. Grad-CAM Visualization", styles["Heading2"]),
            Spacer(1, 8),
            Image(str(gradcam_path), width=width * scale, height=height * scale),
            Spacer(1, 10),
            Paragraph(
                "Highlighted regions show areas influencing the model prediction, "
                "not confirmed lesions.", styles["Normal"]
            ),
        ]))

    add_heading("6. Similar Image Retrieval - Top 5 Matches")
    if similar_images:
        # Use a two-column layout so all five actual X-rays are readable.
        for offset in range(0, min(5, len(similar_images)), 2):
            cells = []
            for match in similar_images[offset:offset + 2]:
                path = Path(match["image_path"])
                if not path.is_file():
                    cells.append(Paragraph(
                        "Reference image unavailable: " + escape(path.name),
                        styles["Normal"]
                    ))
                    continue
                try:
                    with ImageReader(str(path)) as reader:
                        width, height = reader.getSize()
                    scale = min(220 / width, 145 / height)
                    score = float(match["similarity_score"])
                    label = escape(str(match["dataset_class"]))
                    rank = int(match["rank"])
                    cells.append(KeepTogether([
                        Image(str(path), width=width * scale, height=height * scale),
                        Spacer(1, 5),
                        Paragraph(
                            f"Match {rank} | Reference class: {label}",
                            styles["Normal"]
                        ),
                        Paragraph(
                            f"Cosine similarity: {score:.4f}",
                            styles["Normal"]
                        )
                    ]))
                except (OSError, ValueError):
                    cells.append(Paragraph(
                        "Unable to display reference image: " + escape(path.name),
                        styles["Normal"]
                    ))
            while len(cells) < 2:
                cells.append("")
            row = Table([cells], colWidths=[250, 250], hAlign="LEFT")
            row.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 15),
            ]))
            elements.append(row)
        elements.append(Paragraph(
            "These are visually similar training references, not diagnostic "
            "confirmation. Similarity scores are not disease probabilities.",
            styles["Normal"]
        ))
    else:
        elements.append(Paragraph(
            "No similar reference images were available.",
            styles["Normal"]
        ))

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
