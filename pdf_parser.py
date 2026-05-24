import fitz  # PyMuPDF
import re


def extract_lab_values(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()

    pattern = r"([A-Za-z\s/]+)\s+([\d.]+)\s*([a-zA-Z/%µ]+)?\s+([<>]?[\d.\-\s]+)?"
    results = []
    for match in re.finditer(pattern, text):
        name, value, unit, ref = match.groups()
        try:
            numeric_value = float(value)
        except ValueError:
            continue
        results.append(
            {
                "test": name.strip(),
                "value": numeric_value,
                "unit": unit or "",
                "reference": ref.strip() if ref else "N/A",
            }
        )
    return results
