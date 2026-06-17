"""
Document Processor
Handles PDF ingestion, classification, and structured data extraction.
Uses keyword/regex heuristics (no paid APIs required).
"""

import os
import re
import json
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader


def extract_text_from_file(file_path: str) -> str:
    """Extract raw text from a PDF or plain-text file."""
    path = Path(file_path)
    if path.suffix.lower() == ".txt":
        try:
            return path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return ""

    return extract_text_from_pdf(file_path)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF file (pypdf with pdfminer fallback)."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        text = text.strip()
        if text:
            return text
    except Exception:
        pass

    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        return pdfminer_extract(pdf_path).strip()
    except Exception:
        return ""


def _count_phrase_hits(text_lower: str, phrases: list[str]) -> int:
    return sum(1 for phrase in phrases if phrase in text_lower)


def classify_document(text: str, filename: str) -> str:
    """
    Classify a document into one of:
    Invoice | Resume | Utility Bill | Other | Unclassifiable
    Uses keyword/regex scoring with confidence thresholds.
    """
    if not text:
        return "Unclassifiable"

    text_lower = text.lower()
    fn = filename.lower().replace("_", " ").replace("-", " ")

    scores = {
        "Invoice": 0,
        "Resume": 0,
        "Utility Bill": 0,
        "Other": 0,
    }
    resume_primary_hits = 0

    # ── Other: academic, technical, and general documents (check first) ────────
    academic_phrases = [
        "assignment", "homework", "coursework", "course work", "submission date",
        "submitted by", "due date", "total marks", "marks:", "roll no", "roll number",
        "student id", "student name", "enrollment no", "semester", "instructor",
        "course code", "course:", "question 1", "question 2", "question:", "answer:",
        "problem set", "lab report", "project report", "term paper", "case study",
        "tutorial", "worksheet", "assignment no", "assignment #", "lecture notes",
        "discussion forum", "rubric", "plagiarism", "faculty of", "department of",
    ]
    general_phrases = [
        "data sheet", "datasheet", "technical specification", "product overview",
        "user manual", "operating manual", "white paper", "whitepaper",
        "product catalog", "general document", "random information", "does not fit",
        "brochure", "fact sheet", "research paper", "conference paper",
    ]

    academic_hits = _count_phrase_hits(text_lower, academic_phrases)
    general_hits = _count_phrase_hits(text_lower, general_phrases)
    scores["Other"] += academic_hits * 5 + general_hits * 4

    academic_fn_tokens = [
        "assignment", "homework", "coursework", "lab report", "lab ",
        "project report", "term paper", "essay", "exam", "quiz", "tutorial",
        "worksheet", "submission", "coursework",
    ]
    general_fn_tokens = [
        "datasheet", "data sheet", "spec sheet", "specification",
        "brochure", "manual", "catalog", "whitepaper", "white paper",
    ]

    if any(token in fn for token in academic_fn_tokens):
        scores["Other"] += 9
    elif any(token in fn for token in general_fn_tokens):
        scores["Other"] += 8
    elif "report" in fn or "paper" in fn:
        scores["Other"] += 6

    # Strong academic signal → never classify as resume
    if academic_hits >= 2 or (academic_hits >= 1 and scores["Other"] >= 9):
        return "Other"

    # ── Invoice ──────────────────────────────────────────────────────────────
    invoice_strong = ["invoice #", "invoice number", "bill to", "payment due", "purchase order"]
    invoice_medium = ["invoice", "inv-", "total amount", "subtotal", "amount due", "po number"]
    for kw in invoice_strong:
        if kw in text_lower:
            scores["Invoice"] += 3
    for kw in invoice_medium:
        if kw in text_lower:
            scores["Invoice"] += 2

    # ── Resume — only explicit job-seeking indicators count as primary ───────
    resume_primary_phrases = [
        "curriculum vitae", "work history", "employment history",
        "professional experience", "years of experience", "references available",
    ]
    for phrase in resume_primary_phrases:
        if phrase in text_lower:
            scores["Resume"] += 4
            resume_primary_hits += 1

    if re.search(r"\bresume\b", text_lower) or re.search(r"\bcurriculum vitae\b", text_lower):
        scores["Resume"] += 5
        resume_primary_hits += 1

    if re.search(r"\bcv\b", text_lower) and not re.search(r"\bcv joint\b", text_lower):
        scores["Resume"] += 4
        resume_primary_hits += 1

    if re.search(r"experience[:\s]+\d+\s*years?", text_lower):
        scores["Resume"] += 4
        resume_primary_hits += 1

    if re.search(r"\d+\+?\s*years?\s+of\s+experience", text_lower):
        scores["Resume"] += 3
        resume_primary_hits += 1

    # Contact block typical of resumes (email + phone together near top)
    head = text_lower[:800]
    has_email = bool(re.search(r"email[:\s]", head)) or bool(re.search(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", head))
    has_phone = bool(re.search(r"phone[:\s]", head)) or bool(re.search(r"\+\d[\d\s\-()]{7,}", head))
    if has_email and has_phone and resume_primary_hits > 0:
        scores["Resume"] += 2

    # Secondary resume terms only matter when primary evidence already exists
    if resume_primary_hits > 0:
        resume_support = ["education", "skills", "bachelor", "master", "references", "objective:"]
        support_hits = sum(1 for kw in resume_support if kw in text_lower)
        if support_hits >= 2:
            scores["Resume"] += support_hits

    # ── Utility Bill ─────────────────────────────────────────────────────────
    utility_strong = ["kwh", "meter reading", "utility provider", "billing date", "service address"]
    utility_medium = ["utility", "electricity", "account number", "cityelectric", "powergrid"]
    utility_light = ["electric", "gas", "water", "usage", "amount due"]
    for kw in utility_strong:
        if kw in text_lower:
            scores["Utility Bill"] += 3
    for kw in utility_medium:
        if kw in text_lower:
            scores["Utility Bill"] += 2
    for kw in utility_light:
        if kw in text_lower:
            scores["Utility Bill"] += 1

    # ── Filename hints for known categories ──────────────────────────────────
    if "invoice" in fn or fn.startswith("inv "):
        scores["Invoice"] += 6
    elif "resume" in fn or re.search(r"\bcv\b", fn):
        scores["Resume"] += 6
        resume_primary_hits += 1
    elif "utility" in fn or "utilitybill" in fn.replace(" ", ""):
        scores["Utility Bill"] += 6
    elif "other" in fn:
        scores["Other"] += 6
    elif "unclassifiable" in fn:
        return "Unclassifiable"

    best = max(scores, key=scores.get)
    best_score = scores[best]
    if best_score == 0:
        return "Unclassifiable"

    ranked = sorted(scores.values(), reverse=True)
    margin = ranked[0] - ranked[1]

    # Academic or general docs must not be forced into Resume
    if best == "Resume" and (academic_hits >= 1 or scores["Other"] >= 8):
        return "Other"

    # Resume requires a primary indicator — university/skills alone is not enough
    if best == "Resume" and resume_primary_hits == 0:
        return "Other"

    if best_score < 4:
        return "Other"
    if margin < 2 and best_score < 7:
        return "Other"

    return best


def extract_invoice_fields(text: str) -> dict:
    """Extract structured fields from an Invoice."""
    fields = {
        "invoice_number": None,
        "date": None,
        "company": None,
        "total_amount": None,
    }

    # Invoice number
    match = re.search(r"invoice\s*#?\s*(\w[\w-]*)", text, re.IGNORECASE)
    if match:
        fields["invoice_number"] = match.group(1).strip()

    # Date
    date_match = re.search(
        r"date[:\s]+(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        text, re.IGNORECASE
    )
    if date_match:
        fields["date"] = date_match.group(1).strip()

    # Company
    company_match = re.search(r"company[:\s]+(.+)", text, re.IGNORECASE)
    if company_match:
        fields["company"] = company_match.group(1).strip()

    # Total amount
    amount_match = re.search(
        r"total\s+amount[:\s]+\$?([\d,]+\.?\d*)", text, re.IGNORECASE
    )
    if amount_match:
        try:
            fields["total_amount"] = float(amount_match.group(1).replace(",", ""))
        except ValueError:
            pass

    return fields


def extract_resume_fields(text: str) -> dict:
    """Extract structured fields from a Resume."""
    fields = {
        "name": None,
        "email": None,
        "phone": None,
        "experience_years": None,
    }

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Name: first non-empty line that isn't a label
    for line in lines[:5]:
        if not re.match(r"(email|phone|resume|cv|experience|summary)", line, re.IGNORECASE):
            fields["name"] = line
            break

    # Email
    email_match = re.search(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", text, re.IGNORECASE)
    if email_match:
        fields["email"] = email_match.group(0)

    # Phone
    phone_match = re.search(
        r"(?:phone|tel|mobile)[:\s]*([\+\d\s\-\(\)]{7,20})", text, re.IGNORECASE
    )
    if phone_match:
        fields["phone"] = phone_match.group(1).strip()
    else:
        phone_match2 = re.search(r"(\+?1[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})", text)
        if phone_match2:
            fields["phone"] = phone_match2.group(1).strip()

    # Experience years
    exp_match = re.search(
        r"experience[:\s]+(\d+)\s*year", text, re.IGNORECASE
    )
    if exp_match:
        fields["experience_years"] = int(exp_match.group(1))
    else:
        exp_match2 = re.search(r"(\d+)\+?\s*years?\s+of\s+experience", text, re.IGNORECASE)
        if exp_match2:
            fields["experience_years"] = int(exp_match2.group(1))

    return fields


def extract_utility_fields(text: str) -> dict:
    """Extract structured fields from a Utility Bill."""
    fields = {
        "account_number": None,
        "date": None,
        "usage_kwh": None,
        "amount_due": None,
    }

    # Account number
    acc_match = re.search(r"account\s*(?:number|no\.?|#)?[:\s]+([\w-]+)", text, re.IGNORECASE)
    if acc_match:
        fields["account_number"] = acc_match.group(1).strip()

    # Billing date
    date_match = re.search(
        r"(?:billing\s+)?date[:\s]+(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        text, re.IGNORECASE
    )
    if date_match:
        fields["date"] = date_match.group(1).strip()

    # Usage in kWh
    usage_match = re.search(r"usage[:\s]+([\d,]+)\s*kwh", text, re.IGNORECASE)
    if usage_match:
        try:
            fields["usage_kwh"] = float(usage_match.group(1).replace(",", ""))
        except ValueError:
            pass

    # Amount due
    amount_match = re.search(
        r"amount\s+due[:\s]+\$?([\d,]+\.?\d*)", text, re.IGNORECASE
    )
    if amount_match:
        try:
            fields["amount_due"] = float(amount_match.group(1).replace(",", ""))
        except ValueError:
            pass

    return fields


def process_folder(folder_path: str) -> dict:
    """
    Process all PDFs in a folder.
    Returns a dict mapping filename → classification + extracted fields.
    """
    folder = Path(folder_path)
    results = {}

    doc_files = sorted(
        list(folder.glob("*.pdf")) + list(folder.glob("*.txt"))
    )
    if not doc_files:
        print(f"[WARNING] No PDF or text files found in {folder_path}")
        return results

    for doc_path in doc_files:
        filename = doc_path.name
        print(f"  Processing: {filename}")

        text = extract_text_from_file(str(doc_path))
        doc_class = classify_document(text, filename)

        record = {"class": doc_class, "raw_text": text}

        if doc_class == "Invoice":
            record.update(extract_invoice_fields(text))
        elif doc_class == "Resume":
            record.update(extract_resume_fields(text))
        elif doc_class == "Utility Bill":
            record.update(extract_utility_fields(text))

        # Remove raw_text from final output (keep for search index)
        results[filename] = record

    return results


def save_output(results: dict, output_path: str):
    """Save results to output.json, excluding raw_text."""
    clean = {}
    for fname, data in results.items():
        clean[fname] = {k: v for k, v in data.items() if k != "raw_text"}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)
    print(f"\n[OK] Results saved to: {output_path}")
