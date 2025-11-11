"""
scrape_gemini_rate_limits.py

Usage:
    from scrape_gemini_rate_limits import scrape_gemini_rate_limits

    url = "https://ai.google.dev/gemini-api/docs/rate-limits#current-rate-limits"
    records = scrape_gemini_rate_limits(url, to_json_path="gemini_rate_limits.json")
"""

from collections import defaultdict
from pathlib import Path
import json
from typing import List, Dict, Optional, Tuple
import re
import requests
from bs4 import BeautifulSoup, Tag


def _text_to_number(s: Optional[str]) -> Optional[int]:
    """
    Convert text like '1,000' or '1000' to int. Treat '*', '—', '', None as None.
    If the cell contains words like 'no limit' return None.
    """
    if s is None:
        return None
    s = s.strip()
    if s == "" or s in {"*", "—", "-", "No limit", "no limit", "n/a", "N/A"}:
        return None
    # extract first number-like substring
    m = re.search(r"(\d{1,3}(?:[,.\s]\d{3})*|\d+)", s)
    if not m:
        return None
    num_text = m.group(1)
    # remove commas/spaces
    num_text = re.sub(r"[,\s]", "", num_text)
    try:
        return int(num_text)
    except ValueError:
        return None


def _flatten_header(header_rows: List[List[str]] | List[str]) -> List[str]:
    """
    Flatten an HTML table header (multiple <tr> in <thead> or first rows in tbody)
    into column names by taking the last non-empty label in each column stack.
    """
    if not header_rows:
        return []
    n_cols = max(len(r) for r in header_rows)
    cols = []
    for col_idx in range(n_cols):
        # take last non-empty entry in that column stack
        label = None
        for row in header_rows:
            if col_idx < len(row) and row[col_idx].strip():
                label = row[col_idx].strip()
        cols.append(label if label is not None else f"col_{col_idx}")
    return cols


def _extract_table_rows(table: Tag) -> Tuple[List[str], List[List[str]]]:
    """
    Given a BeautifulSoup <table> tag, extract header (list of rows -> cells)
    and body rows (list of cell-lists). Try to handle <thead>, or multiple leading
    <tr>s with <th> content. Return (header_rows, body_rows)
    """
    # find header rows
    header_rows = []
    thead = table.find("thead")
    if thead:
        for tr in thead.find_all("tr"):
            header_rows.append(
                [td.get_text(" ", strip=True) for td in tr.find_all(["th", "td"])]
            )
    else:
        # fallback: find first consecutive <tr>s with <th>, or take first 1-2 rows if no <th>
        trs = table.find_all("tr")
        if not trs:
            return [], []
        # collect initial rows that contain <th>
        for tr in trs[:3]:
            if tr.find("th"):
                header_rows.append(
                    [td.get_text(" ", strip=True) for td in tr.find_all(["th", "td"])]
                )

    # extract body rows after header rows (guess)
    body_rows = []
    # If <tbody> present, use that
    tbody = table.find("tbody")
    if tbody:
        trs_body = tbody.find_all("tr")
    else:
        trs_body = table.find_all("tr")
        # if header rows were taken from the earliest trs, skip them
        if header_rows:
            header_count = len(header_rows)
            trs_body = trs_body[header_count:]
    for tr in trs_body:
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        if not any(cells):
            continue
        body_rows.append(cells)
    return header_rows, body_rows


def _find_col_index(possible_names: List[str], cols_norm: List[str]):
    for name in possible_names:
        for i, c in enumerate(cols_norm):
            if c and name in c:
                return i
    return None


def scrape_gemini_rate_limits(
    url: Optional[str] = None,
    session: Optional[requests.Session] = None,
    to_json_path: Optional[str | Path] = (Path("data") / "gemini_rate_limits.json"),
) -> Dict[str, Dict[str, Dict[str, str | int | float | None]]]:
    """Scrape the Gemini API rate limits documentation page.

    This function fetches the content from the specified URL, parses the HTML
    to identify tables containing rate limit information, and extracts relevant
    data such as model names, RPM (requests per minute), TPM (tokens per minute),
    and RPD (requests per day) for different usage tiers.

    Args:
        url (Optional[str]): The URL of the Gemini API rate limits documentation page.
                             Defaults to "https://ai.google.dev/gemini-api/docs/rate-limits#current-rate-limits".
        session (Optional[requests.Session]): An optional requests session object to use for the HTTP request.
                                              If None, a new session will be created.
        to_json_path (Optional[str | Path]): The file path to save the scraped data as a JSON file.
                                             If None, the data will not be saved to a file.
                                             Defaults to 'data/gemini_rate_limits.json'.

    Returns:
        Dict[str, Dict[str, Dict[str, str | int | float | None]]]: A dictionary where top-level keys
        represent usage tiers (e.g., 'Free Tier', 'Tier 1', 'Tier 2', 'Tier 3'). Each tier
        contains a dictionary of models, with model names (normalized) as keys.
        Each model entry is a dictionary with 'name', 'RPM', 'TPM', and 'RPD'.

    Example:
        {
            'Free Tier': {
                'gemini-2.5-pro': {'name': 'Gemini 2.5 Pro', 'RPM': 2, 'TPM': 125000, 'RPD': 50},
                'gemini-1.5-flash': {'name': 'Gemini 1.5 Flash', 'RPM': 15, 'TPM': 1000000, 'RPD': 1000}
            },
            'Tier 1': {
                'gemini-2.5-pro': {'name': 'Gemini 2.5 Pro', 'RPM': 10, 'TPM': 500000, 'RPD': 250}
            }
        }
    """
    url = url or "https://ai.google.dev/gemini-api/docs/rate-limits#current-rate-limits"
    s = session or requests.Session()
    resp = s.get(url, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Heuristic: find all tables near headings that mention 'rate' or 'limits'
    tables = soup.find_all("table")
    records = defaultdict(dict)

    for table in tables:
        # try to determine a friendly title for the table:
        # nearest previous <h1>-<h4> or <p> containing 'rate' or 'limit'
        table_title = None
        prev = table.find_previous(
            lambda tag: tag.name in ["h1", "h2", "h3", "h4", "p"]
        )
        if prev:
            txt = prev.get_text(" ", strip=True)
            if re.search(r"rate|limit|free", txt, re.I):
                table_title = txt
            else:
                # if prev is h2/h3 etc, still use as title even if no 'rate' word
                if prev.name in ["h1", "h2", "h3", "h4"]:
                    table_title = txt

        header_rows, body_rows = _extract_table_rows(table)

        if not body_rows:
            continue

        # flatten header into column names
        cols = (
            _flatten_header(header_rows)
            if header_rows
            else [f"c{i}" for i in range(len(body_rows[0]))]
        )

        # normalize column names: lower-case and strip
        cols_norm = [c.lower() if c else "" for c in cols]

        # try to detect which column corresponds to model/tier/RPM/TPM/RPD
        # Common column header tokens
        model_idx = _find_col_index(
            ["model", "model name", "model/version", "api model"], cols_norm
        )
        tier_idx = _find_col_index(["tier", "usage tier", "plan"], cols_norm)
        rpm_idx = _find_col_index(
            ["rpm", "requests per minute", "requests/minute", "requests/min"], cols_norm
        )
        tpm_idx = _find_col_index(
            ["tpm", "tokens per minute", "tokens/minute", "tokens/min"], cols_norm
        )
        rpd_idx = _find_col_index(
            ["rpd", "requests per day", "requests/day", "daily"], cols_norm
        )

        # fallback heuristics: if not found, try to guess positions by looking for numeric-looking columns
        if model_idx is None:
            # assume first text-like column is model
            for i, c in enumerate(cols_norm):
                if "model" in c or "name" in c or i == 0:
                    model_idx = i
                    break

        # iterate body rows and produce records
        for row in body_rows:
            # pad row to same length as cols
            if len(row) < len(cols):
                row = row + [""] * (len(cols) - len(row))

            model = (
                row[model_idx].strip()
                if model_idx is not None and model_idx < len(row)
                else None
            )

            if model and isinstance(model, str):
                rec: dict[str, str | int | float | None] = {
                    "model": model,
                }

                rpm = (
                    _text_to_number(row[rpm_idx])
                    if rpm_idx is not None and rpm_idx < len(row)
                    else None
                )
                tpm = (
                    _text_to_number(row[tpm_idx])
                    if tpm_idx is not None and tpm_idx < len(row)
                    else None
                )
                rpd = (
                    _text_to_number(row[rpd_idx])
                    if rpd_idx is not None and rpd_idx < len(row)
                    else None
                )

                # if RPM/TPM/RPD not found by index, try to extract numbers by searching row cell-by-cell for common suffixes
                if rpm is None:
                    for i, cell in enumerate(row):
                        if re.search(
                            r"rpm|requests per minute|req/min|requests/min",
                            cols_norm[i] if i < len(cols_norm) else "",
                        ):
                            rpm = _text_to_number(cell)
                if tpm is None:
                    for i, cell in enumerate(row):
                        if re.search(
                            r"tpm|tokens per minute|tokens/min",
                            cols_norm[i] if i < len(cols_norm) else "",
                        ):
                            tpm = _text_to_number(cell)

                # Add extra normalized model name if present (strip footnote markers)
                model_normalized = re.sub(r"\[\d+\]|\*\s*$", "", model).strip()
                model_normalized = model_normalized.strip().lower().replace(" ", "-")

                if all(
                    [
                        isinstance(rpm, int | float),
                        isinstance(tpm, int | float),
                        isinstance(rpd, int | float),
                    ]
                ):
                    records[table_title].update(
                        {
                            model_normalized: {
                                "name": model,
                                "RPM": rpm,
                                "TPM": tpm,
                                "RPD": rpd,
                            }
                        }
                    )

    if records and to_json_path:
        with open(to_json_path, "w") as f:
            json.dump(records, f, indent=2)

    return records

    # Post-process: drop duplicates (same model+tier) keeping first
    # seen = set()
    # deduped = []
    # for r in records:
    #     key = (r.get("model_normalized") or r.get("model"), r.get("tier"))
    #     if key in seen:
    #         continue
    #     seen.add(key)
    #     deduped.append(r)

    # Optionally export
    # if (to_json_path) and deduped:
    #     df = pd.DataFrame(deduped)
    #     # flatten lists in raw_row to string for CSV
    #     if "raw_row" in df.columns:
    #         df["raw_row"] = df["raw_row"].apply(lambda rr: " | ".join(rr) if isinstance(rr, list) else rr)

    #     if to_json_path:
    #         df.to_json(to_json_path, orient="records", indent=2)

    # with open(to_json_path, 'w') as f:
    #     json.dump(deduped, f, indent=2)

    # return deduped


# Example quick-run if module executed directly
if __name__ == "__main__":
    url = "https://ai.google.dev/gemini-api/docs/rate-limits#current-rate-limits"
    print("Scraping:", url)
    records = scrape_gemini_rate_limits(url, to_json_path="gemini_rate_limits.json")
    print(f"Found {len(records)} rate-limit rows. Saved CSV/JSON.")
    # print first few
    for k, v in records.items():
        print(k, v)
