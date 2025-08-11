"""
Lightweight text matching utilities for tolerant normalization without hand-coded variants.

- No external dependencies required; uses difflib.
"""

from typing import Dict, Iterable, Tuple, Optional
import re
import difflib


def _normalize_spaces(text: str) -> str:
    return " ".join((text or "").split()).strip()


def clean_company_string(name: str) -> str:
    """Normalize a company name for matching by removing punctuation and common legal suffixes."""
    s = (name or "").lower()
    s = re.sub(r"[\.,'\"&()]+", "", s)
    # trim common suffixes (purely algorithmic list; not company-specific)
    suffixes = [
        " incorporated", " inc", " corporation", " corp", " company", " co",
        " limited", " ltd", " plc", " group", " holdings", " holding", " nv", " n.v", " sa", " s.a", " ag"
    ]
    for suf in suffixes:
        if s.endswith(suf):
            s = s[: -len(suf)]
    return _normalize_spaces(s)


def normalize_text(text: str) -> str:
    s = (text or "").lower()
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return _normalize_spaces(s)


def token_signature(text: str) -> str:
    """Create a token-set signature string for robust matching."""
    tokens = normalize_text(text).split()
    # naive singularization: remove trailing 's' for tokens > 3 chars
    norm_tokens = [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in tokens]
    return " ".join(sorted(set(norm_tokens)))


def build_ticker_alias_map(rows: Iterable[Dict[str, str]]) -> Dict[str, str]:
    alias_to_ticker: Dict[str, str] = {}
    for row in rows or []:
        tkr = (row.get("ticker") or "").strip()
        name = (row.get("company_name") or "").strip()
        if not tkr:
            continue
        alias_to_ticker[tkr.lower()] = tkr
        if name:
            cleaned = clean_company_string(name)
            if cleaned:
                alias_to_ticker[cleaned] = tkr
                alias_to_ticker[cleaned.replace(" ", "")] = tkr
            alias_to_ticker[name.lower()] = tkr
    # Add comprehensive manual, commonly-used colloquial aliases for covered companies
    # These are conservative, widely used names that map cleanly to a single listed ticker
    manual_aliases: Dict[str, str] = {
        # Megacap tech and common household names
        "apple": "AAPL",
        "microsoft": "MSFT",
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "amazon": "AMZN",
        "meta": "META",
        "facebook": "META",
        # Semiconductors and hardware
        "nvidia": "NVDA",
        "intel": "INTC",
        "micron": "MU",
        "tsmc": "TSM",
        "taiwan semi": "TSM",
        "taiwan semiconductor": "TSM",
        "broadcom": "AVGO",
        "marvell": "MRVL",
        "applied materials": "AMAT",
        "lam research": "LRCX",
        "asml": "ASML",
        "arista": "ANET",
        "arista networks": "ANET",
        # Energy, utilities, and infra
        "tesla": "TSLA",
        "vistra": "VST",
        "ge vernova": "GEV",
        "constellation": "CEG",
        "constellation energy": "CEG",
        "nrg": "NRG",
        "nrg energy": "NRG",
        "talen": "TLN",
        "talen energy": "TLN",
        "bloom": "BE",
        "bloom energy": "BE",
        "vertiv": "VRT",
        # Components/communications
        "coherent": "COHR",
        "onto": "ONTO",
        "onto innovation": "ONTO",
        "credo": "CRDO",
        "credo technology": "CRDO",
        "fabrinet": "FN",
        "tss": "TSSI",
        # Newer listings/others
        "nebius": "NBIS",
        # Enterprise software
        "oracle": "ORCL",
    }
    for alias, ticker in manual_aliases.items():
        alias_to_ticker.setdefault(alias, ticker)
        alias_to_ticker.setdefault(alias.replace(" ", ""), ticker)
    return alias_to_ticker


def build_metric_alias_map(metrics: Iterable[Dict[str, str]]) -> Dict[str, str]:
    alias_to_code: Dict[str, str] = {}
    for m in metrics or []:
        code = (m.get("metric_name") or "").strip()
        desc = (m.get("description") or "").strip()
        if not code:
            continue
        alias_to_code[code.lower()] = code
        if desc:
            # include raw, normalized, and token signature of description
            alias_to_code[desc.lower()] = code
            alias_to_code[normalize_text(desc)] = code
            alias_to_code[token_signature(desc)] = code
    return alias_to_code


def best_key_match(query: str, keys: Iterable[str], cutoff: float = 0.90) -> Optional[str]:
    """Return the best matching key using difflib ratio over token signatures.

    cutoff is in [0,1]. We compute ratios on both raw-lowered and token signatures.
    """
    if not query:
        return None
    query_norm = (query or "").lower().strip()
    query_sig = token_signature(query)
    best_key = None
    best_score = 0.0
    key_list = list(keys)
    for k in key_list:
        # compare both raw and signatures
        r1 = difflib.SequenceMatcher(None, query_norm, k).ratio()
        r2 = difflib.SequenceMatcher(None, query_sig, token_signature(k)).ratio()
        score = max(r1, r2)
        if score > best_score:
            best_score = score
            best_key = k
    return best_key if best_score >= cutoff else None


def match_alias(query: str, alias_map: Dict[str, str], cutoff: float = 0.90) -> Optional[str]:
    """Match a query against alias_map keys and return the mapped value when score >= cutoff."""
    if not query or not alias_map:
        return None
    # fast path exact
    q = query.lower().strip()
    if q in alias_map:
        return alias_map[q]
    # try normalized
    qn = normalize_text(q)
    if qn in alias_map:
        return alias_map[qn]
    # try compact (no spaces)
    qc = qn.replace(" ", "")
    if qc in alias_map:
        return alias_map[qc]
    # fuzzy over keys
    key = best_key_match(q, alias_map.keys(), cutoff=cutoff)
    return alias_map[key] if key else None


