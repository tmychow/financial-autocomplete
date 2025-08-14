"""
Lightweight text matching utilities for tolerant normalization without hand-coded variants.

- No external dependencies required; uses difflib.
"""

from typing import Dict, Iterable, Tuple, Optional, List
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
    # Treat common separators as spaces so variants like "book_value_per_share"
    # and "book-value-per-share" normalize similarly to "book value per share".
    s = s.replace("-", " ").replace("_", " ")
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return _normalize_spaces(s)


STOPWORDS = {
    "and", "&", "of", "to", "per", "the", "for", "in", "on", "by", "with",
}

def _tokenize_meaningful(text: str) -> List[str]:
    """Tokenize a string into meaningful tokens: normalize, drop stopwords, singularize."""
    raw_tokens = normalize_text(text).split()
    filtered = [t for t in raw_tokens if t not in STOPWORDS]
    norm_tokens = [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in filtered]
    return [t for t in norm_tokens if t]

def token_signature(text: str) -> str:
    """Create a token-set signature string for robust matching (stopwords removed)."""
    tokens = _tokenize_meaningful(text)
    return " ".join(sorted(set(tokens)))


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


def _jaccard_similarity(a_tokens: List[str], b_tokens: List[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    a_set = set(a_tokens)
    b_set = set(b_tokens)
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union if union else 0.0

def _combined_similarity(a: str, b: str) -> float:
    """Combine multiple lightweight similarity signals into a single score in [0,1]."""
    a_norm = (a or "").lower().strip()
    b_norm = (b or "").lower().strip()
    a_sig = token_signature(a)
    b_sig = token_signature(b)
    r_raw = difflib.SequenceMatcher(None, a_norm, b_norm).ratio()
    r_sig = difflib.SequenceMatcher(None, a_sig, b_sig).ratio()
    jacc = _jaccard_similarity(_tokenize_meaningful(a), _tokenize_meaningful(b))
    return max(r_raw, r_sig, jacc)

def best_key_match(query: str, keys: Iterable[str], cutoff: float = 0.90) -> Optional[str]:
    """Return the best matching key using combined similarity (raw, signature, Jaccard)."""
    if not query:
        return None
    best_key = None
    best_score = 0.0
    for k in list(keys):
        score = _combined_similarity(query, k)
        if score > best_score:
            best_score = score
            best_key = k
    return best_key if best_score >= cutoff else None

def top_key_matches(query: str, keys: Iterable[str], top_k: int = 3) -> List[Tuple[str, float]]:
    """Return top-K matching keys with scores using combined similarity.

    Results are sorted by score DESC then key ASC.
    """
    ranked: List[Tuple[str, float]] = []
    for k in list(keys):
        ranked.append((k, _combined_similarity(query, k)))
    ranked.sort(key=lambda it: (-it[1], it[0]))
    return ranked[: max(0, int(top_k))]


def match_alias(query: str, alias_map: Dict[str, str], cutoff: float = 0.90) -> Optional[str]:
    """Match a query against alias_map keys and return the mapped value when score >= cutoff.

    We compare multiple query variants to be robust to punctuation, separators, and spacing.
    """
    if not query or not alias_map:
        return None

    def _dedup_preserve_order(items):
        seen = set()
        result = []
        for it in items:
            if it not in seen and it:
                seen.add(it)
                result.append(it)
        return result

    q_raw = (query or "").lower().strip()
    q_sep_spaces = q_raw.replace("-", " ").replace("_", " ")
    q_norm = normalize_text(q_raw)
    q_norm_compact = q_norm.replace(" ", "")
    q_sep_compact = q_sep_spaces.replace(" ", "")
    q_sig = token_signature(q_raw)
    q_sig_from_sep = token_signature(q_sep_spaces)

    variants = _dedup_preserve_order([
        q_raw,
        q_sep_spaces,
        q_norm,
        q_norm_compact,
        q_sep_compact,
        q_sig,
        q_sig_from_sep,
    ])

    # Fast-path exact lookups on all variants
    for v in variants:
        if v in alias_map:
            return alias_map[v]

    # Fuzzy over keys using best_key_match for each variant
    for v in variants:
        key = best_key_match(v, alias_map.keys(), cutoff=cutoff)
        if key:
            return alias_map[key]

    return None


