"""Heuristic filter for an everyday-English training mix: drop maths records
(OpenMathInstruct-style Q/A, LaTeX, equations) and list-like records
(Wikipedia category footers, tables of names and years)."""
import re

_EQ = re.compile(rb"\d\s*[+\-*/=]\s*\d")
_MATHQ = re.compile(rb"\b(Let|Find|Compute|Calculate|Solve|Determine|Simplify|Evaluate)\b")


def is_math(t: bytes) -> bool:
    if t.count(b"$") >= 2 or b"\\" in t or b"boxed" in t or b"^" in t:
        return True
    if t.startswith(b"Q: ") or b"\nA: " in t or b"\nQ: " in t:
        return True
    if len(_EQ.findall(t)) >= 2:
        return True
    if _MATHQ.search(t) and b"=" in t:
        return True
    return False


def is_list(t: bytes) -> bool:
    if b"Living people" in t or b" births\n" in t or b" deaths\n" in t:
        return True
    lines = [l for l in t.split(b"\n") if l.strip()]
    if len(lines) >= 5:
        prose = sum(1 for l in lines if l.rstrip()[-1:] in (b".", b"!", b"?", b'"'))
        if prose / len(lines) < 0.2:
            return True
    return False


def keep(t: bytes) -> bool:
    return not (is_math(t) or is_list(t))


_CONV = [b" I ", b" I'", b" you", b" You", b" we ", b" We ", b"'m ", b"n't ",
         b"!", b"?", b" my ", b" My ", b" our ", b" your ", b" me ", b"'ll ",
         b"'re ", b"'ve ", b" us "]


def is_conversational(t: bytes, need: int = 3) -> bool:
    """Everyday first/second-person writing: blogs, reviews, messages."""
    return keep(t) and sum(1 for m in _CONV if m in t) >= need
