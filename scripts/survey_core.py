"""Deterministic survey helpers. No I/O, deletions, statistics API, or network.

Input lists represent individual responses, not aggregated frequency weights.
Normalize library-specific missing scalars explicitly before using this module.
All test data shipped with this skill are synthetic.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path


class SurveyError(ValueError):
    """A validation/decision gate; do not silently coerce or drop the input."""


def is_missing(value):
    return value is None or (isinstance(value, str) and not value.strip()) or (
        isinstance(value, Real) and not isinstance(value, bool) and math.isnan(value)
    )


def _number(value, *, finite=True):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SurveyError("Expected a numeric scalar, not a code string or boolean")
    value = float(value)
    if finite and not math.isfinite(value):
        raise SurveyError("Non-finite numeric value")
    return value


def _mask(mask, n):
    if len(mask) != n or any(type(v) is not bool for v in mask):
        raise SurveyError("Eligibility must be an aligned, fully resolved bool list")


def unique_field_map(dictionary, requests, raw_headers, *, standard_key, raw_key):
    """Map canonical names to unique actual raw headers from permitted records.

    requests = {canonical_name: exact_standard_name}. No positional matching.
    Caller is responsible for reading only authorized dictionary columns.
    """
    if len(raw_headers) != len(set(raw_headers)):
        raise SurveyError("Duplicate raw headers")
    result = {}
    for canonical, standard in requests.items():
        matches = [r for r in dictionary if r.get(standard_key) == standard]
        if len(matches) != 1:
            raise SurveyError(f"{canonical}: expected one dictionary match; got {len(matches)}")
        raw = matches[0].get(raw_key)
        if raw not in raw_headers:
            raise SurveyError(f"{canonical}: mapped raw field does not exist")
        result[canonical] = raw
    if len(set(result.values())) != len(result):
        raise SurveyError("Multiple canonical targets map to the same raw field")
    return result


_SEC = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*(?:秒|s|sec|secs|second|seconds)", re.I)
_MIN = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*(?:分钟|分|min|mins|minute|minutes)", re.I)
_BOTH = re.compile(
    r"(\d+)\s*(?:分钟|分|min|mins|minute|minutes)\s*"
    r"(\d+(?:\.\d+)?)\s*(?:秒|s|sec|secs|second|seconds)", re.I
)
_BARE = re.compile(r"[+-]?\d+(?:\.\d+)?")


def parse_duration(value, *, bare_numeric_unit=None):
    """Return seconds/None; unknown or ambiguous formats raise SurveyError.

    Bare numeric values require explicit unit 'seconds' or 'minutes'. Negative
    and zero seconds remain numeric for the separate quality/decision gate.
    Colon times and spreadsheet time fractions are never guessed.
    """
    if bare_numeric_unit not in (None, "seconds", "minutes"):
        raise SurveyError("Unknown bare numeric unit")
    if is_missing(value):
        return None
    if isinstance(value, Real) and not isinstance(value, bool):
        if bare_numeric_unit is None:
            raise SurveyError("Bare numeric duration requires a confirmed unit")
        return _number(value) * (60 if bare_numeric_unit == "minutes" else 1)
    if not isinstance(value, str):
        raise SurveyError("Unsupported duration type")
    text = value.strip()
    match = _BOTH.fullmatch(text)
    if match:
        minutes, seconds = map(float, match.groups())
        if seconds >= 60:
            raise SurveyError("Seconds component in a minute-second pair must be below 60")
        result = minutes * 60 + seconds
    elif match := _SEC.fullmatch(text):
        result = float(match.group(1))
    elif match := _MIN.fullmatch(text):
        result = float(match.group(1)) * 60
    elif _BARE.fullmatch(text) and bare_numeric_unit:
        result = float(text) * (60 if bare_numeric_unit == "minutes" else 1)
    else:
        raise SurveyError("Unrecognized or ambiguous duration format")
    if not math.isfinite(result):
        raise SurveyError("Duration is not finite")
    return result


def audit_durations(values, record_ids, *, bare_numeric_unit=None):
    if len(values) != len(record_ids):
        raise SurveyError("Duration values and identifiers are not aligned")
    records = []
    for row, (raw, record_id) in enumerate(zip(values, record_ids), start=1):
        try:
            seconds = parse_duration(raw, bare_numeric_unit=bare_numeric_unit)
            status = "missing" if seconds is None else "negative" if seconds < 0 else "zero" if seconds == 0 else "ok"
            error = None
        except SurveyError as exc:
            seconds, status, error = None, "unparseable", str(exc)
        records.append(dict(position=row, record_id=record_id, raw=raw,
                            seconds=seconds, status=status, error=error))
    counts = Counter(r["status"] for r in records)
    parsed = sum(counts[s] for s in ("ok", "negative", "zero"))
    nonempty = sum(not is_missing(v) for v in values)
    assert parsed + counts["unparseable"] == nonempty
    assert sum(counts.values()) == len(values)
    return dict(records=records, counts=dict(counts), parsed=parsed, nonempty=nonempty)


def require_duration_ready(audit, *, allow_missing=False, allow_nonpositive=False):
    """Stop downstream thresholds unless parsing and project policy are settled."""
    forbidden = {"unparseable"}
    if not allow_missing:
        forbidden.add("missing")
    if not allow_nonpositive:
        forbidden.update(("negative", "zero"))
    problems = [r for r in audit["records"] if r["status"] in forbidden]
    if problems:
        raise SurveyError(f"Duration gate paused: {len(problems)} records need resolution; inspect audit records")
    assert audit["parsed"] == audit["nonempty"]
    return [r["seconds"] for r in audit["records"]]


def audit_skip_path(rows, eligible, *, skip_codes_by_field, entered_values_by_field):
    """Check resolved predecessor eligibility; never change answers.

    The same code can be a whole-question skip outside the route and a valid
    unranked-item code inside it, if entered_values_by_field explicitly allows it.
    Returned position is one-based input position, not an Excel row number.
    """
    _mask(eligible, len(rows))
    fields = list(skip_codes_by_field)
    if not fields or set(fields) != set(entered_values_by_field):
        raise SurveyError("Skip and entered-value dictionaries must cover the same fields")
    issues = []
    for position, (row, entered) in enumerate(zip(rows, eligible), start=1):
        for field in fields:
            if field not in row:
                raise SurveyError(f"Missing route field: {field}")
            allowed = entered_values_by_field[field] if entered else skip_codes_by_field[field]
            if row[field] not in allowed:
                issues.append(dict(position=position, field=field, entered=entered,
                                   value=row[field], status="跳题逻辑待核查"))
    return issues


def exact_selection_score(options, correct_options, *, selected=1, unselected=0, missing_policy="error"):
    """Score an exact selected-option set, not 'contains the right choice'.

    options maps option IDs to values. missing_policy may be 'error', 'missing',
    or user-confirmed 'incorrect'. Out-of-range codes always raise.
    """
    if missing_policy not in ("error", "missing", "incorrect") or selected == unselected:
        raise SurveyError("Invalid scoring policy")
    correct = set(correct_options)
    if not options or not correct.issubset(options):
        raise SurveyError("Answer key has missing/unknown options")
    missing = False
    for value in options.values():
        if is_missing(value):
            missing = True
        elif value not in (selected, unselected):
            raise SurveyError("Out-of-range multi-select code")
    if missing:
        if missing_policy == "error":
            raise SurveyError("Missing answer needs a confirmed scoring policy")
        return None if missing_policy == "missing" else 0
    chosen = {key for key, value in options.items() if value == selected}
    return int(chosen == correct)


def multi_select_summary(rows, eligible, options, *, selected=1, unselected=0):
    """Common respondent denominator; unresolved eligible answers STOP.

    This helper does not silently choose complete-case vs item-wise exclusion.
    For confirmed item-wise denominators, prepare separate masks and record n.
    """
    _mask(eligible, len(rows))
    if not options or len(options) != len(set(options)) or selected == unselected:
        raise SurveyError("Invalid multi-select options/codes")
    active = [r for r, keep in zip(rows, eligible) if keep]
    n = len(active)
    result = []
    for option in options:
        values = []
        for row in active:
            if option not in row or is_missing(row[option]) or row[option] not in (selected, unselected):
                raise SurveyError("Eligible multi-select answer is unresolved; confirm denominator policy")
            values.append(row[option])
        count = sum(v == selected for v in values)
        result.append(dict(option=option, n=n, count=count, percent=100 * count / n if n else None))
    return result


def ranking_summary(rows, eligible, options, *, top_k, unranked_values, require_complete=True):
    """Rank-1/top-K percentages among eligible respondents, not all records."""
    _mask(eligible, len(rows))
    if not options or len(options) != len(set(options)) or type(top_k) is not int or not 1 <= top_k <= len(options):
        raise SurveyError("Invalid ranking design")
    if any(v in range(1, top_k + 1) for v in unranked_values):
        raise SurveyError("Unranked codes overlap real ranks")
    active = [r for r, keep in zip(rows, eligible) if keep]
    for row in active:
        ranks = []
        for option in options:
            if option not in row:
                raise SurveyError("Missing ranking field")
            value = row[option]
            if value in unranked_values:
                continue
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value != int(value) or not 1 <= value <= top_k:
                raise SurveyError("Invalid or unresolved rank")
            ranks.append(int(value))
        if len(ranks) != len(set(ranks)):
            raise SurveyError("Duplicate ranks in one response")
        if require_complete and set(ranks) != set(range(1, top_k + 1)):
            raise SurveyError("Incomplete ranking needs an explicit policy")
    n = len(active)
    result = []
    for option in options:
        first = sum(r[option] == 1 for r in active)
        top = sum(r[option] not in unranked_values for r in active)
        result.append(dict(option=option, n=n, first_count=first, top_count=top,
                           first_percent=100 * first / n if n else None,
                           top_percent=100 * top / n if n else None))
    return result


def exclusion_log(rows, flags, reasons, *, log_fields, excel_row_field="excel_row"):
    """Deduplicate by response, with all reasons; input flags must be real bools.

    Only explicitly selected log_fields are copied. Caller must exclude personal
    fields such as full IP, name, phone. No dataframe or workbook is mutated.
    """
    if len(rows) != len(flags) or not reasons:
        raise SurveyError("Unaligned inputs or no approved rule definitions")
    original_rows = [r.get(excel_row_field) for r in rows]
    if any(type(v) is not int or v < 2 for v in original_rows) or len(set(original_rows)) != len(rows):
        raise SurveyError("Excel data row IDs must be unique integers and cannot include the header")
    if any(not isinstance(reason, str) or not reason.strip() for reason in reasons.values()):
        raise SurveyError("Every rule must have an exclusion reason")
    invalid, logs = [], []
    counts = {rule: 0 for rule in reasons}
    for row, rule_flags in zip(rows, flags):
        if set(rule_flags) != set(reasons) or any(type(v) is not bool for v in rule_flags.values()):
            raise SurveyError("All approved rules must have resolved True/False flags")
        triggered = [rule for rule in reasons if rule_flags[rule]]
        invalid.append(bool(triggered))
        for rule in triggered:
            counts[rule] += 1
        if triggered:
            if any(field not in row for field in log_fields):
                raise SurveyError("Requested log field is missing")
            item = {field: row[field] for field in log_fields}
            item.update({excel_row_field: row[excel_row_field], "rules": "；".join(triggered),
                         "reasons": "；".join(reasons[r] for r in triggered),
                         "action": "删除无效问卷"})
            logs.append(item)
    logs.sort(key=lambda r: r[excel_row_field])
    deleted = sum(invalid)
    assert deleted == len(logs)
    assert len(rows) == (len(rows) - deleted) + len(logs)
    return dict(invalid=invalid, log=logs, before=len(rows), after=len(rows) - deleted,
                deleted=deleted, rule_counts=counts, rule_events=sum(counts.values()),
                multiple_rules=sum(sum(f.values()) > 1 for f in flags))


def quantile_linear(values, q):
    if not values or not 0 <= q <= 1:
        raise SurveyError("Quantile needs finite observations and q in [0,1]")
    values = sorted(_number(v) for v in values)
    index = (len(values) - 1) * q
    lower, upper = math.floor(index), math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def score_distribution(scores, levels):
    """A single group's full score distribution, including zero-frequency bins."""
    if not scores or not levels or len(set(levels)) != len(levels):
        raise SurveyError("Scores/levels must be nonempty and levels unique")
    values = [_number(v) for v in scores]
    if any(v not in levels for v in values):
        raise SurveyError("Score outside the configured range")
    count = Counter(values)
    n = len(values)
    return dict(n=n, bins=[dict(score=s, count=count[s], percent=100 * count[s] / n) for s in levels],
                median=quantile_linear(values, .5), q1=quantile_linear(values, .25), q3=quantile_linear(values, .75))


def chi_square_gate(observed):
    """Apply the documented conservative, user-confirmed policy; NO p-values.

    Returns zero-based positions for expected cells below 1 and 5. Does not
    silently remove zero marginals, merge groups, or substitute another test.
    """
    if not observed or not observed[0] or any(len(r) != len(observed[0]) for r in observed):
        raise SurveyError("Contingency table is empty or nonrectangular")
    table = [[_number(v) for v in row] for row in observed]
    if any(v < 0 or v != int(v) for row in table for v in row):
        raise SurveyError("Independent observed counts must be nonnegative integers")
    nr, nc = len(table), len(table[0])
    rows = [sum(row) for row in table]
    cols = [sum(table[i][j] for i in range(nr)) for j in range(nc)]
    n = sum(rows)
    expected = [[rows[i] * cols[j] / n for j in range(nc)] for i in range(nr)] if n else None
    lt1 = [(i, j) for i in range(nr) for j in range(nc) if expected[i][j] < 1] if expected else []
    lt5 = [(i, j) for i in range(nr) for j in range(nc) if expected[i][j] < 5] if expected else []
    result = dict(status="paused", method=None, correction=None, n=int(n),
                  expected=expected, cells=nr * nc, lt1_positions=lt1, lt5_positions=lt5,
                  lt1_count=len(lt1) if expected else None, lt5_count=len(lt5) if expected else None,
                  lt5_fraction=len(lt5) / (nr * nc) if expected else None,
                  pearson_rc_conditions_met=bool(nr >= 2 and nc >= 2 and expected is not None
                                                and 0 not in rows and 0 not in cols
                                                and not lt1 and 5 * len(lt5) <= nr * nc),
                  df=(nr - 1) * (nc - 1), reason=None)
    if nr < 2 or nc < 2 or not n or 0 in rows or 0 in cols:
        result["reason"] = "不可估计：类别不足、全空或存在零边际；等待人工决定"
    elif nr == nc == 2:
        if n < 40 or lt1:
            result["reason"] = "四格表 N<40 或存在 T<1；未形成正式检验结果，等待人工决定"
        else:
            result.update(status="ok", method="Yates" if lt5 else "Pearson", correction=bool(lt5))
    elif lt1 or 5 * len(lt5) > nr * nc:
        result["reason"] = "行列表存在 T<1 或超过20%的 T<5；未形成正式检验结果，等待人工决定"
    else:
        result.update(status="ok", method="Pearson", correction=False)
    return result


def format_p(value):
    if is_missing(value):
        return ""
    value = _number(value)
    if not 0 <= value <= 1:
        raise SurveyError("P-value must be between 0 and 1")
    return "<0.001" if value < .001 else f"{value:.3f}"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
