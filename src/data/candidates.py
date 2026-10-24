"""data / candidates."""

from __future__ import annotations

import json
import re


def _first_non_empty(mapping, keys, default=None):
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return default


def normalize_text_value(value, lowercase=False):
    text = " ".join(str(value or "").strip().split())
    return text.lower() if lowercase else text


_TIME_PATTERN = re.compile(r"(\d{1,2})\s*[:.]\s*(\d{2})\s*(AM|PM|am|pm)?")
_TIME_SPACE_PATTERN = re.compile(r"(\d{1,2})\s+(\d{2})\s*(AM|PM|am|pm)?")




def _coerce_optional_bool(value):
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def parse_candidate_bbox(raw_bbox, bbox_format="xywh"):
    """Parse a bbox payload into the internal [x, y, width, height] format."""
    if raw_bbox in (None, ""):
        return None

    bbox_format = str(bbox_format or "xywh").strip().lower()
    if bbox_format not in {"xywh", "xyxy"}:
        return None

    values = raw_bbox
    if isinstance(raw_bbox, (list, tuple)):
        if len(raw_bbox) < 4:
            return None
        values = raw_bbox[:4]
    else:
        payload = str(raw_bbox).strip()
        if not payload:
            return None
        if payload.startswith("["):
            try:
                values = json.loads(payload)
            except Exception:
                values = None
        else:
            values = [part.strip() for part in payload.split(",")]
        if not isinstance(values, (list, tuple)) or len(values) < 4:
            return None

    try:
        x, y, third, fourth = [float(v) for v in values[:4]]
    except Exception:
        return None

    if bbox_format == "xyxy":
        w = third - x
        h = fourth - y
    else:
        w = third
        h = fourth
    if w < 0 or h < 0:
        return None
    return [x, y, w, h]


def normalize_candidate(candidate, bbox_format=None):
    """Normalize raw candidate payloads into a consistent dict schema."""
    if isinstance(candidate, str):
        payload = candidate.strip()
        if not payload:
            return None
        try:
            candidate = json.loads(payload)
        except Exception:
            return None

    if not isinstance(candidate, dict):
        return None

    out = dict(candidate)
    attrs = out.get("attributes", {})
    if isinstance(attrs, str):
        payload = attrs.strip()
        if payload:
            try:
                attrs = json.loads(payload)
            except Exception:
                attrs = {}
        else:
            attrs = {}
    elif not isinstance(attrs, dict):
        attrs = {}
    out["attributes"] = attrs

    bid = _first_non_empty(out, ["backend_node_id"]) or _first_non_empty(
        attrs, ["backend_node_id"]
    )
    out["backend_node_id"] = str(bid) if bid not in (None, "") else ""

    bbox = _first_non_empty(
        out,
        ["bbox", "bounding_box_rect", "bounding_box", "box", "rect"],
    )
    if bbox in (None, ""):
        bbox = _first_non_empty(
            attrs,
            ["bbox", "bounding_box_rect", "bounding_box", "box", "rect"],
        )
    input_bbox_format = (
        str(bbox_format or out.get("bbox_format") or "xywh").strip().lower()
    )
    out["bbox"] = parse_candidate_bbox(bbox, bbox_format=input_bbox_format)
    out["bbox_format"] = "xywh"
    if "bbox_normalized" not in out:
        parsed_bbox = out["bbox"]
        out["bbox_normalized"] = bool(
            parsed_bbox and all(-1e-6 <= value <= 1.0 + 1e-6 for value in parsed_bbox)
        )
    else:
        out["bbox_normalized"] = bool(out["bbox_normalized"])

    text_value = _first_non_empty(
        out,
        ["text", "content", "label", "name", "value", "title"],
    )
    if text_value in (None, ""):
        text_value = _first_non_empty(
            attrs,
            [
                "text",
                "text_value",
                "value",
                "placeholder",
                "aria_label",
                "title",
                "name",
            ],
        )
    out["text"] = str(text_value).strip() if text_value not in (None, "") else ""

    tag = _first_non_empty(out, ["tag", "tag_name"]) or _first_non_empty(
        attrs, ["tag", "tag_name"]
    )
    out["tag"] = str(tag).strip().lower() if tag not in (None, "") else ""

    element_type = _first_non_empty(
        out, ["element_type", "type", "role"]
    ) or _first_non_empty(attrs, ["element_type", "type", "role", "aria_role"])
    out["element_type"] = (
        str(element_type).strip().lower() if element_type not in (None, "") else ""
    )

    for field in ("visible", "clickable"):
        candidate_keys = [field]
        attr_keys = [field, f"is_{field}"]
        if field == "clickable":
            candidate_keys.extend(["interactivity", "interactive"])
            attr_keys.extend(["interactivity", "interactive"])
        value = _first_non_empty(out, candidate_keys)
        if value in (None, ""):
            value = _first_non_empty(attrs, attr_keys)
        out[field] = _coerce_optional_bool(value)

    parser_conf = _first_non_empty(
        out, ["parser_conf", "confidence", "score", "probability"]
    )
    if parser_conf in (None, ""):
        parser_conf = _first_non_empty(
            attrs, ["parser_conf", "confidence", "score", "probability"]
        )
    try:
        out["parser_conf"] = (
            float(parser_conf) if parser_conf not in (None, "") else None
        )
    except Exception:
        out["parser_conf"] = None

    source = str(out.get("source", "")).strip().lower()
    out["source"] = source or "dom"
    return out
