"""data / dom."""

from __future__ import annotations

import copy
import hashlib
import json
import re

from lxml import etree

from data.candidates import normalize_candidate

ATTR_ORDER = [
    "role",
    "aria_role",
    "type",
    "alt",
    "aria_description",
    "aria_label",
    "label",
    "title",
    "name",
    "text_value",
    "value",
    "placeholder",
    "input_checked",
    "input_value",
    "option_selected",
    "class",
]

SKIP_VALUES = {"hidden", "none", "presentation", "null", "undefined"}


def _extract_attr_string(node, max_value_length=5, max_length=20):
    seen = set()
    parts = []
    for attr in ATTR_ORDER:
        if attr in node.attrib and node.attrib[attr] is not None:
            value = node.attrib[attr].lower()
            if value in SKIP_VALUES or value.startswith("http"):
                continue
            value = " ".join(v for v in value.split() if len(v) < 15)
            value = " ".join(value.split()[:max_value_length])
            if value and value not in seen:
                seen.add(value)
                parts.append(value)
    return " ".join(parts[:max_length])


def get_tree_repr(
    tree,
    max_value_length=5,
    max_length=20,
    id_mapping=None,
    copy_tree=True,
):
    """Get string representation of DOM tree."""
    if id_mapping is None:
        id_mapping = {}
    if isinstance(tree, str):
        tree = etree.fromstring(tree)
    elif copy_tree:
        tree = copy.deepcopy(tree)

    for node in tree.xpath("//*"):
        if node.tag != "text":
            if "backend_node_id" in node.attrib:
                if node.attrib["backend_node_id"] not in id_mapping:
                    id_mapping[node.attrib["backend_node_id"]] = len(id_mapping)
                node.attrib["backend_node_id"] = str(
                    id_mapping[node.attrib["backend_node_id"]]
                )
            attr_values = _extract_attr_string(node, max_value_length, max_length)
            uid = node.attrib.get("backend_node_id", "")
            node.attrib.clear()
            if uid:
                node.attrib["id"] = uid
            if attr_values:
                node.attrib["meta"] = attr_values
        else:
            node.text = " ".join(node.text.split()[:max_length])

    tree_repr = etree.tostring(tree, encoding="unicode")
    tree_repr = tree_repr.replace('"', " ")
    tree_repr = (
        tree_repr.replace("meta= ", "").replace("id= ", "id=").replace(" >", ">")
    )
    tree_repr = re.sub(r"<text>(.*?)</text>", r"\1", tree_repr)

    tree_repr = tree_repr.replace("/>", "$/$>")
    tree_repr = re.sub(r"</(.+?)>", r")", tree_repr)
    tree_repr = re.sub(r"<(.+?)>", r"(\1", tree_repr)
    tree_repr = tree_repr.replace("$/$", ")")

    html_escape_table = [
        ("&quot;", '"'),
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&nbsp;", " "),
        ("&ndash;", "-"),
        ("&rsquo;", "'"),
        ("&lsquo;", "'"),
        ("&ldquo;", '"'),
        ("&rdquo;", '"'),
        ("&#39;", "'"),
        ("&#40;", "("),
        ("&#41;", ")"),
    ]
    for key, value in html_escape_table:
        tree_repr = tree_repr.replace(key, value)

    tree_repr = re.sub(r"\s+", " ", tree_repr).strip()
    return tree_repr, id_mapping


def fast_get_node_repr(node, max_value_length=5, max_length=20):
    if node.tag == "text":
        return node.text if node.text else ""

    attr_values = _extract_attr_string(node, max_value_length, max_length)
    uid = node.attrib.get("backend_node_id", "")

    parts = [node.tag]
    if uid:
        parts.append(f"id={uid}")
    if attr_values:
        parts.append(f"meta={attr_values}")

    return " ".join(parts)


def build_node_cache(dom_tree):
    """Build backend_node_id -> node cache for fast candidate lookup."""
    node_cache = {}
    if dom_tree is None:
        return node_cache
    for node in dom_tree.xpath("//*[@backend_node_id]"):
        bid = node.attrib.get("backend_node_id")
        if bid:
            node_cache[str(bid)] = node
    return node_cache


def build_candidate_ids(backend_node_ids):
    """Disambiguate repeated backend IDs without consulting labels."""
    counts = {}
    candidate_ids = []
    for index, backend_node_id in enumerate(backend_node_ids):
        base_id = str(backend_node_id).strip() or f"cand_{index}"
        count = counts.get(base_id, 0)
        counts[base_id] = count + 1
        candidate_ids.append(base_id if count == 0 else f"{base_id}__{count}")
    return candidate_ids


def canonical_candidates(candidates):
    """Return a deterministic label-independent candidate order."""

    def key(candidate):
        node_id = str(candidate.get("backend_node_id", ""))
        content = {k: v for k, v in candidate.items() if k != "label"}
        return (
            hashlib.sha256(node_id.encode()).digest(),
            node_id,
            json.dumps(content, sort_keys=True, ensure_ascii=False),
        )

    return sorted(candidates, key=key)


def candidate_order_identity(sample):
    """Hash the ordered backend-node sequence consumed by proposal indices."""
    values = list(map(str, sample["backend_node_ids"]))
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest()


def _format_non_dom_candidate(candidate):
    target_parts = []
    for key in ("text",):
        value = candidate.get(key)
        if value:
            target_parts.append(str(value).strip())
            break
    for key in ("element_type", "tag"):
        value = candidate.get(key)
        if value:
            target_parts.append(f"type={value}")
            break
    bbox = candidate.get("bbox")
    if bbox:
        target_parts.append("bbox=" + ",".join(f"{float(v):.1f}" for v in bbox))
    source = candidate.get("source")
    if source:
        target_parts.append(f"source={source}")
    target = " ".join(part for part in target_parts if part).strip()
    return f"ancestors: \ntarget: {target}"


def fast_format_candidate(
    dom_tree,
    candidate,
    node_cache=None,
    already_normalized=False,
):
    """Format candidate element with ancestors and subtree context."""
    normalized = candidate if already_normalized else normalize_candidate(candidate)
    if normalized is None:
        return "ancestors: \ntarget: "

    bid = normalized.get("backend_node_id", "")
    if not bid:
        return _format_non_dom_candidate(normalized)

    if node_cache:
        c_node = node_cache.get(bid)
        if c_node is None:
            return _format_non_dom_candidate(normalized)
    else:
        if dom_tree is None:
            return _format_non_dom_candidate(normalized)
        found = dom_tree.xpath(f'//*[@backend_node_id="{bid}"]')
        if not found:
            return _format_non_dom_candidate(normalized)
        c_node = found[0]

    try:
        c_copy = copy.deepcopy(c_node)
    except Exception:
        return "ancestors: \ntarget: "

    stack = [(c_copy, 0)]
    while stack:
        node, depth = stack.pop()
        if depth >= 5:
            for child in list(node):
                node.remove(child)
        else:
            children = list(node)
            if len(children) > 50:
                for child in children[50:]:
                    node.remove(child)
                children = children[:50]
            for child in reversed(children):
                if child.tag != "text":
                    stack.append((child, depth + 1))

    subtree_repr, _ = get_tree_repr(
        c_copy,
        id_mapping={},
        copy_tree=False,
    )
    subtree_repr = " ".join(subtree_repr.split()[:100])

    curr = c_node
    path = []
    while curr is not None:
        path.append(curr)
        curr = curr.getparent()
    path = path[::-1]

    ancestor_strs = []
    for ans in path[:-1]:
        ans_str = fast_get_node_repr(ans)
        ancestor_strs.append(f"/{ans_str}")

    if path:
        real_parent = c_node.getparent()
        if real_parent is not None:
            all_children = [ch for ch in real_parent if ch.tag != "text"]
            try:
                idx = all_children.index(c_node)
            except ValueError:
                idx = 0

            start = max(0, idx - 3)
            end = min(len(all_children), idx + 4)

            for node in all_children[start:end]:
                if node != c_node:
                    sib_str = fast_get_node_repr(node)
                    ancestor_strs.append(f"/{sib_str}")

    ancestor_repr = " ".join(ancestor_strs)
    if ancestor_repr.strip():
        ancestor_repr = re.sub(r"\s*\)\s*", "", ancestor_repr)
        ancestor_repr = " ".join(ancestor_repr.split()[-50:])
    else:
        ancestor_repr = ""

    return f"ancestors: {ancestor_repr}\ntarget: {subtree_repr}"




def build_dom_candidate_record(
    dom_tree,
    candidate,
    label,
    node_cache=None,
):
    """Build a unified candidate record from a DOM-linked candidate."""
    normalized = normalize_candidate(candidate)
    if normalized is None:
        return None

    bid = normalized.get("backend_node_id", "")
    node = node_cache.get(bid) if node_cache and bid else None
    tag = normalized.get("tag", "")
    if not tag and node is not None and node.tag != "text":
        tag = str(node.tag)

    attrs = normalized.get("attributes", {})
    element_type = (
        normalized.get("element_type", "")
        or str(attrs.get("type") or attrs.get("role") or "").strip().lower()
    )
    text = fast_format_candidate(
        dom_tree,
        normalized,
        node_cache=node_cache,
        already_normalized=True,
    )
    fallback_id = re.sub(
        r"\s+",
        "_",
        f"{tag or element_type or 'candidate'}:{normalized.get('text', '')[:24]}",
    )

    return {
        "candidate_id": f"dom:{bid}" if bid else f"dom:{fallback_id}",
        "backend_node_id": bid,
        "text": text,
        "dom_text": text,
        "dom_value_text": normalized.get("text", ""),
        "label": int(label),
        "source": "dom",
        "bbox": normalized.get("bbox"),
        "bbox_normalized": normalized.get("bbox_normalized", False),
        "tag": tag,
        "element_type": element_type,
        "visible": normalized.get("visible"),
        "clickable": normalized.get("clickable"),
        "attributes": attrs,
    }


def build_dom_labeled_candidates(
    dom_tree,
    pos_candidates,
    neg_candidates,
    node_cache=None,
):
    """Build structured DOM candidates from positive and negative pools."""
    labeled = []
    for label, candidates in ((1, pos_candidates), (0, neg_candidates)):
        for candidate in candidates:
            try:
                record = build_dom_candidate_record(
                    dom_tree,
                    candidate,
                    label,
                    node_cache=node_cache,
                )
            except Exception:
                record = None
            if record is not None:
                labeled.append(record)
    return labeled
