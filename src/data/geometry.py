"""data / geometry."""

from __future__ import annotations

from data.candidates import parse_candidate_bbox

DISPLAY_SIZE = 1000


def display_point_coordinate(value: float, extent: float) -> int:
    """Project an in-frame point into the valid 0..999 display coordinates."""
    return min(DISPLAY_SIZE - 1, max(0, round(value / extent * DISPLAY_SIZE)))




def document_size(dom_tree):
    """Return the DOM page width/height used by normalized OmniParser boxes."""
    if dom_tree is None:
        return None

    root = dom_tree.getroot() if hasattr(dom_tree, "getroot") else dom_tree
    if hasattr(root, "getroottree"):
        root = root.getroottree().getroot()
    attributes = getattr(root, "attrib", {})
    for key in ("bounding_box_rect", "bbox", "bounding_box", "box", "rect"):
        bbox = parse_candidate_bbox(attributes.get(key), bbox_format="xywh")
        if bbox and bbox[2] > 0.0 and bbox[3] > 0.0:
            return bbox[2], bbox[3]
    return None


def project_frame_candidates(candidates, node_cache, *, source_space):
    if source_space not in {"frame_local", "page"}:
        raise ValueError(f"Unknown bbox coordinate space: {source_space}")
    by_id = {str(c["backend_node_id"]): c for c in candidates}
    result = []
    for candidate in candidates:
        c = dict(candidate)
        if c.get("bbox_space") == "page":
            result.append(c)
            continue
        original = c.get("bbox")
        c["bbox_source"] = dict(
            space=source_space, bbox=original, normalized=bool(c.get("bbox_normalized"))
        )
        chain = []
        status = "unchanged"
        node = node_cache.get(str(c["backend_node_id"]))
        if source_space == "frame_local" and node is not None:
            for ancestor in node.iterancestors():
                if ancestor.tag not in {"iframe", "frame"}:
                    continue
                fid = ancestor.get("backend_node_id", "")
                frame = by_id.get(fid, {})
                raw = frame.get("bbox_source", {})
                box = raw.get("bbox", frame.get("bbox"))
                if box is None:
                    box = parse_candidate_bbox(ancestor.get("bounding_box_rect"))
                if not box or frame.get("bbox_normalized"):
                    status = "unresolved_frame"
                    break
                chain.append({"id": fid, "offset": list(box[:2])})
            if chain and c.get("bbox_normalized"):
                status = "unresolved_normalized_frame"
        elif source_space == "frame_local" and node is None:
            status = "unresolved_node"
        if status.startswith("unresolved"):
            c["bbox"] = None
        elif chain and original:
            c["bbox"] = [
                original[0] + sum(f["offset"][0] for f in chain),
                original[1] + sum(f["offset"][1] for f in chain),
                original[2],
                original[3],
            ]
            status = "translated"
        c["bbox_space"] = (
            "page" if not status.startswith("unresolved") else "unresolved"
        )
        c["bbox_frame_chain"] = chain
        c["bbox_projection_status"] = status
        result.append(c)
    return result
