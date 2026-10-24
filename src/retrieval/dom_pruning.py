"""retrieval / dom pruning."""

from __future__ import annotations

import copy
import re

from data.dom import get_tree_repr


def get_descendants(node, max_depth: int, current_depth: int = 0):
    """Return descendants with the depth convention used by Mind2Web."""
    if current_depth > max_depth:
        return []
    descendants = []
    for child in node:
        descendants.append(child)
        descendants.extend(get_descendants(child, max_depth, current_depth + 1))
    return descendants


def prune_tree(
    dom_tree,
    candidate_set,
    max_depth: int = 5,
    max_children: int = 50,
    sibling_window: int = 3,
):
    """Prune a DOM exactly like the official MindAct candidate formatter."""
    nodes_to_keep = set()
    for candidate_id in candidate_set:
        candidate_node = dom_tree.xpath(f'//*[@backend_node_id="{candidate_id}"]')[0]
        nodes_to_keep.add(candidate_node.attrib["backend_node_id"])
        nodes_to_keep.update(
            node.attrib.get("backend_node_id", "")
            for node in candidate_node.xpath("ancestor::*")
        )
        nodes_to_keep.update(
            node.attrib.get("backend_node_id", "")
            for node in get_descendants(candidate_node, max_depth)[:max_children]
        )
        parent = candidate_node.getparent()
        if parent is not None:
            siblings = [node for node in parent.getchildren() if node.tag != "text"]
            sibling_index = siblings.index(candidate_node)
            nodes_to_keep.update(
                node.attrib.get("backend_node_id", "")
                for node in siblings[
                    max(0, sibling_index - sibling_window) : sibling_index
                    + sibling_window
                    + 1
                ]
            )

    new_tree = copy.deepcopy(dom_tree)
    for node in new_tree.xpath("//*")[::-1]:
        if node.tag != "text":
            is_keep = node.attrib.get("backend_node_id", "") in nodes_to_keep
            is_candidate = node.attrib.get("backend_node_id", "") in candidate_set
        else:
            parent_id = node.getparent().attrib.get("backend_node_id", "")
            is_keep = parent_id in nodes_to_keep
            is_candidate = parent_id in candidate_set
        if not is_keep and node.getparent() is not None:
            node.getparent().remove(node)
        else:
            if not is_candidate or node.tag == "text":
                node.attrib.pop("backend_node_id", None)
            if (
                len(node.attrib) == 0
                and not any(child.tag == "text" for child in node.getchildren())
                and node.getparent() is not None
                and node.tag != "text"
                and len(node.getchildren()) <= 1
            ):
                for child in node.getchildren():
                    node.addprevious(child)
                node.getparent().remove(node)
    return new_tree


def format_candidate(dom_tree, candidate) -> str:
    """Format one DOM candidate exactly as MindAct candidate generation does."""
    node_tree = prune_tree(dom_tree, [candidate["backend_node_id"]])
    candidate_node = node_tree.xpath("//*[@backend_node_id]")[0]
    if candidate_node.getparent() is not None:
        candidate_node.getparent().remove(candidate_node)
        ancestor_repr, _ = get_tree_repr(node_tree, id_mapping={})
    else:
        ancestor_repr = ""
    subtree_repr, _ = get_tree_repr(candidate_node, id_mapping={})
    subtree_repr = " ".join(subtree_repr.split()[:100]) if subtree_repr.strip() else ""
    if ancestor_repr.strip():
        ancestor_repr = re.sub(r"\s*\(\s*", "/", ancestor_repr)
        ancestor_repr = re.sub(r"\s*\)\s*", "", ancestor_repr)
        ancestor_repr = " ".join(ancestor_repr.split()[-50:])
    else:
        ancestor_repr = ""
    return f"ancestors: {ancestor_repr}\ntarget: {subtree_repr}"
