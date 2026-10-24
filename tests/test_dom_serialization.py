from lxml import etree

from data.dom import get_tree_repr
from retrieval.dom_pruning import format_candidate


def test_attribute_word_limit_is_not_a_character_limit():
    tree = etree.fromstring(
        '<button backend_node_id="7" aria_label="book room now please today"/>'
    )
    representation, _ = get_tree_repr(tree)
    assert "book room now please today" in representation


def test_mindact_candidate_uses_shared_dom_serialization():
    tree = etree.fromstring(
        '<body backend_node_id="1"><button backend_node_id="7" '
        'aria_label="book room now please today"/></body>'
    )
    representation = format_candidate(tree, {"backend_node_id": "7"})
    assert "book room now please today" in representation
