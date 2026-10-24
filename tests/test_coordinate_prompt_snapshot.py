import hashlib
import json

from PIL import Image

from grounding.coordinate_prompt import get_computer_use_system_prompt


def test_frozen_fara_prompt_snapshot_is_self_contained():
    prompt = get_computer_use_system_prompt(
        Image.new("RGB", (1280, 720)),
        {
            "patch_size": 14,
            "merge_size": 2,
            "min_pixels": 3136,
            "max_pixels": 12845056,
        },
        display_size=1000,
    )

    assert prompt["im_size"] == (1288, 728)
    assert prompt["conversation"][0]["role"] == "system"
    system_text = prompt["conversation"][0]["content"][0]["text"]
    assert "screen's resolution is 1000x1000" in system_text
    assert '"name": "computer_use"' in system_text
    snapshot = json.dumps(prompt, ensure_ascii=False, sort_keys=True).encode()
    assert hashlib.sha256(snapshot).hexdigest() == (
        "f57d563659b5d14430d0fb48b0db1e2143f3752f54468247eac7decd1032271d"
    )
