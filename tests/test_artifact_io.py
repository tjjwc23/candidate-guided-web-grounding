"""Artifact bytes and bounded file hashing survive IO consolidation."""
import hashlib
import json
from pathlib import Path

import pytest

from artifacts import file_digest; from artifacts import write_json


@pytest.mark.parametrize('newline,indent', [(False,2),(True,2),(False,None)])
def test_writer_preserves_explicit_artifact_format(tmp_path, newline, indent):
    value = {'中文': [1, 0.125, None, False], 'nested': {'text': 'a\nb'}}
    path = tmp_path / 'nested/artifact.json'
    write_json(path, value, newline=newline, indent=indent)
    assert path.read_bytes() == (json.dumps(value, ensure_ascii=False, indent=indent)
                                + ('\n' if newline else '')).encode('utf-8')
    assert not path.with_suffix('.json.tmp').exists()


def test_hash_streams_multiple_blocks_without_read_bytes(tmp_path, monkeypatch):
    data = bytes(range(256)) * 9000
    path = tmp_path / 'weights.bin'
    path.write_bytes(data)
    def forbidden(*args, **kwargs):
        raise AssertionError('Hashing must not load the complete file')
    monkeypatch.setattr(Path, 'read_bytes', forbidden)
    assert file_digest(path) == hashlib.sha256(data).hexdigest()


def test_atomic_serialization_failure_keeps_previous_artifact(tmp_path):
    path = tmp_path / 'record.json'
    path.write_bytes(b'previous')
    with pytest.raises(TypeError):
        write_json(path, {'unsupported': object()}, atomic=True)
    assert path.read_bytes() == b'previous'
