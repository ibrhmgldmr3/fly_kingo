import hashlib

import pytest

from flyhire.brain import data as brain_data


def _existing_file(tmp_path, content=b"connectome"):
    path = tmp_path / "file.bin"
    path.write_bytes(content)
    return path, hashlib.sha256(content).hexdigest()


def test_sha256_of_matches_hashlib(tmp_path):
    path, expected = _existing_file(tmp_path)
    assert brain_data.sha256_of(path) == expected


def test_existing_file_with_matching_checksum_is_not_redownloaded(tmp_path):
    path, digest = _existing_file(tmp_path)
    item = brain_data.DataFile(path.name, "http://invalid.invalid/x", digest, "test")
    assert brain_data.download_file(item, tmp_path) == path


def test_checksum_mismatch_raises(tmp_path):
    path, _ = _existing_file(tmp_path)
    item = brain_data.DataFile(path.name, "http://invalid.invalid/x", "0" * 64, "test")
    with pytest.raises(ValueError, match="checksum"):
        brain_data.download_file(item, tmp_path)


def test_manifest_is_pinned_and_checksummed():
    for item in brain_data.MANIFEST:
        assert brain_data.SHIU_COMMIT in item.url or brain_data.ANNOTATIONS_COMMIT in item.url
        assert len(item.sha256) == 64, f"{item.name} için sha256 eksik"


def test_missing_files_lists_everything_in_empty_dir(tmp_path):
    assert brain_data.missing_files(tmp_path) == [item.name for item in brain_data.MANIFEST]
