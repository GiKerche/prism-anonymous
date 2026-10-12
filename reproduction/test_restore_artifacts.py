"""Integrity and filesystem-safety checks for downloaded artifact restoration."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from restore_artifacts import restore


class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original = b"Recorded scientific input\x00\xff" * 100
        archive = gzip.compress(self.original, mtime=0)
        parts = []
        folder = self.root / "reproduction/parts"
        folder.mkdir(parents=True)
        for i, block in enumerate([archive[:19], archive[19:]]):
            path = folder / f"part{i}.bin"
            path.write_bytes(block)
            parts.append({"path": path.relative_to(self.root).as_posix(), "bytes": len(block),
                          "sha256": hashlib.sha256(block).hexdigest()})
        self.entry = {"path": "data/artifact.npz", "bytes": len(self.original),
                      "sha256": hashlib.sha256(self.original).hexdigest(), "encoding": "gzip",
                      "archive_bytes": len(archive), "archive_sha256": hashlib.sha256(archive).hexdigest(),
                      "parts": parts}
        self.manifest = self.root / "reproduction/artifact_parts.json"
        self.save_manifest()

    def save_manifest(self):
        self.manifest.write_text(json.dumps({"version": 1, "artifacts": [self.entry]}))

    def test_exact_restoration_and_repeat(self):
        self.assertEqual(restore(self.root)["restored"], 1)
        target = self.root / self.entry["path"]
        self.assertEqual(target.read_bytes(), self.original)
        before = target.stat().st_mtime_ns
        self.assertEqual(restore(self.root)["already_verified"], 1)
        self.assertEqual(target.stat().st_mtime_ns, before)

    def test_damaged_part_cannot_publish_output(self):
        (self.root / self.entry["parts"][0]["path"]).write_bytes(b"damaged")
        with self.assertRaisesRegex(ValueError, "damaged artifact part"):
            restore(self.root)
        self.assertFalse((self.root / self.entry["path"]).exists())

    def test_existing_different_file_is_preserved(self):
        target = self.root / self.entry["path"]
        target.parent.mkdir()
        target.write_bytes(b"local changes")
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            restore(self.root)
        self.assertEqual(target.read_bytes(), b"local changes")

    def test_final_checksum_is_checked_before_publication(self):
        self.entry["sha256"] = "0" * 64
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "Restored artifact checksum mismatch"):
            restore(self.root)
        self.assertFalse((self.root / self.entry["path"]).exists())

    def test_decompression_cannot_exceed_declared_size(self):
        self.entry["bytes"] = 1
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "exceeds declared size"):
            restore(self.root)
        self.assertFalse((self.root / self.entry["path"]).exists())

    def test_outside_destination_is_rejected(self):
        self.entry["path"] = "../outside.npz"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "Invalid artifact path"):
            restore(self.root)

    def test_symlink_destination_is_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            (self.root / "data").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "leaves the repository"):
                restore(self.root)
            self.assertFalse((Path(outside) / "artifact.npz").exists())


if __name__ == "__main__":
    unittest.main()
