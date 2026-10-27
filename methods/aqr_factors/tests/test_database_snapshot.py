"""A clone can restore its database safely from the lossless Git LFS archive."""
import gzip
import hashlib
import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from methods.aqr_factors.database import restore_snapshot


class DatabaseSnapshotTests(unittest.TestCase):
    def prepare(self, folder):
        folder = Path(folder)
        original = folder / "fixture.sqlite3"
        with closing(sqlite3.connect(original)) as db:
            db.execute("CREATE TABLE facts (ticker TEXT, value REAL)")
            db.execute("INSERT INTO facts VALUES (?, ?)", ("A", 12.5))
            db.commit()
        raw = original.read_bytes()
        (folder / "aqr.sqlite3.gz").write_bytes(gzip.compress(raw))
        (folder / "aqr.sqlite3.manifest.json").write_text(json.dumps(dict(
            database_bytes=len(raw), database_sha256=hashlib.sha256(raw).hexdigest())), encoding="utf-8")
        return raw

    def test_restore_exact_bytes_and_readable_sqlite_without_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            raw = self.prepare(folder)
            restored = restore_snapshot(folder)
            self.assertEqual(restored.read_bytes(), raw)
            with closing(sqlite3.connect(restored)) as db:
                self.assertEqual(db.execute("SELECT * FROM facts").fetchall(), [("A", 12.5)])
            with self.assertRaises(FileExistsError):
                restore_snapshot(folder)
            self.assertEqual(restored.read_bytes(), raw)

    def test_bad_checksum_does_not_publish_a_database_or_leave_partial_files(self):
        with tempfile.TemporaryDirectory() as folder:
            self.prepare(folder)
            manifest = Path(folder) / "aqr.sqlite3.manifest.json"
            metadata = json.loads(manifest.read_text())
            metadata["database_sha256"] = "0" * 64
            manifest.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "checksum"):
                restore_snapshot(folder)
            self.assertFalse((Path(folder) / "aqr.sqlite3").exists())
            self.assertEqual(list(Path(folder).glob(".aqr-restore-*")), [])

    def test_lfs_pointer_requires_the_actual_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "aqr.sqlite3.gz"
            archive.write_text("version https://git-lfs.github.com/spec/v1\n")
            with self.assertRaisesRegex(ValueError, "git lfs pull"):
                restore_snapshot(folder)


if __name__ == "__main__":
    unittest.main()
