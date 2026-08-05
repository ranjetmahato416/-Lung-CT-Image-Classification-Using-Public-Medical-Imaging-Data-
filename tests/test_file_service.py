from pathlib import Path
import os
import time

from services.file_service import (
    cleanup_old_files
)


def test_cleanup_old_files(
    tmp_path
):

    old_file = (
        tmp_path
        / "old.png"
    )

    recent_file = (
        tmp_path
        / "recent.png"
    )

    old_file.write_bytes(
        b"old"
    )

    recent_file.write_bytes(
        b"recent"
    )

    # Make one file two hours old
    old_timestamp = (
        time.time()
        - 7200
    )

    os.utime(
        old_file,
        (
            old_timestamp,
            old_timestamp
        )
    )

    deleted = cleanup_old_files(
        tmp_path,
        max_age_seconds=3600
    )

    assert deleted == 1

    assert (
        not old_file.exists()
    )

    assert (
        recent_file.exists()
    )