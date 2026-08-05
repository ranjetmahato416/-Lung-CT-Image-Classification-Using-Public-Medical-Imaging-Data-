from pathlib import Path
import time


def cleanup_old_files(
    directory,
    max_age_seconds=3600
):
    """
    Remove files older than max_age_seconds.

    Default retention:
        1 hour
    """

    directory = Path(
        directory
    )

    if not directory.exists():
        return 0

    current_time = time.time()

    deleted_count = 0

    for file_path in directory.iterdir():

        if not file_path.is_file():
            continue

        try:

            file_age = (
                current_time
                - file_path.stat().st_mtime
            )

            if file_age > max_age_seconds:

                file_path.unlink()

                deleted_count += 1

        except OSError as error:

            print(
                "File cleanup warning:",
                repr(error)
            )

    return deleted_count