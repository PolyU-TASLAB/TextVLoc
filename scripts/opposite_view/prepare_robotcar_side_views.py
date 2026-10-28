"""Extract aligned RobotCar mono left/right/rear frames from camera tar files.

Input archives must come from the 2014-12-09-13-21-02 traversal. Frames are
demosaiced and rectified with the RobotCar SDK, then written as RGB PNGs for
robotcar_four_view_surroundings.py. Run inside clip-hloc-dev.
"""

import argparse
import bisect
import io
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PREPARED = Path("datasets/SensorKit/LoST_Robotcar/prepared/overcast_autumn_2m_2km_rectified")
SDK = Path("datasets/SensorKit/LoST_Robotcar/robotcar-dataset-sdk")
TRAVERSAL = "2014-12-09-13-21-02"


def index_archives(archive_dir, camera):
    archives = sorted(archive_dir.glob(f"{TRAVERSAL}_{camera}_*.tar"))
    if not archives:
        raise FileNotFoundError(f"No {TRAVERSAL}_{camera}_*.tar in {archive_dir}")
    indexed = {}
    for archive in archives:
        with tarfile.open(archive) as handle:
            for member in handle:
                if not member.isfile() or not member.name.endswith(".png"):
                    continue
                if (f"/{camera}/" not in member.name
                        and f"/mono/{camera.removeprefix('mono_')}/" not in member.name):
                    continue
                timestamp = int(Path(member.name).stem)
                indexed[timestamp] = (archive, member.name)
    if not indexed:
        raise ValueError(f"No {camera} PNGs in {archives}")
    return indexed


def selected_frames(indexed, query_timestamps, tolerance_us, excluded_timestamps=()):
    excluded = set(excluded_timestamps)
    stamps = sorted(stamp for stamp in indexed if stamp not in excluded)
    if not stamps:
        raise ValueError("No eligible frames after exclusions")
    selected = {}
    for query in query_timestamps:
        at = bisect.bisect_left(stamps, query)
        nearby = stamps[max(0, at - 1):min(len(stamps), at + 1)]
        closest = min(nearby, key=lambda stamp: abs(stamp - query))
        if abs(closest - query) > tolerance_us:
            raise ValueError(f"No side frame within {tolerance_us} us of {query}")
        selected[closest] = indexed[closest]
    return selected


def prepare(archive_dir, out_dir, tolerance_us, cameras):
    sys.path.insert(0, str(SDK / "python"))
    from camera_model import CameraModel

    query_names = [line.strip() for line in (PREPARED / "queries_front.txt").read_text().splitlines() if line.strip()]
    query_timestamps = [int(Path(name).stem) for name in query_names]
    assert len(query_names) == 866
    for camera in cameras:
        indexed = index_archives(archive_dir, camera)
        excluded = ()
        if camera == "mono_rear":
            excluded = {int(Path(name).stem) for name in
                        (PREPARED / "database_rear.txt").read_text().splitlines() if name.strip()}
        selected = selected_frames(indexed, query_timestamps, tolerance_us, excluded)
        model = CameraModel(str(SDK / "models"), camera)
        destination = out_dir / camera
        destination.mkdir(parents=True, exist_ok=True)
        by_archive = {}
        for stamp, (archive, member) in selected.items():
            by_archive.setdefault(archive, []).append((stamp, member))
        processed = skipped = 0
        for archive, frames in by_archive.items():
            with tarfile.open(archive) as handle:
                for stamp, member in frames:
                    output = destination / f"{stamp}.png"
                    if output.is_file() and Image.open(output).mode == "RGB":
                        skipped += 1
                        continue
                    source = handle.extractfile(member)
                    if source is None:
                        raise FileNotFoundError(f"{archive}:{member}")
                    raw = np.asarray(Image.open(io.BytesIO(source.read())))
                    rgb = cv2.cvtColor(raw, cv2.COLOR_BayerRG2RGB)
                    rectified = model.undistort(rgb)
                    Image.fromarray(rectified.astype(np.uint8), mode="RGB").save(output)
                    processed += 1
        print(f"{camera}: {len(indexed)} archive frames, {len(selected)} selected, "
              f"{processed} processed, {skipped} skipped", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=PREPARED / "images")
    parser.add_argument("--timestamp-tolerance-us", type=int, default=100_000)
    parser.add_argument("--cameras", nargs="+", choices=("mono_left", "mono_right", "mono_rear"),
                        default=("mono_left", "mono_right", "mono_rear"))
    args = parser.parse_args()
    if args.timestamp_tolerance_us < 0:
        parser.error("--timestamp-tolerance-us must be nonnegative")
    prepare(args.archive_dir, args.out_dir, args.timestamp_tolerance_us, args.cameras)


if __name__ == "__main__":
    main()
