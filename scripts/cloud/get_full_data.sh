#!/usr/bin/env bash
# Download, check and unpack the full data set on the cloud VM (docs/cloud-guide.md step 4).
# Source: README of thunil/Deep-Flow-Prediction, "Full data set with 53.8k samples plus test data (10GB)":
#   https://dataserv.ub.tum.de/s/m1459172/download?path=%2F&files=data_full.tar.gz  (mediaTUM share m1459172)
# Size 10 961 766 491 bytes (Content-Range of the server, 2026-10-19). No checksum is published: the sha256 is
# computed and printed (record it in the lab notebook).
# Layout (first 40 MiB of the archive, 2026-10-19): data/test/*.npz (the 90 test files), data/train/sup/,
# data/train/reg/*.npz, ... (the rest, e.g. data/train/shear/, is checked here from the full listing).
#
# Steps: download with resume (curl -C -), byte count, full listing (tar tzvf reads the whole gzip stream: CRC and
# length checked) with path checks (all under data/, no absolute or '..' paths, only files and directories), sha256,
# then unpack ONLY data/train/ (the test set is not unpacked on the VM; its names go to archive_test_names.txt to check
# splits/test_profiles.txt), counts per subdirectory on disk vs listing, symlink data/train_full -> DEST/data/train.
#
# Usage: scripts/cloud/get_full_data.sh      Overridable: DEST (~/data/dfp_full), KEEP_ARCHIVE (1); URL and
#        EXPECTED_BYTES only for a local test with a small fake archive
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="${DEST:-$HOME/data/dfp_full}"
KEEP_ARCHIVE="${KEEP_ARCHIVE:-1}"
URL="${URL:-https://dataserv.ub.tum.de/s/m1459172/download?path=%2F&files=data_full.tar.gz}"
EXPECTED_BYTES="${EXPECTED_BYTES:-10961766491}"
ARCHIVE="$DEST/data_full.tar.gz"
mkdir -p "$DEST"

free_gb=$(df -BG --output=avail "$DEST" | tail -1 | tr -dc 0-9)
echo "Free disk at $DEST: ${free_gb} GB (needed: ~11 GB archive + ~11 GB unpacked train)"
[ "$free_gb" -ge "${MIN_FREE_GB:-25}" ] || { echo "ERROR: less than ${MIN_FREE_GB:-25} GB free" >&2; exit 1; }

if [ -f "$ARCHIVE" ] && [ "$(stat -c %s "$ARCHIVE")" -eq "$EXPECTED_BYTES" ]; then
    echo "Archive already complete: $ARCHIVE"
else
    echo "Downloading (resumable; re-run this script if it stops) ..."
    curl -L --fail --retry 20 --retry-delay 10 --retry-all-errors -C - -o "$ARCHIVE" "$URL"
fi
bytes=$(stat -c %s "$ARCHIVE")
echo "Archive bytes: $bytes (expected $EXPECTED_BYTES)"
[ "$bytes" -eq "$EXPECTED_BYTES" ] || { echo "ERROR: size differs; delete $ARCHIVE and download again" >&2; exit 1; }

echo "Listing the whole archive (checks the gzip stream) ..."
tar tzvf "$ARCHIVE" > "$DEST/archive_listing.txt"
awk '{print $6}' "$DEST/archive_listing.txt" > "$DEST/archive_names.txt"
if grep -qvE '^data/' "$DEST/archive_names.txt" || grep -qE '(^/|(^|/)\.\.(/|$))' "$DEST/archive_names.txt"; then
    echo "ERROR: entries outside data/ or with absolute / '..' paths" >&2; exit 1
fi
if cut -c1 "$DEST/archive_listing.txt" | grep -qvE '^[-d]$'; then
    echo "ERROR: entries other than regular files and directories (links, devices)" >&2; exit 1
fi
grep -E '^data/test/.+\.npz$' "$DEST/archive_names.txt" > "$DEST/archive_test_names.txt" || true
echo "Entries per directory in the archive (.npz files):"
grep -E '\.npz$' "$DEST/archive_names.txt" | sed -E 's#/[^/]+$##' | sort | uniq -c
echo "Other files (not .npz, not directories):"
grep -vE '(\.npz|/)$' "$DEST/archive_names.txt" || echo "  none"

echo "sha256 (record it in the notebook):"
sha256sum "$ARCHIVE" | tee "$DEST/data_full.tar.gz.sha256"

echo "Unpacking data/train only ..."
tar xzf "$ARCHIVE" --no-same-owner -C "$DEST" data/train
for d in "$DEST"/data/train/*/; do
    sub=$(basename "$d")
    on_disk=$(find "$d" -maxdepth 1 -type f -name '*.npz' | wc -l)
    listed=$(grep -cE "^data/train/$sub/[^/]+\.npz$" "$DEST/archive_names.txt" || true)
    echo "  data/train/$sub: $on_disk .npz on disk, $listed in the listing"
    [ "$on_disk" -eq "$listed" ] || { echo "ERROR: counts differ for $sub" >&2; exit 1; }
done
[ ! -e "$DEST/data/test" ] || { echo "ERROR: data/test was unpacked; remove it" >&2; exit 1; }
ln -sfn "$DEST/data/train" "$REPO/data/train_full"
echo "Symlink: $(ls -l "$REPO/data/train_full" | awk '{print $9, $10, $11}')"
if [ "$KEEP_ARCHIVE" = 0 ]; then rm -f "$ARCHIVE"; echo "Archive removed"; fi
echo "Done. Next: scripts/cloud/smoke_test.sh (it runs scripts/cloud/inspect_full.py first)."
