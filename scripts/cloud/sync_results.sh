#!/usr/bin/env bash
# Bring the big-dataset results back to the laptop (docs/cloud-guide.md step 9). Runs on the LAPTOP; the VM must be
# running (start it again if it powered itself off; stop it right after).
# On the VM a tarball ~/bigdata-results.tgz is made of: ~/runs/bigdata (config.json, metrics.csv, model.pt, eval_val*,
# post_val.npz, post_trainfit.json, queue_timing.json, logs, queue.log, queue-effective.txt, post.log),
# ~/runs/bigdata-smoke (smoke runs, inspect.json/log, estimate.log, est.json, gpu.csv), ~/setup_vm.log,
# ~/pip-freeze-vm.txt, ~/pip-freeze-diff.txt and, from ~/data/dfp_full, the archive sha256, the archive listing and
# the test names (names only; no data). It is copied and unpacked into $DEST (default ~/runs/bigdata-vm, outside the repo).
#
# Usage (gcloud, default):  VM=mdf-l4 ZONE=<zone> scripts/cloud/sync_results.sh
#        (ssh/rsync instead):  MODE=ssh HOST=<user>@<external-ip> scripts/cloud/sync_results.sh
# Overridable: DEST (~/runs/bigdata-vm), MODE (gcloud | ssh)
set -euo pipefail

MODE="${MODE:-gcloud}"
DEST="${DEST:-$HOME/runs/bigdata-vm}"
REMOTE_CMD='cd ~ && tar czf bigdata-results.tgz --ignore-failed-read runs/bigdata runs/bigdata-smoke setup_vm.log \
pip-freeze-vm.txt pip-freeze-diff.txt data/dfp_full/data_full.tar.gz.sha256 data/dfp_full/archive_listing.txt \
data/dfp_full/archive_test_names.txt && ls -l bigdata-results.tgz && sha256sum bigdata-results.tgz'
mkdir -p "$DEST"

case "$MODE" in
    gcloud)
        : "${VM:?give VM=<instance name>}" "${ZONE:?give ZONE=<zone, e.g. the one chosen in the console>}"
        gcloud compute ssh "$VM" --zone "$ZONE" --command "$REMOTE_CMD"
        gcloud compute scp --zone "$ZONE" "$VM:~/bigdata-results.tgz" "$DEST/bigdata-results.tgz"
        ;;
    ssh)
        : "${HOST:?give HOST=<user>@<external ip>}"
        ssh "$HOST" "$REMOTE_CMD"
        rsync -av --partial "$HOST:bigdata-results.tgz" "$DEST/bigdata-results.tgz"
        ;;
    *) echo "MODE must be gcloud or ssh" >&2; exit 1 ;;
esac
sha256sum "$DEST/bigdata-results.tgz"
tar xzf "$DEST/bigdata-results.tgz" -C "$DEST"
echo "Unpacked into $DEST:"
ls "$DEST/runs/bigdata" 2>/dev/null | grep -vE '\.log$' || echo "(no runs/bigdata yet)"
echo "Finished runs:"
grep -l '"status": "finished"' "$DEST"/runs/bigdata/*/config.json 2>/dev/null | wc -l
echo "Next: scripts/bigdata_analysis.py --full-root $DEST/runs/bigdata (docs/cloud-guide.md step 10)"
