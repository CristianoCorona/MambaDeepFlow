# Cloud guide: the big-dataset experiment on a Google Cloud VM

Step-by-step guide for someone who has not used Google Cloud before. Experiment: B, MDv64-local and MDv64 trained on the
full 53.8k data set, validation only (`docs/preregistration-bigdata.md`). Everything that runs on the VM is in
`scripts/cloud/`; the results come back to the laptop and are analysed there (`scripts/bigdata_analysis.py`).

**Budget: 50 USD of credits.** A running VM with a GPU costs money every minute, also when it does nothing. The queue
powers the VM off by itself when it ends, but **check that the VM is stopped** whenever you leave it (step 8).
No price is written in this guide: prices depend on the region and change; read them in the console (section "Prices").

## 0. Overview

| Step | Where | What | Time (rough) |
|---|---|---|---|
| 1 | console | project, billing / credits, GPU quota, budget alert | 15 min (+ quota approval: minutes to 2 days) |
| 2 | console or gcloud | create the VM (L4 GPU, g2-standard-8) | 5 min |
| 3 | VM | `scripts/cloud/setup_vm.sh` (env, checks, tests) | 15-25 min |
| 4 | VM | `scripts/cloud/get_full_data.sh` (10 GB download + unpack) | 20-60 min, depends on the TUM server |
| 5-6 | VM | `scripts/cloud/smoke_test.sh` (names, RAM, short runs, estimate) | 10 min |
| 7 | VM | `CUTOFF=... scripts/cloud/launch_bigdata.sh` (queue in tmux) | ~1-2 days of GPU time for all 9 runs (estimate from step 6) |
| 8 | console | VM stopped at the end (automatic) - check it | - |
| 9 | laptop | `scripts/cloud/sync_results.sh` (start the VM, copy, stop) | 10 min |
| 10 | laptop | `scripts/bigdata_analysis.py`, notebook, commit | - |
| 11 | console | delete VM and disk when the results are safe | 2 min |

## 1. Project, credits, GPU quota, budget alert (console, once)

1. Open <https://console.cloud.google.com>, sign in with the account that holds the credits.
2. **Project**: top bar, project selector > *New project* (e.g. `mdf-bigdata`). Select it.
3. **Billing**: menu > *Billing*: the project must be linked to the billing account that holds the 50 USD credit
   (*Billing > Credits* shows the credit and what remains). A free-trial account may not be allowed to use GPUs at all;
   if the GPU quota request below is refused for that reason, the account has to be upgraded (the credits stay).
4. **Compute Engine API**: menu > *Compute Engine*; the first visit asks to enable the API (1-2 min).
5. **GPU quota** (new projects usually have 0 GPUs): menu > *IAM & Admin > Quotas & system limits*. Filter for
   `GPUs (all regions)` and set a request of **1**; also check `NVIDIA L4 GPUs` in the region you will use (step 2).
   Click *Edit quotas*, ask for 1, give a short reason ("course project, one L4 for ~2 days"). Wait for the email.
6. **Budget alert**: *Billing > Budgets & alerts > Create budget*: amount 50 USD, alerts at 50 %, 75 %, 90 %, 100 %
   (email). **An alert does not stop anything**: it only sends an email; stopping is up to you (STOP file, step 7, or
   stopping the VM).

## 2. Create the VM

Choices (all checked against `docs/lab-notebook.md`, 2026-10-19 entry):

| Item | Value | Why |
|---|---|---|
| GPU | 1 x NVIDIA L4 (24 GB) | asked for; the models need < 1 GB of GPU memory |
| Machine type | **g2-standard-8** (8 vCPU, **32 GB RAM**); fallback g2-standard-12 (48 GB) | the loader keeps the whole set in RAM: ~21 GiB peak per training process (estimate below); 32 GB leaves ~8 GiB |
| Boot disk | **150 GB**, *Balanced persistent disk* | image + env (~15 GB) + archive (10.2 GiB) + unpacked train (~11 GB) + runs; 100 GB is the minimum |
| Image | a *Deep Learning on Linux* image with CUDA 12.x and the NVIDIA driver (>= 570) | driver already installed or installed at first boot |
| Provisioning | *Standard* (on-demand) | *Spot* is cheaper but can be stopped at any time; a stopped run restarts from scratch (no mid-run checkpoint) |
| Region / zone | one that offers L4 and where the quota was granted | L4 is not in every zone; prices differ per region |

RAM estimate (measured, not assumed): the loader keeps float32 inputs + targets, 2 x 3 x 128 x 128 x 4 = 393 216 bytes
per sample (`mdf.data.load_raw` on a real file). The 6.4k runs peaked at 3 817 MiB RSS (`~/runs/night1/ep133_s0`,
5 998 + 402 samples = 2 400 MiB of arrays), i.e. ~1.4 GiB of process overhead (Python, torch, CUDA context). For at
most 53 800 samples: 53 800 x 393 216 B = 19.7 GiB + 1.4 GiB = **~21 GiB**. Step 6 (`inspect_full.py --load-ram`)
measures it on the VM before any run; if it reports more than ~26 GiB, recreate the VM as g2-standard-12.

### 2a. From the console

*Compute Engine > VM instances > Create instance*:
1. Name `mdf-l4`; region / zone: pick one, then in *Machine configuration* choose **GPUs**, GPU type **NVIDIA L4**,
   number 1, machine type **g2-standard-8**. If L4 is not offered, try another zone.
2. The right-hand panel shows the **estimated price** (per month and per hour): write down the hourly figure (step 6
   needs it).
3. *Boot disk > Change*: operating system **Deep Learning on Linux**, a version with **CUDA 12.x** (e.g. "CUDA 12.8"
   with an NVIDIA 570+ driver; take the newest CUDA 12 one offered, not CUDA 13), disk type *Balanced persistent
   disk*, size **150 GB**.
4. *Advanced > Management*: VM provisioning model **Standard**; *On host maintenance*: Terminate (required with GPUs).
5. *Create*. Billing starts now.

### 2b. Or with gcloud (on the laptop, after `gcloud init`)

```bash
ZONE=<zone with L4, e.g. one listed by the next command>
gcloud compute accelerator-types list --filter="name=nvidia-l4" --format="value(zone)"
gcloud compute images list --project deeplearning-platform-release --no-standard-images \
    --filter="family~cu12" --format="value(family)" | sort -u      # pick a cu12x family with nvidia-570 or newer
FAMILY=<chosen family>
gcloud compute instances create mdf-l4 --zone="$ZONE" --machine-type=g2-standard-8 \
    --image-family="$FAMILY" --image-project=deeplearning-platform-release \
    --boot-disk-size=150GB --boot-disk-type=pd-balanced \
    --maintenance-policy=TERMINATE --metadata=install-nvidia-driver=True
```
(G2 machine types come with their L4 attached; no `--accelerator` flag is needed.)

## 3. Connect and set up the environment

- Console: *VM instances* > **SSH** button (browser terminal). Or from the laptop: `gcloud compute ssh mdf-l4 --zone $ZONE`.
- On a Deep Learning image the first login may ask to install the NVIDIA driver: answer **y**, wait, then check
  `nvidia-smi` (it must show the L4).

```bash
git clone -b big-dataset https://github.com/CristianoCorona/MambaDeepFlow.git
cd MambaDeepFlow
scripts/cloud/setup_vm.sh          # log: ~/setup_vm.log; ends with "setup_vm.sh OK"
```
`setup_vm.sh` installs Miniforge (if there is no conda), creates env `mdf` with Python 3.12 and installs
`environment/pip-freeze.txt` (torch 2.10.0+cu128, causal-conv1d 1.7.0 and mamba-ssm 2.3.2.post1 from the same GitHub
wheels, sha256-pinned; never built from source), checks the versions, runs `scripts/check_env.py --seed 0` (the Mamba
CUDA kernels must actually be used) and the unit tests that need no data. If it stops, read the last lines, fix, and run
it again (finished steps are skipped).

## 4. Full data set

```bash
scripts/cloud/get_full_data.sh     # ~/data/dfp_full; data/train_full -> ~/data/dfp_full/data/train
```
Downloads `data_full.tar.gz` (10 961 766 491 bytes, resumable: run it again if it stops), lists the whole archive
(checks the gzip stream), prints the files per directory and the **sha256** (write it in the notebook), unpacks only
`data/train/` (the test set is not unpacked on the VM) and links `data/train_full`.

## 5-6. Checks and short runs

```bash
HOURLY_USD=<hourly price from step 2> SPENT_USD=<credit used so far, Billing > Reports> scripts/cloud/smoke_test.sh
```
1. `inspect_full.py`: files per subdirectory (`reg`, `shear`, `sup`), names that do not parse, samples per profile,
   the 85 validation and 30 test profiles in the full set, the split sizes, and (`--load-ram`) the **peak RAM** of
   loading train + validation. It stops on any problem (see "If something differs" below).
2. One short run (600 steps, 10 % of the training files per profile, full validation) of B, MDv64-local and MDv64,
   each evaluated on validation: checks the whole pipeline and measures it/s and memory on the L4.
3. `estimate_budget.py`: hours per 133-epoch run and for the queue, cost, **which runs fit the budget**, and a
   suggested `CUTOFF`. Outputs in `~/runs/bigdata-smoke/` (`estimate.log`).

Decision (pre-registered budget rule): the queue is launched in priority order; MDv64 runs only if they fit. The
cutoff makes `run_queue.py` start no run whose estimated end is after it. Choose the cutoff as the earlier of the
suggested one and the time you need the results (report: Monday 26/10; analysis needs ~1 day).

### If something differs

- `sup/` has files: they are not used (default subdirectories `reg,shear`); nothing to do, it is declared.
- Names in `shear/` do not parse: use the regular airfoils only. Run every later command with
  `export MDF_FULL_SUBDIRS=reg` (smoke test, launcher), and write it in the notebook (declared fallback).
- A validation profile is missing or a test profile appears in train / validation: stop, do not launch; bring
  `~/runs/bigdata-smoke/inspect.log` back.
- RAM peak above ~26 GiB: stop the VM, change the machine type to g2-standard-12 (*Edit* on the stopped VM), start again.

## 7. Launch the queue

```bash
CUTOFF=2026-10-21T18:00 scripts/cloud/launch_bigdata.sh     # use the value decided in step 6
```
- Refuses on a dirty tree, without the committed pre-registration, without the smoke-test outputs.
- Runs `queues/bigdata.txt` in a tmux session `bigdata`: B s0, MDv64-local s0, B s1, MDv64-local s1, B s2,
  MDv64-local s2, MDv64 s0, s1, s2; each run is evaluated on validation right after training.
- After the queue: `post_bigdata.py` (wall band, divergence, training-error check; ~30 min), then the VM **powers
  itself off** (3 minutes later; `sudo shutdown -c` cancels). `AUTO_SHUTDOWN=0` disables it.

Monitoring (after `gcloud compute ssh` / SSH button; closing the SSH window does not stop the queue):
```bash
tail -f ~/runs/bigdata/queue.log          # Ctrl-C leaves tail, not the queue
tmux attach -t bigdata                    # detach again with Ctrl-b then d
nvidia-smi                                # GPU busy?
touch ~/runs/bigdata/STOP                 # stop after the current run (then post-processing and power-off)
```
Budget during the queue: check *Billing > Reports* (costs appear with a delay of hours) and the credit left. If the
spent credit approaches the limit (e.g. 45 USD): `touch ~/runs/bigdata/STOP`, or stop the VM at once (the current run
is lost). Runs not done are declared in the notebook.

## 8. Stop the VM (always check)

- Console: *VM instances* > the VM > **Stop**. Status must become *Stopped* (or *Terminated*). Or `sudo shutdown -h now`
  on the VM, or `gcloud compute instances stop mdf-l4 --zone $ZONE` on the laptop.
- A stopped VM costs no GPU / CPU time, but **the disk is still billed** (small, per GB-month) until it is deleted.

## 9. Bring the results back (laptop)

Start the VM (*Start / Resume*; ~1 min), then on the laptop:
```bash
VM=mdf-l4 ZONE=$ZONE scripts/cloud/sync_results.sh        # into ~/runs/bigdata-vm (outside the repo)
```
(Without gcloud on the laptop: `MODE=ssh HOST=<user>@<external ip>`.) Then **stop the VM again**.

## 10. Analysis (laptop)

```bash
python scripts/bigdata_analysis.py --full-root ~/runs/bigdata-vm/runs/bigdata
```
Outputs `docs/results/val-bigdata/` and `docs/figures/val-bigdata/bigdata_rel_vy.pdf`; notebook entry and commit as
usual (all runs reported, missing ones declared).

## 11. Clean up

When the results are on the laptop (and committed): *VM instances* > the VM > **Delete** (deletes the boot disk too,
unless *Keep boot disk* was chosen; check *Compute Engine > Disks* is empty). Check *Billing > Reports* the next day.

## Prices: where to read them

- *Create instance* page: the right-hand panel shows the estimated cost of the chosen configuration (GPU + machine +
  disk), per month and per hour, for the chosen region and provisioning model.
- Google Cloud pricing calculator: <https://cloud.google.com/products/calculator> (Compute Engine, G2, L4, region).
- Pricing pages: "GPU pricing" and "VM instance pricing" (accelerator-optimized G2) in the Compute Engine docs.
- What was really spent: *Billing > Reports* (filter on the project) and *Billing > Credits* (credit left).
