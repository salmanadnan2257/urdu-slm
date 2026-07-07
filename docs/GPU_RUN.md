# Phase 2: the base-model GPU run

Everything for the ~124M `base` model is wired. Phase 1 (this repo) verified the
whole stack on CPU; phase 2 is a budget decision, not an engineering one. This
doc gives the one command, the cost, and the token-budget reasoning.

## Token budget (Chinchilla-style, honest for the corpus we actually have)

The Chinchilla rule of thumb is roughly 20 training tokens per parameter for
compute-optimal training. For the 124M `base` model that points at about
**2.5B tokens**.

Our cleaned corpus is **111.7M training tokens** (112.8M including validation;
see the README corpus table). That is only about **0.9 tokens per parameter**,
roughly 4.5% of the compute-optimal budget. This is the honest headline: the
`base` model is **data-limited, not compute-limited**.

Reaching 2.5B tokens would require repeating the corpus ~22 times. Repetition up
to ~4 epochs degrades little in practice (Muennighoff et al., "Scaling
Data-Constrained Language Models", 2023), but 22 epochs would overfit badly and
the extra compute would mostly memorize. So the defensible plan is:

1. **Train 2-4 epochs (~220M-450M tokens).** This is what `configs/base.yaml`
   is set to (`max_tokens: 450000000`, ~4 epochs). The data loader wraps the
   corpus automatically when it runs out. Expect a capable-for-its-data model,
   not a compute-optimal one.
2. **Add more data before spending GPU money.** The real lever is corpus size,
   not compute. CC-100 (ur) alone is several GB and would roughly 10-20x the
   token count, moving the budget toward Chinchilla-optimal. The pipeline is
   built to ingest additional plain-text sources; adding them and re-running is
   the highest-value next step, and it is called out in the README's
   "What I'd do differently".

The config targets option 1 so the run is honest about the data on hand. Raise
`max_tokens` only after the corpus grows.

## Cost estimate

Prices below are from the RunPod pricing page (https://www.runpod.io/pricing),
Community Cloud, checked 2026-07-06:

| GPU              | USD/hr (Community) |
| ---------------- | ------------------ |
| RTX 4090 24GB    | 0.69               |
| RTX A6000 48GB   | 0.49               |
| A100 PCIe 80GB   | 1.39               |
| A100 SXM 80GB    | 1.49               |
| H100 PCIe 80GB   | 2.89               |

Throughput estimate for a 124M model with bf16 + Flash attention (via
`scaled_dot_product_attention`) on a single A100 80GB: roughly 25k-40k tokens/s
for this width/depth at `block_size=1024`. Taking a conservative 30k tok/s:

- The planned 450M-token (~4-epoch) run: 450M / 30,000 tok/s = ~4.2 hours
  wall-clock on one A100 80GB. At 1.39 USD/hr that is about **6 USD**.
- A single-epoch run (111.7M tokens) is ~1 hour, roughly **1.5 USD**.
- On an RTX 4090 (24GB) the model fits with `grad_accum` raised and
  `batch_size` lowered; expect ~1.5-2x the wall-clock at 0.69 USD/hr, landing in
  a similar single-digit-dollar total. This is a cheap run precisely because the
  corpus is small; the cost ceiling is data, not GPU hours.

These are estimates, not measured; the only measured throughput in this repo is
the CPU proof run.

## One-command launch

On a fresh RunPod / Lambda box with an NVIDIA GPU and the three raw corpus files
already downloaded (or reachable), from the repo root:

```bash
# 1. environment
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 2. data + tokenizer (skip if data/ is already populated)
python -m pipeline.build_corpus --raw-dir $URDU_SLM_RAW_DIR --out-dir data \
    --wiki urwiki-latest-pages-articles.xml.bz2 \
    --leipzig urd_newscrawl_2016_1M.tar.gz:leipzig_newscrawl_2016 \
    --leipzig urd_newscrawl_2011_300K.tar.gz:leipzig_newscrawl_2011
python -m pipeline.train_tokenizer --data-dir data --out-dir tokenizer
python -m pipeline.tokenize_corpus --data-dir data --tokenizer tokenizer/tokenizer.json

# 3. the launch (single GPU)
python train.py --config configs/base.yaml

# ... or multi-GPU with DDP:
torchrun --nproc_per_node=<N> train.py --config configs/base.yaml
```

Training is resumable: re-run the same command with `--resume` and it picks up
from `runs/base/ckpt.pt` (model, optimizer, LR schedule, and data cursor all
restored).

## After training

```bash
python -m eval.perplexity  --ckpt runs/base/ckpt.pt --bin data/val.bin
python -m eval.script_mix   --ckpt runs/base/ckpt.pt
python -m eval.cloze        --ckpt runs/base/ckpt.pt --shots 2
python sample.py            --ckpt runs/base/ckpt.pt
```

## Real run log (RunPod, A100 SXM 80GB, 2026-07-06)

The A100 PCIe tier was out of capacity at signup time; A100 SXM was used instead,
same 80GB VRAM, $1.49/hr instead of $1.39/hr, no meaningful cost difference.

SSH key setup: the account-level public key RunPod asks for is only injected into
a pod's `authorized_keys` at creation time. If you add the key after the pod is
already up, a plain reconnect or even a Stop/Start cycle may not pick it up. The
reliable fix: open the pod's **Web Terminal** and run this once, using your own
public key:
```bash
mkdir -p ~/.ssh && echo "<your ssh-ed25519 public key line>" >> ~/.ssh/authorized_keys \
  && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys
```
Use the **"SSH over exposed TCP"** connection string from the Connect tab (the
`ssh root@<ip> -p <port>` form), not the proxied `ssh <pod-id>@ssh.runpod.io` one;
only the direct-TCP form supports SCP/SFTP, which rsync needs.

A local SSH config alias makes every later command shorter and works for
`ssh`/`scp`/`rsync` alike:
```
# ~/.ssh/config
Host runpod-urdu
    HostName <pod-ip>
    Port <pod-ssh-port>
    User root
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking accept-new
```

Transfer (data/ and tokenizer/ are already built locally, no need to rebuild on
the pod):
```bash
rsync -avz --progress -e ssh ./ runpod-urdu:/workspace/urdu-slm/
```

The RunPod PyTorch template ships a working CUDA-enabled torch already
(verified: 2.8.0+cu128 here). Installing this repo's pinned `torch==2.12.1` from
PyPI would silently replace it with a CPU-only build, so install everything
except torch:
```bash
grep -viE '^torch' requirements.txt > /tmp/reqs_no_torch.txt
pip install --break-system-packages -r /tmp/reqs_no_torch.txt
python3 -c "import torch; print(torch.cuda.is_available())"   # must print True
```
(`--break-system-packages` is fine on a disposable single-purpose pod; the
"externally-managed-environment" guard otherwise blocks a plain `pip install`.)

Run detached so it survives an SSH disconnect over the multi-hour run:
```bash
tmux new-session -d -s training \
  'python3 train.py --config configs/base.yaml 2>&1 | tee train_run.log'
# reattach any time with: tmux attach -t training
```

**Stopping and cost control**: SSH access only reaches the guest OS. It cannot
stop RunPod's billing meter, that is an account-level action. Once training
finishes (or you want to stop early), pull the checkpoint back, then go to the
RunPod dashboard and click **Terminate** on the pod (not just Stop, which still
bills for the reserved disk/GPU allocation). There is no way to do this step
from inside the pod over SSH.
