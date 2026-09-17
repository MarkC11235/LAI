# Setup

From a bare Ubuntu 26.04 install to a working stack. Each step ends with a
verification; do not start the next step until it passes, because every later
failure looks the same ("the model won't start") no matter which step broke.

Step labels mark what is specific to this machine, so the guide can be adapted:
**[UNIVERSAL]** any Intel discrete GPU · **[OS]** Ubuntu 26.04 · **[GPU]**
Battlemage · **[COUNT]** two GPUs · **[PLATFORM]** consumer AM5 board.

The Local Deep Research and SearXNG layer is optional and set up separately in
[deep-research.md](deep-research.md).

---

## 1. Firmware

### 1.1 Resizable BAR **[GPU]**

Enable **Above 4G Decoding**, then **Re-Size BAR Support** (usually greyed out
until the first is on). Arc depends on ReBAR far more than NVIDIA or AMD cards
do: without it the cards underperform badly or fail to enumerate.

### 1.2 Memory profile **[PLATFORM]**

The Corsair `CMK64GX5M2B6000C30` kit has an Intel XMP profile and no AMD EXPO
profile, so the board silently runs it at 4800 MT/s. Enable XMP in the MSI OC
menu. A CMOS clear reverts this, so after any BIOS reset check:

```bash
sudo dmidecode -t 17 | grep -i 'configured memory speed'    # expect 6000 MT/s
```

### 1.3 Slots and power **[PLATFORM]** **[COUNT]**

AM5 gives the CPU 16 graphics lanes. On the X670E Carbon both cards run at
x8/x8, and there is no NVLink-style interconnect: every cross-card transfer
goes over PCIe. This is why layer split, not tensor split, is the default here
([findings](findings.md#split-mode)).

```bash
sudo lspci -vv | grep -A3 -i 'vga\|display' | grep LnkSta
```

The B70 is 230 W TBP (160–290 W configurable). Two cards and a Ryzen 9 need a
1000 W+ supply with two independent 12V-2x6 feeds; no daisy-chained adapters.

---

## 2. Drivers and runtime

### 2.1 Kernel driver: nothing to install **[OS]** **[GPU]**

Linux 7.0 includes the mainline `xe` driver for Battlemage. Do not install DKMS
packages or out-of-tree modules.

```bash
lspci -nn | grep -i vga          # two 8086:e223
sudo dmesg | grep -i xe          # xe bound to both
ls -l /dev/dri/renderD*          # renderD128 and renderD129
```

One render node means a firmware, slot, or power problem, not a driver problem.

### 2.2 Compute runtime **[OS]** **[GPU]**

The stock archive's compute-runtime predates the B70, so use the Intel graphics
PPA. `libze-dev` and `intel-ocloc` are needed for the ahead-of-time build in
step 4.

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
sudo apt install -y libze-intel-gpu1 libze1 intel-metrics-discovery \
                    intel-opencl-icd clinfo intel-gsc libze-dev intel-ocloc
```

### 2.3 Permissions **[UNIVERSAL]**

```bash
sudo gpasswd -a "$USER" render
sudo usermod -aG video "$USER"
```

**Log out and back in.** `newgrp render` only patches one shell, and when
several lines are pasted its subshell swallows the next command, which looks
like a silent failure.

```bash
clinfo -l        # two Intel devices
```

If `sudo clinfo -l` lists both cards and plain `clinfo -l` lists nothing, the
problem is group membership and nothing else.

---

## 3. oneAPI **[UNIVERSAL]**

Intel Deep Learning Essentials contains the compiler, oneDPL, oneDNN and oneMKL
that the SYCL backend needs, and is much smaller than the Base Toolkit.

```bash
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install -y intel-deep-learning-essentials
```

Keep the default `/opt/intel/oneapi`. Do **not** source `setvars.sh` from
`.bashrc`: it shadows the system compilers and fights conda. LAI sources it per
backend process instead (`gen/llama-env.sh`).

```bash
source /opt/intel/oneapi/setvars.sh && sycl-ls    # two [level_zero:gpu] B70 entries
```

oneAPI 2026.1 is newer than llama.cpp's verified list and prints
`MKL Warning: Incompatible OpenCL driver version` against the PPA runtime. It
works; whether it costs throughput is an open question.

---

## 4. llama.cpp (SYCL)

Build in a fresh shell with conda deactivated: conda has shadowed
`libstdc++`/`libsycl` and broken SYCL builds on this machine.

```bash
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cd ~/llama.cpp
conda deactivate 2>/dev/null || true
source /opt/intel/oneapi/setvars.sh

cmake -B build-new -DGGML_SYCL=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DGGML_SYCL_F16=ON -DGGML_SYCL_DEVICE_ARCH=bmg \
  -DGGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON
cmake --build build-new --config Release -j "$(nproc)" \
  --target llama-server llama-bench llama-cli llama-completion llama-mtmd-cli
```

`GGML_SYCL_DEVICE_ARCH=bmg` compiles kernels ahead of time for Battlemage
**[GPU]**; without it the first start of every model spends minutes JIT
compiling. `GGML_SYCL_F16` changes kernels at compile time, so toggling it
needs a full rebuild. The build directory name is what
`Settings.llama_server` in `models.py` points at; building into a new directory
keeps the previous build as a rollback ([operations](operations.md#rebuilding-llamacpp)).

```bash
./build-new/bin/llama-server --version                       # prints "version: NNNNN"
ZES_ENABLE_SYSMAN=1 ./build-new/bin/llama-cli --list-devices  # SYCL0 and SYCL1, ~31.5 and ~32.6 GiB free
```

Qwen3.8 Flash-Next (architecture `qwen4exp`) needs build b10664 or newer;
`lai check` enforces that.

---

## 5. llama-swap

Download the Linux amd64 binary from the
[releases page](https://github.com/mostlygeek/llama-swap/releases):

```bash
mkdir -p ~/bin
mv ~/Downloads/llama-swap ~/bin/ && chmod +x ~/bin/llama-swap
grep -q 'HOME/bin' ~/.bashrc || echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
exec bash
command -v llama-swap       # ~/bin/llama-swap
```

LAI calls `GET /health`, `GET /running`, `GET /logs[/stream]` and
`POST /api/models/unload[/<id>]`. These have moved between releases; a 404 from
`lai ps` or `lai unload` means the binary is older or newer than LAI expects.

---

## 6. Models

Weights live under `~/models/<name>/` (`Settings.model_dir`), and each registry
entry's `weights` path is relative to it. Download with an `--include` pattern
for one quantization, never a whole repository:

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download unsloth/Qwen3.8-27B-GGUF --local-dir ~/models/Qwen3.8-27B --include "*UD-Q5_K_XL*"
hf download unsloth/Qwen3.8-27B-GGUF --local-dir ~/models/Qwen3.8-27B --include "*mmproj-F16*"
```

Separate calls for weights and projector, so one failure doesn't discard the
other. One `--include` per pattern: several values after a single flag are
ignored with `Ignoring --include since filenames set`. If a download fetches
zero files, the pattern didn't match; list the repository's files first.

The registry expects:

| Directory under `~/models` | Used by |
|---|---|
| `Qwen3.8-27B/` (UD-Q5_K_XL + `mmproj-F16.gguf`) | `qwen38-27b-*` |
| `Qwen3.8-Flash-Next/UD-IQ3_XXS/` (3 shards) + `Qwen3.8-Flash-Next/mmproj-F16.gguf` | `qwen38-flash-next-*` |
| `Qwen3.6-35B-A3B-MTP/UD-Q4_K_XL/` | `qwen36-moe-c1-mtp` |
| `Qwen3.8-27B-GPTQ-Int4-MTP/` (safetensors, SergiioB's checkpoint, rev `9d189a60`) | `qwen38-vllm` |

Use Unsloth `UD-*` quantizations of Qwen3.8: the plain quants shipped a chat
template that throws `System message must be at the beginning`, and opencode
always sends a system message. A split GGUF whose first shard is only a few MB
is normal. The `ggml-vocab-*.gguf` files inside the llama.cpp repository are
tokenizer test fixtures with no weights.

`lai ls` shows `--` in the FILE column for anything missing.

---

## 7. LAI

```bash
git clone <this repo> ~/projects/lai
ln -s ~/projects/lai/bin/lai ~/bin/lai

cd ~/projects/lai
$EDITOR models.py          # Settings: llama_server path, card sizes, default model
lai check                  # every FAIL must be fixed; NOTEs are advisory
lai gen                    # writes gen/*, backs up and installs ~/.config/opencode/opencode.json
lai env                    # the environment wrapper alone: must print a llama.cpp version
```

Run it as a user service, so it survives logout and reboot:

```bash
lai service
sudo loginctl enable-linger "$USER"
lai status                 # up, http://127.0.0.1:9090 (systemd --user)
```

Prove the whole path on the fastest-loading model, then on each model you
intend to use:

```bash
lai smoke qwen36-moe-c1-mtp
```

---

## 8. vLLM engine (for `qwen38-vllm`) **[GPU]**

vLLM runs in Docker, launched by llama-swap through `gen/vllm-qwen38-vllm.sh`.
It needs:

1. **Docker Engine**, with your user in the `docker` group. llama-swap runs the
   launcher as your user under systemd, so `docker ps` must work without sudo.
2. **The pinned image.** The digest is in the launcher in `models.py`; pull it
   once so the first load isn't a multi-GB download inside a health-check
   timeout:
   ```bash
   docker pull "$(grep -o 'vllm/vllm-openai-xpu@sha256:[0-9a-f]*' models.py)"
   ```
3. **The MTP patches** in `vendor/vllm-xpu-mtp/`; see [its README](../vendor/vllm-xpu-mtp/README.md).
   The launcher refuses to start without them.
4. **A compile cache directory:** `mkdir -p ~/.cache/vllm-xpu`. Without it every
   start recompiles (~93 s instead of ~12.7 s).

Then `lai smoke qwen38-vllm`, and confirm in `lai logs` that the startup log
contains `XPUwNa16LinearKernel for AutoGPTQLinearMethod` (the integer XMX path)
and a populated `speculative_config=` (MTP is live). Without the first, the
server works at half speed and reports no error.

---

## 9. opencode

Install opencode per its documentation. LAI owns the global config: `lai gen`
writes `~/.config/opencode/opencode.json` with one provider (`llamacpp`) pointed
at `http://127.0.0.1:9090/v1`, one model entry per enabled registry model, and
whatever is in `Settings.opencode_extra` (the SearXNG MCP server and the
`webfetch` override). Never edit the installed file; `lai gen` replaces it
(keeping a timestamped backup when it differs).

Web search needs the MCP server installed at the path `models.py` names:

```bash
npm config set prefix ~/.npm-global
npm install -g mcp-searxng
opencode mcp list          # searxng: connected
```

Personal instructions for the model go in `~/.config/opencode/AGENTS.md`,
which LAI does not touch. Include a section telling the model when to use
`searxng_web_search`: local models will not reach for search unprompted.

Switch models inside opencode with `/models`. Restart opencode only after
`lai gen` adds a model.

---

## What changes on different hardware

**Another Intel GPU generation:** change `GGML_SYCL_DEVICE_ARCH`. Alchemist and
older use the `i915` driver, and the stock compute runtime is sufficient without
the PPA.

**One GPU:** set `Settings.cards` to a single entry. Nothing can co-reside, and
the split-mode findings no longer apply.

**Workstation or server platform (Threadripper, EPYC, Xeon):** full x16 links
change the tensor-split trade-off; re-measure rather than inheriting
`lai check`'s block on `split_mode="tensor"`.

**Another distribution or older Ubuntu:** the kernel step stops being free and
the PPA is Ubuntu-only.
