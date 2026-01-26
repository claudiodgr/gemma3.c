# gemma3.c

> 🚀 **Pure C inference engine for Google's Gemma 3 4B IT model**
>
> A fully working, dependency‑free implementation of a modern large language model, written in pure C.
> No Python, no PyTorch, no CUDA. Just you, your CPU, and a lot of floating‑point math.

---

## ✨ Highlights

* ⚙️ **100% Pure C (C11)** – zero external dependencies
* 🧠 **Full Gemma 3 architecture** – GQA, hybrid local/global attention, SwiGLU MLP
* 🗺️ **Memory‑mapped weights** – efficient loading via `mmap` from BF16 SafeTensors
* 🔤 **Native SentencePiece tokenizer** – protobuf parsing, 262K vocabulary
* 🌊 **Streaming output** – token‑by‑token callbacks
* 💬 **Interactive chat mode** – with Gemma 3 chat templates
* 📦 **Library + CLI** – use it as a C library or as a standalone executable
* 🐧 **POSIX‑first design** – native on Linux and macOS
* 🪟 **Windows via compatibility layers** – WSL (recommended) or MinGW

---

## 📸 What is this project?

`gemma3.c` is a **from‑scratch CPU inference engine** for the *Gemma 3 4B IT* model.
It demonstrates that modern LLMs can be run without frameworks, without Python, and without GPUs.

This is not a toy: it fully loads the official model, runs inference, streams tokens, and supports chat.

---

## 🚀 Quick Start

> ⚠️ **Note on Windows**
> `gemma3.c` is a **POSIX-first project**. It runs natively on Linux and macOS.
> On Windows you must use **WSL** (recommended) or build with **MinGW** (with reduced features, no `mmap`).

### 1️⃣ Download the model (recommended: Python script)

The fastest and safest way to get the Gemma 3 model is via the built‑in Python downloader:

```bash
python download_model.py --token YOUR_HF_TOKEN
```

Or set your token once:

```bash
export HF_TOKEN=your_token_here
python download_model.py
```

This will create the `./gemma-3-4b-it` directory with all required files.

---

### 2️⃣ Build the project

```bash
make
```

---

### 3️⃣ Run

```bash
# Run a single prompt
./gemma3 -m ./gemma-3-4b-it -p "Explain quantum computing in simple terms."

# Interactive chat
./gemma3 -m ./gemma-3-4b-it -i

# Custom system prompt
./gemma3 -m ./gemma-3-4b-it -i -s "You are a pirate. Respond in pirate speak."
```

---

## 🛠️ Building

### Requirements

* C11 compiler (GCC / Clang)
* ~3–4 GB of free RAM

### Linux / macOS

```bash
make          # Optimized build (-O3)
make debug    # Debug symbols
make fast     # Aggressive CPU optimizations
make clean    # Cleanup
```

### 🪟 Windows

Two options:

#### Option 1 — WSL (Recommended)

Install WSL and Ubuntu, then:

```bash
sudo apt update
sudo apt install build-essential
make
```

This gives you the exact same environment as Linux.

#### Option 2 — MinGW

```bash
gcc -O3 -std=c11 -o gemma3.exe *.c
```

Note: Windows builds use standard file IO instead of `mmap`.

---

## 📥 Model Download (Recommended way: Python script)

The repository includes a **fully automated Python downloader** that:

* Handles HuggingFace authentication
* Downloads all model shards
* Resumes broken downloads
* Verifies integrity

### 🔥 One‑command setup

```bash
python download_model.py --token YOUR_HF_TOKEN
```

Or set the token once:

```bash
export HF_TOKEN=your_token_here
python download_model.py
```

This is the **recommended method**.

---

### Manual alternatives

```bash
# huggingface-cli
pip install huggingface_hub
huggingface-cli download google/gemma-3-4b-it --local-dir ./gemma-3-4b-it

# or git-lfs
git lfs install
git clone https://huggingface.co/google/gemma-3-4b-it
```

The model directory must contain:

* `model*.safetensors`
* `tokenizer.model`

---

## 🧪 Usage

### CLI Options

```
-m, --model <path>      Path to model directory (required)
-p, --prompt <text>     Input prompt
-i, --interactive       Interactive chat
-s, --system <text>     System prompt
-n, --max-tokens <n>    Max tokens (default 512)
-t, --temperature <f>   Temperature (default 0.7)
-k, --top-k <n>         Top‑k sampling
--top-p <f>             Top‑p sampling
-c, --context <n>       Context size
--seed <n>              RNG seed
-v, --verbose           Verbose output
```

---

## 📚 Library API

```c
#include "gemma3.h"

gemma3_ctx *ctx = gemma3_load_dir("./gemma-3-4b-it");

gemma3_gen_params params = gemma3_default_params();
char *out = gemma3_generate(ctx, "Hello!", &params, NULL, NULL);
printf("%s\n", out);
free(out);

gemma3_free(ctx);
```

Streaming:

```c
int cb(int id, const char *tok, void *u) {
    printf("%s", tok);
    return 0;
}
```

---

## 🧠 Architecture

| Parameter       | Value              |
| --------------- | ------------------ |
| Vocabulary      | 262,208            |
| Hidden size     | 2,560              |
| Layers          | 34                 |
| Attention heads | 8                  |
| KV heads        | 4 (GQA)            |
| Context length  | 128K               |
| Sliding window  | 1,024              |
| Pattern         | 5 local : 1 global |

---

## 💾 Memory

| Component           | Size       |
| ------------------- | ---------- |
| Weights (BF16 mmap) | ~8 GB disk |
| KV cache            | ~70 MB     |
| Activations         | ~100 MB    |
| **Total RAM**       | **~3 GB**  |

Lower memory:

```bash
./gemma3 -m ./gemma-3-4b-it -c 512 -p "Hello"
```

---

## ⚡ Performance (CPU)

* Prefill: 2–5 tok/s
* Generation: 1–3 tok/s

Optimizations:

```bash
make fast
```

---

## ⚠️ Limitations

* Text‑only
* CPU only
* No quantization (yet)

---

## 🧩 Project Layout

```
gemma3.c/
├── gemma3.h
├── gemma3.c
├── gemma3_transformer.c
├── gemma3_safetensors.c
├── gemma3_tokenizer.c
├── gemma3_kernels.c
├── main.c
├── download_model.py
└── README.md
```

---

## 🪪 License

MIT License.
Model weights are under Google’s Gemma license.

---

## 🙌 Credits

Inspired by:

* llama.cpp
* llama2.c
* flux2.c

---

If you ever wanted to see an LLM breathe in pure C, this is it.
