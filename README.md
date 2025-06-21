# Universal LoRA

**Universal LoRA** is a lightweight, modular PyTorch library that enables the integration of **Low-Rank Adaptation (LoRA)** into common neural network layers. The library provides easy-to-use wrappers and utility functions to inject LoRA modules into any PyTorch model, including models from `timm`.

---

## 🔧 Features

- 🧩 Drop-in LoRA wrappers for:
  - `nn.Linear`
  - `nn.Conv1d`, `nn.Conv2d`, `nn.Conv3d`
  - `nn.Embedding`
- 🔁 Recursively inject LoRA layers using `apply_lora()`
- ❄️ Automatically freeze original layer weights
- 💾 Save/load only the LoRA-specific parameters
- 🧪 Includes PyTest-based test suite

---

## 🚀 Installation

```bash
git clone https://github.com/foiv0s/universal_lora.git
cd universal_lora
pip install -e .