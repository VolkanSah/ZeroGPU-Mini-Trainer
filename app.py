# Source https://github.com/VolkanSah/ZeroGPU-Mini-Trainer/
# Working Demo ( till yet) : https://huggingface.co/spaces/Alibrown/ZeroGPU-Mini-Trainer
# License: AGPL Version 3! Free for Humanity! 
# Cheap AI traning :D Fuck off :D 
import math
import os
import re
import tempfile
import time

import gradio as gr
import spaces
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODELS = [ # only exmples you can chose what you want... you need only quota :(
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
] # please setup to your own defaults, normaly you dont need but if you want :D i made it clickable for a female cat in our team. 
DEFAULT_DATASET = "Alibrown/universal-ai-hub" # gilt nur für den Besitzer ändere das für dich mimi!
DEFAULT_TARGET = "Alibrown/universal-ai-hub"  # gilt nur für den Besitzer, ändere das für dich mimi
OWNER = "Alibrown" # gilt nur für den Besitzer, ändere das für dich mimi
MAX_LEN = 384
BATCH_SIZE = 8


def clean_id(s: str) -> str:
    """Akzeptiert URL oder ID: https://huggingface.co/datasets/user/name -> user/name"""
    s = (s or "").strip().rstrip("/")
    s = re.sub(r"^https?://huggingface\.co/", "", s)
    s = re.sub(r"^(datasets|models)/", "", s)
    s = re.sub(r"/(tree|blob|viewer)/.*$", "", s)
    return s


def build_texts(ds, tok, text_col):
    cols = ds.column_names
    texts = []
    for row in ds:
        if "messages" in cols:
            t = tok.apply_chat_template(row["messages"], tokenize=False)
        elif text_col and text_col in cols:
            t = str(row[text_col])
        elif "text" in cols:
            t = str(row["text"])
        else:
            q = next((row[c] for c in ("instruction", "prompt", "question", "input") if c in cols and row[c]), None)
            a = next((row[c] for c in ("output", "response", "answer", "completion") if c in cols and row[c]), None)
            if q is None or a is None:
                raise gr.Error(
                    f"Spalten nicht erkannt: {cols}. Nutze 'messages', 'text' oder instruction/output "
                    "(oder gib den Spaltennamen an)."
                )
            t = tok.apply_chat_template(
                [{"role": "user", "content": str(q)}, {"role": "assistant", "content": str(a)}],
                tokenize=False,
            )
        texts.append(t)
    return texts


def gpu_duration(model, input_ids, attention_mask, seconds, epochs, lr, out_dir):
    return int(seconds) + 20


@spaces.GPU(duration=gpu_duration)
def train_gpu(model, input_ids, attention_mask, seconds, epochs, lr, out_dir):
    model.to("cuda")
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    n = input_ids.size(0)
    start = time.time()
    losses, stop = [], False
    for _ in range(int(epochs)):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            if time.time() - start > seconds:
                stop = True
                break
            idx = perm[i : i + BATCH_SIZE]
            ids = input_ids[idx].to("cuda")
            am = attention_mask[idx].to("cuda")
            labels = ids.clone()
            labels[am == 0] = -100
            loss = model(input_ids=ids, attention_mask=am, labels=labels).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())
        if stop:
            break
    model.save_pretrained(out_dir)
    return losses, time.time() - start, stop


def run(
    base, dataset_in, split, text_col, max_examples, epochs, seconds, lr, rank,
    target_repo, manual_token, profile: gr.OAuthProfile | None, oauth_token: gr.OAuthToken | None,
    progress=gr.Progress(),
):
    token = (manual_token or "").strip() or (oauth_token.token if oauth_token else None)
    if not token:
        raise gr.Error("Bitte oben mit Hugging Face einloggen (oder Token eingeben).")
    api = HfApi(token=token)
    user = profile.username if profile else api.whoami()["name"]
    dataset_id = clean_id(dataset_in)
    if not dataset_id:
        raise gr.Error("Dataset-URL oder -ID fehlt.")
    if clean_id(target_repo):
        repo = clean_id(target_repo)
    elif user.lower() == OWNER.lower():
        repo = DEFAULT_TARGET
    else:
        repo = f"{user}/{base.split('/')[-1]}-ft"

    yield "Lade Tokenizer & Dataset (CPU, zählt nicht zur GPU-Zeit) ..."
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = load_dataset(dataset_id, split=f"{split}[:{int(max_examples)}]", token=token)
    texts = build_texts(ds, tok, text_col)
    enc = tok(texts, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")

    yield f"{len(texts)} Beispiele bereit. Lade Modell + LoRA (CPU) ..."
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    cfg = LoraConfig(
        r=int(rank), lora_alpha=int(rank) * 2, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)

    out_dir = tempfile.mkdtemp()
    yield f"Training startet (max. {int(seconds)} s GPU-Zeit) ..."
    losses, took, stopped = train_gpu(
        model, enc["input_ids"], enc["attention_mask"], float(seconds), int(epochs), float(lr), out_dir
    )
    if not losses:
        raise gr.Error("Kein Trainingsschritt gelaufen (GPU-Zeit zu kurz?).")
    first = sum(losses[:5]) / len(losses[:5])
    last = sum(losses[-5:]) / len(losses[-5:])

    yield f"Fertig trainiert ({len(losses)} Steps, {took:.0f} s). Lade hoch nach {repo} ..."
    api.create_repo(repo, exist_ok=True, repo_type="model")
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(
            f"---\nbase_model: {base}\nlibrary_name: peft\ntags:\n- lora\n- zerogpu-trainer\n---\n"
            f"LoRA-Adapter auf `{base}`, trainiert mit `{dataset_id}`.\n"
        )
    api.upload_folder(folder_path=out_dir, repo_id=repo, repo_type="model")
    note = " (Zeitlimit erreicht, Training vorzeitig beendet)" if stopped else ""
    yield (
        f"Fertig{note}\n"
        f"Steps: {len(losses)} | Loss: {first:.3f} → {last:.3f}\n"
        f"Adapter: https://huggingface.co/{repo}"
    )


def load_repos(profile: gr.OAuthProfile | None, oauth_token: gr.OAuthToken | None):
    """Listet die Datasets und Modelle des eingeloggten Nutzers für die Auswahl."""
    if not oauth_token or not profile:
        return gr.update(), gr.update(), "Noch nicht eingeloggt."
    try:
        api = HfApi(token=oauth_token.token)
        ds = sorted(d.id for d in api.list_datasets(author=profile.username, limit=200))
        md = sorted(m.id for m in api.list_models(author=profile.username, limit=200))
    except Exception as e:  # noqa: BLE001
        return gr.update(), gr.update(), f"Konnte Repos nicht laden: {e}"
    ds_choices = ds + ([DEFAULT_DATASET] if DEFAULT_DATASET not in ds else [])
    return (
        gr.update(choices=ds_choices),
        gr.update(choices=md),
        f"Eingeloggt als **{profile.username}** – {len(ds)} Datasets, {len(md)} Modelle gefunden.",
    )


with gr.Blocks(title="ZeroGPU Mini-Trainer") as demo:
    gr.Markdown(
        "# ⚡ ZeroGPU Mini-Trainer\n"
        "Fine-Tune ein kleines SmolLM per LoRA in wenigen Minuten – mit deiner eigenen ZeroGPU-Tagesquote."
    )
    gr.LoginButton()
    with gr.Row():
        refresh = gr.Button("🔄 Meine Datasets & Modelle laden")
        repo_info = gr.Markdown("Logge dich ein, dann erscheinen deine Repos in den Auswahlfeldern.")
    with gr.Row():
        base = gr.Dropdown(BASE_MODELS, value=BASE_MODELS[1], label="Basismodell", allow_custom_value=True)
        dataset_in = gr.Dropdown(
            choices=[DEFAULT_DATASET], value=DEFAULT_DATASET, allow_custom_value=True,
            label="Dataset (auswählen oder URL / user/name einfügen)",
        )
    with gr.Row():
        split = gr.Textbox(value="train", label="Split")
        text_col = gr.Textbox(label="Textspalte (optional)", placeholder="leer = automatisch")
        max_examples = gr.Slider(50, 5000, value=1000, step=50, label="Max. Beispiele")
    with gr.Row():
        epochs = gr.Slider(1, 5, value=2, step=1, label="Epochen")
        seconds = gr.Slider(30, 240, value=150, step=10, label="Max. GPU-Sekunden")
        lr = gr.Number(value=2e-4, label="Learning Rate")
        rank = gr.Slider(4, 64, value=16, step=4, label="LoRA Rank")
    target_repo = gr.Dropdown(
        choices=[], value=None, allow_custom_value=True,
        label="Ziel-Modell-Repo (auswählen oder eintippen; leer = automatisch)",
    )
    manual_token = gr.Textbox(label="Optional: HF-Token (nur falls Login nicht geht)", type="password")
    btn = gr.Button("Training starten", variant="primary")
    out = gr.Textbox(label="Status", lines=6)
    btn.click(
        run,
        [base, dataset_in, split, text_col, max_examples, epochs, seconds, lr, rank, target_repo, manual_token],
        out,
    )

    refresh.click(load_repos, None, [dataset_in, target_repo, repo_info])
    demo.load(load_repos, None, [dataset_in, target_repo, repo_info])

demo.queue().launch()
