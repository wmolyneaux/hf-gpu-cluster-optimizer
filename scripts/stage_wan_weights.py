"""Stage Wan 2.2 VACE-Fun weights + Lightning LoRAs onto the swarmcrp-wan-weights volume.

Datacenter-side transfer (HF -> Modal volume), per SwarmCRP RENDER_CONTRACT M-4:
a local copy would never be read; staging locally makes the upload strictly slower.

Run (operator):
    modal run scripts/stage_wan_weights.py

Idempotent: files already on the volume at the right byte size are skipped.
Prints the byte size of every staged file for C-2 (take-cache key needs real sizes).
"""
from __future__ import annotations

import os
from pathlib import Path

import modal

app = modal.App("swarmcrp-wan-weight-staging")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub[hf_transfer]==0.36.0"
).env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})

# create_if_missing=False ON PURPOSE (harness_patch.md sec.4.1): a typo'd volume
# name must fail at launch, not mount empty.
weights_volume = modal.Volume.from_name("swarmcrp-wan-weights", create_if_missing=False)

MOUNT = "/wan_models"

FILES = [
    # (repo_id, path_in_repo, dest_rel)
    ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
     "split_files/diffusion_models/wan2.2_fun_vace_high_noise_14B_fp8_scaled.safetensors",
     "wan2.2_fun_vace_high_noise_14B_fp8_scaled.safetensors"),
    ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
     "split_files/diffusion_models/wan2.2_fun_vace_low_noise_14B_fp8_scaled.safetensors",
     "wan2.2_fun_vace_low_noise_14B_fp8_scaled.safetensors"),
    ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
     "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
     "umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
    ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
     "split_files/vae/wan_2.1_vae.safetensors",
     "wan_2.1_vae.safetensors"),
    ("lightx2v/Wan2.2-Lightning",
     "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors",
     "loras/lightning_t2v_a14b_4step_seko_v1.1_high.safetensors"),
    ("lightx2v/Wan2.2-Lightning",
     "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors",
     "loras/lightning_t2v_a14b_4step_seko_v1.1_low.safetensors"),
]


@app.function(image=image, volumes={MOUNT: weights_volume}, timeout=5400, cpu=4)
def stage() -> list[tuple[str, int]]:
    from huggingface_hub import hf_hub_download

    staged: list[tuple[str, int]] = []
    for repo_id, rel, dest_rel in FILES:
        dest = Path(MOUNT) / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 0:
            print(f"SKIP (exists): {dest_rel} {dest.stat().st_size}", flush=True)
            staged.append((dest_rel, dest.stat().st_size))
            continue
        if dest.is_symlink():
            dest.unlink()  # dangling link from a prior buggy attempt
        print(f"DOWNLOADING {repo_id}/{rel} ...", flush=True)
        local = hf_hub_download(repo_id=repo_id, filename=rel)
        # hf_hub_download returns a SYMLINK into the blob cache; move the real
        # bytes. Volume mount is a different filesystem (os.replace -> EXDEV).
        import shutil
        shutil.move(os.path.realpath(local), dest)
        size = dest.stat().st_size
        print(f"STAGED {dest_rel} {size} bytes", flush=True)
        staged.append((dest_rel, size))
    weights_volume.commit()
    print("COMMIT OK", flush=True)
    return staged


@app.local_entrypoint()
def main() -> None:
    for name, size in stage.remote():
        print(f"{size:>14d}  {name}")
    print("STAGING COMPLETE")
