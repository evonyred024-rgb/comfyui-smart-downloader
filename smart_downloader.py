import os
import urllib.request
import threading
import re
import folder_paths

_download_status = {}

FOLDER_MAP = {
    "checkpoints":  "checkpoints",
    "vae":          "vae",
    "clip":         "clip",
    "text_encoders": "text_encoders",
    "loras":        "loras",
    "controlnet":   "controlnet",
    "unet":         "diffusion_models",
    "upscale":      "upscale_models",
    "embeddings":   "embeddings",
}


def _get_folder_path(folder_key):
    name = FOLDER_MAP.get(folder_key, folder_key)
    try:
        paths = folder_paths.get_folder_paths(name)
        if paths:
            return paths[0]
    except Exception:
        pass
    base = os.path.dirname(folder_paths.base_path)
    return os.path.join(base, "ComfyUI", "models", name)


def _filename_from_url(url, custom_name):
    if custom_name.strip():
        return custom_name.strip()
    path = url.split("?")[0].rstrip("/")
    name = path.split("/")[-1]
    return name if name else "downloaded_model.safetensors"


def _build_request(url, hf_token, civitai_token):
    headers = {"User-Agent": "Mozilla/5.0"}
    if "huggingface.co" in url or "hf.co" in url:
        if hf_token.strip():
            headers["Authorization"] = f"Bearer {hf_token.strip()}"
    elif "civitai.com" in url:
        if civitai_token.strip():
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}token={civitai_token.strip()}"
    return urllib.request.Request(url, headers=headers)


def _do_download(key, url, dest_path, hf_token, civitai_token):
    try:
        _download_status[key] = {"status": "connecting", "progress": "0%"}
        req = _build_request(url, hf_token, civitai_token)
        with urllib.request.urlopen(req) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 1024 * 1024
            cd = resp.headers.get("Content-Disposition", "")
            match = re.search(r'filename="?([^";]+)"?', cd)
            if match:
                fname = match.group(1).strip()
                dest_path = os.path.join(os.path.dirname(dest_path), fname)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as f:
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    downloaded += len(buf)
                    if total:
                        pct = int(downloaded / total * 100)
                        mb  = downloaded / (1024 * 1024)
                        tmb = total / (1024 * 1024)
                        _download_status[key] = {
                            "status": "downloading",
                            "progress": f"{pct}%  ({mb:.1f} / {tmb:.1f} MB)",
                        }
                    else:
                        mb = downloaded / (1024 * 1024)
                        _download_status[key] = {
                            "status": "downloading",
                            "progress": f"{mb:.1f} MB downloaded",
                        }
        _download_status[key] = {
            "status": "done",
            "progress": f"Saved to {dest_path}",
        }
    except urllib.error.HTTPError as e:
        msgs = {
            401: "401 Unauthorized - API token اشتباهه",
            403: "403 Forbidden - token دسترسی نداره",
            404: "404 Not Found - URL اشتباهه",
        }
        _download_status[key] = {
            "status": "error",
            "progress": msgs.get(e.code, f"HTTP {e.code}: {e.reason}"),
        }
    except Exception as e:
        _download_status[key] = {"status": "error", "progress": str(e)}


class SmartModelDownloader:
    CATEGORY = "Smart Downloader"
    FUNCTION = "download"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "https://huggingface.co/...", "multiline": False}),
                "folder": (list(FOLDER_MAP.keys()),),
                "action": (["start_download", "check_status", "list_files"],),
            },
            "optional": {
                "huggingface_token": ("STRING", {"default": "", "multiline": False}),
                "civitai_token": ("STRING", {"default": "", "multiline": False}),
                "custom_filename": ("STRING", {"default": "", "multiline": False}),
            },
        }

    def download(self, url, folder, action, huggingface_token="", civitai_token="", custom_filename=""):
        folder_path = _get_folder_path(folder)

        if action == "list_files":
            if os.path.isdir(folder_path):
                files = os.listdir(folder_path)
                result = f"Folder: {folder_path}\n"
                result += "\n".join(f"  - {f}" for f in sorted(files)) if files else "  (empty)"
            else:
                result = f"Folder not found: {folder_path}"
            return (result,)

        key = f"{folder}::{url}"

        if action == "check_status":
            info = _download_status.get(key)
            if not info:
                return ("هنوز دانلودی شروع نشده.",)
            return (f"[{info['status'].upper()}] {info['progress']}",)

        if action == "start_download":
            if not url.startswith("http"):
                return ("URL باید با http شروع بشه",)
            if _download_status.get(key, {}).get("status") == "downloading":
                return (f"[در حال دانلود] {_download_status[key]['progress']}",)
            filename = _filename_from_url(url, custom_filename)
            dest = os.path.join(folder_path, filename)
            _download_status[key] = {"status": "starting", "progress": "0%"}
            t = threading.Thread(
                target=_do_download,
                args=(key, url, dest, huggingface_token, civitai_token),
                daemon=True,
            )
            t.start()
            return (f"دانلود شروع شد: {dest}",)

        return ("Unknown action",)


NODE_CLASS_MAPPINGS = {"SmartModelDownloader": SmartModelDownloader}
NODE_DISPLAY_NAME_MAPPINGS = {"SmartModelDownloader": "Smart Model Downloader"}
