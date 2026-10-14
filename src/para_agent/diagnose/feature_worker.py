"""Persistent, isolated calls into the existing native 29-feature extractor."""
import fcntl
import importlib
import os
from pathlib import Path
import sys

from dotenv import load_dotenv


def _safe_error(exc):
    message = str(exc)
    for key, value in os.environ.items():
        if value and any(marker in key.upper() for marker in ("API_KEY", "TOKEN", "SECRET", "PASSWORD")):
            message = message.replace(value, "[redacted]")
    return {"error_type": type(exc).__name__, "error": message}


def run_worker(connection, config):
    try:
        sys.dont_write_bytecode = True
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        load_dotenv(config["env_file"], override=False)
        import torch
        if config.get("gpu") is not None:
            gpu = config["gpu"]
            if not torch.cuda.is_available() or gpu >= torch.cuda.device_count():
                raise RuntimeError(f"Requested feature GPU {gpu} is unavailable")
            torch.cuda.set_device(gpu)
        package = str(Path(config["package_path"]).resolve())
        sys.path.insert(0, package)
        module_name, function_name = config["entrypoint"].rsplit(".", 1)
        module = importlib.import_module(module_name)
        if not Path(module.__file__).resolve().is_relative_to(Path(package)):
            raise RuntimeError("Imported feature extractor is outside the configured package")
        extract = getattr(module, function_name)
        while True:
            text = connection.recv()
            if text is None:
                return
            try:
                # Same lock as the existing FEAK adapter; never run its native libraries concurrently.
                with Path("/tmp/feak_feature_extract.lock").open("a+") as lock:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                    try:
                        result = extract(text, explain=True, progress=False)
                    finally:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                connection.send({"ok": True, "details": result,
                                 "device": f"cuda:{torch.cuda.current_device()}"
                                 if torch.cuda.is_available() else "cpu"})
            except Exception as exc:
                connection.send({"ok": False, **_safe_error(exc)})
    except EOFError:
        pass
    except Exception as exc:
        connection.send({"ok": False, **_safe_error(exc)})
    finally:
        connection.close()


class FeatureWorker:
    def __init__(self, config):
        self.config = config
        self.process = self.connection = None

    def __call__(self, text):
        if self.process is None:
            import multiprocessing
            context = multiprocessing.get_context("spawn")
            self.connection, child = context.Pipe()
            self.process = context.Process(target=run_worker, args=(child, self.config.model_dump()), daemon=True)
            self.process.start()
            child.close()
        try:
            self.connection.send(text)
            if not self.connection.poll(self.config.timeout_s):
                self.close()
                raise TimeoutError(f"Feature extraction exceeded {self.config.timeout_s} seconds")
            result = self.connection.recv()
            if not result["ok"]:
                raise RuntimeError(f"Feature extraction failed: {result['error_type']}: {result['error']}")
            return result
        except (EOFError, BrokenPipeError, ConnectionResetError) as exc:
            self.close()
            raise RuntimeError("Native feature worker exited without a result") from exc

    def close(self):
        if self.process is None:
            return
        try:
            self.connection.send(None)
        except (BrokenPipeError, EOFError, OSError):
            pass
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        self.connection.close()
        self.connection = self.process = None
