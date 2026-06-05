"""
ExternalApiServer: a drop-in replacement for cadofactor.api_server.ApiServer
that runs the Rust work-unit server (cado-wu-server-rs, from rust/) as a
subprocess instead of serving in-process with Flask.

It speaks the same HTTP/JSON protocol and uses the *same* wudb SQLite database,
so the Python driver (cadotask.py) keeps populating AVAILABLE work-units and
reading results exactly as before, and the Python (or Rust) clients are
unaffected. Input files are served from the `server_registered_filenames` table
that the driver already maintains, so no extra plumbing is needed.

Enabled by setting the environment variable CADO_RUST_WU_SERVER to the path of
the cado-wu-server-rs binary (cadotask.py switches to this class when it is set).

Only the ApiServer methods the driver actually uses are implemented:
get_port, serve, shutdown, stop_serving_wus, get_url, get_cert_sha1.

Caveats: serves plain HTTP (use server.ssl=no); the Rust server does not (yet)
enforce server.whitelist -- run it on a trusted network / loopback.
"""

import logging
import os
import subprocess
import threading
import time

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger("ExternalApiServer")


class ExternalApiServer(object):
    def __init__(self, serveraddress, serverport, dbdata,
                 threaded=None, debug=False, uploaddir=None, nrsubdir=None,
                 only_registered=True, cafile=None, whitelist=None,
                 timeout_hint=None, **kwargs):
        binary = os.environ.get("CADO_RUST_WU_SERVER")
        if not binary:
            raise RuntimeError("CADO_RUST_WU_SERVER is not set")

        self.address = serveraddress or "localhost"
        self.dbpath = dbdata.path        # DBFactory -> SQLite file path
        self.cafile = cafile

        args = [binary, "--db", self.dbpath,
                "--addr", "127.0.0.1" if self.address == "localhost" else self.address,
                "--port", str(serverport or 0)]
        if uploaddir:
            args += ["--uploaddir", uploaddir]
        if timeout_hint:
            try:
                args += ["--wutimeout", str(int(float(timeout_hint)))]
            except (TypeError, ValueError):
                pass
        # TLS: the Python server uses a self-signed cert (cafile). The Rust
        # server needs cert+key; for now this swap supports plain HTTP only.
        if cafile:
            logger.warning("ExternalApiServer: TLS (cafile=%s) not wired; "
                           "run with server.ssl=no for the Rust server", cafile)

        logger.info("Launching Rust work-unit server: %s", " ".join(args))
        self.proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)

        # the server prints "SERVER_URL http://addr:port" once bound
        self.url = None
        deadline = time.time() + 30
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if line == "":
                if self.proc.poll() is not None:
                    raise RuntimeError("Rust work-unit server exited during startup")
                continue
            logger.info("rust-wu-server: %s", line.rstrip())
            if line.startswith("SERVER_URL "):
                self.url = line.split(None, 1)[1].strip()
                break
        if not self.url:
            self.shutdown()
            raise RuntimeError("Rust work-unit server did not report its URL")
        self.port = int(self.url.rsplit(":", 1)[1])

        # keep draining stdout so the pipe never blocks the child
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        try:
            for line in self.proc.stdout:
                logger.info("rust-wu-server: %s", line.rstrip())
        except Exception:
            pass

    # --- the ApiServer interface used by ServerTask ---

    def get_port(self):
        return self.port

    def serve(self):
        pass  # already serving since __init__

    def shutdown(self, *args):
        if getattr(self, "proc", None) is None:
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass

    def stop_serving_wus(self):
        if requests is None:
            logger.warning("requests not available; cannot signal finish")
            return
        try:
            requests.post(self.url + "/control", data={"action": "finish"}, timeout=5)
        except Exception as e:
            logger.warning("stop_serving_wus: %s", e)

    def get_url(self, origin=None, **kwargs):
        return self.url

    def get_cert_sha1(self):
        return None
