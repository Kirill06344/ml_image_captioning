"""Locale-stable process management for pycocoevalcap's unchanged METEOR 1.5 scorer.

The upstream Java statistics protocol requires a period as decimal separator.
Its Python wrapper inherits the user's locale and can deadlock during error cleanup.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import threading

from pycocoevalcap.meteor import meteor as upstream


class CocoMeteor(upstream.Meteor):
    def __init__(self):
        self.lock = threading.Lock()
        self.meteor_p = None
        self._errors = tempfile.TemporaryFile()
        folder = Path(upstream.__file__).parent
        self.meteor_cmd = ['java', '-Xmx2G', '-Duser.language=en', '-Duser.country=US',
                           '-jar', upstream.METEOR_JAR, '-', '-', '-stdio', '-l', 'en', '-norm']
        env = os.environ.copy()
        env.update({'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8'})
        try:
            self.meteor_p = subprocess.Popen(self.meteor_cmd, cwd=folder, stdin=subprocess.PIPE,
                                             stdout=subprocess.PIPE, stderr=self._errors, env=env)
        except Exception:
            self.close()
            raise

    def compute_score(self, gts, res):
        try:
            # Keep the official SCORE/EVAL protocol and corpus aggregation unchanged.
            return super().compute_score(gts, res)
        except Exception as exc:
            self._stop()
            self._errors.seek(0)
            detail = self._errors.read().decode('utf-8', errors='replace').strip()
            if self.lock.locked():
                self.lock.release()
            raise RuntimeError('METEOR Java failed: ' + (detail[-4000:] or type(exc).__name__)) from exc

    def _stop(self):
        process = self.meteor_p
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait()
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
            self.meteor_p = None

    def close(self):
        self._stop()
        if self._errors is not None:
            self._errors.close()
            self._errors = None

    def __del__(self):
        # Never acquire the upstream lock: it can remain locked after a Java error.
        try:
            self.close()
        except Exception:
            pass
