import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple


Task = Tuple[Callable[..., Any], tuple, dict]


@dataclass
class PoolStats:
    submitted: int = 0
    processed: int = 0
    success: int = 0
    failed: int = 0


class ThreadWorkerPool:
    def __init__(
        self,
        workers: int = 20,
        max_queue_size: int = 10000,
        error_backoff_sec: float = 0.1,
        name: str = "pool",
    ) -> None:
        if workers <= 0:
            raise ValueError("workers must be > 0")
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be > 0")
        if error_backoff_sec < 0:
            raise ValueError("error_backoff_sec must be >= 0")

        self.workers = workers
        self.error_backoff_sec = error_backoff_sec
        self.name = name
        self._queue: queue.Queue[Task] = queue.Queue(maxsize=max_queue_size)
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()
        self._report_stop_event = threading.Event()
        self._reporter: Optional[threading.Thread] = None
        self._stats = PoolStats()
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        for idx in range(self.workers):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"{self.name}-worker-{idx + 1}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        if self._stop_event.is_set():
            raise RuntimeError("pool is stopping, cannot submit new tasks")
        if not self._started:
            raise RuntimeError("pool not started, call start() first")
        self._queue.put((fn, args, kwargs))
        with self._lock:
            self._stats.submitted += 1

    def wait_for_all(self) -> None:
        self._queue.join()

    def stop(self, wait: bool = True) -> None:
        self._stop_event.set()
        self._report_stop_event.set()
        if wait:
            for thread in self._threads:
                thread.join(timeout=2.0)
            if self._reporter is not None:
                self._reporter.join(timeout=2.0)

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return {
                "submitted": self._stats.submitted,
                "processed": self._stats.processed,
                "success": self._stats.success,
                "failed": self._stats.failed,
                "queue_size": self._queue.qsize(),
            }

    def start_reporter(
        self,
        interval_sec: float = 2.0,
        printer: Callable[[str], None] = print,
    ) -> None:
        if interval_sec <= 0:
            raise ValueError("interval_sec must be > 0")
        if self._reporter is not None and self._reporter.is_alive():
            return

        self._report_stop_event.clear()

        def _report_loop() -> None:
            last_processed = -1
            while not self._report_stop_event.wait(interval_sec):
                snap = self.snapshot()
                if snap["processed"] == last_processed:
                    continue
                last_processed = snap["processed"]
                rate = (
                    (snap["success"] / snap["processed"]) * 100
                    if snap["processed"]
                    else 0.0
                )
                printer(
                    f"[POOL] submitted={snap['submitted']} processed={snap['processed']} "
                    f"success={snap['success']} failed={snap['failed']} "
                    f"queue={snap['queue_size']} success_rate={rate:.2f}%"
                )

        self._reporter = threading.Thread(
            target=_report_loop,
            name=f"{self.name}-reporter",
            daemon=True,
        )
        self._reporter.start()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                fn, args, kwargs = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            ok = True
            try:
                fn(*args, **kwargs)
            except Exception:
                ok = False
                if self.error_backoff_sec > 0:
                    time.sleep(self.error_backoff_sec)
            finally:
                with self._lock:
                    self._stats.processed += 1
                    if ok:
                        self._stats.success += 1
                    else:
                        self._stats.failed += 1
                self._queue.task_done()
