from __future__ import annotations

import re
from threading import Event, Lock, Thread
from typing import Any

from app.services.water_quality_service import water_quality_service

_VALUE = r"([-+]?(?:\d+\.?\d*|\.\d+))"


def parse_firmware_line(line: str) -> dict[str, float | None] | None:
    if "Temp:" not in line or "pH:" not in line or "Turbidity:" not in line:
        return None
    segments = line.split("|")

    def value_for(segment: str, pattern: str) -> float | None:
        if "[INVALID]" in segment:
            return None
        match = re.search(pattern, segment, re.IGNORECASE)
        return float(match.group(1)) if match else None

    temperature = value_for(segments[0], rf"Temp:\s*{_VALUE}")
    ph = value_for(segments[1], rf"pH:\s*{_VALUE}")
    turbidity = value_for(segments[2], rf"Turbidity:\s*{_VALUE}\s*idx")
    return {"temperature_c": temperature, "ph": ph, "turbidity": turbidity}


class Esp32SerialReader:
    def __init__(self):
        self._lock = Lock()
        self._stop = Event()
        self._thread: Thread | None = None
        self._status: dict[str, Any] = {"connected": False, "port": None, "error": None, "last_line": None}

    def start(self, port: str) -> dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                if self._status["port"] == port:
                    return dict(self._status)
                raise RuntimeError("Disconnect the current ESP32 serial port before selecting another")
            self._stop.clear()
            self._status = {"connected": False, "port": port, "error": None, "last_line": None}
            self._thread = Thread(target=self._read_loop, args=(port,), daemon=True)
            self._thread.start()
        return self.status()

    def _read_loop(self, port: str) -> None:
        try:
            import serial

            device = serial.Serial()
            device.port = port
            device.baudrate = 115200
            device.timeout = 1
            device.dtr = False
            device.rts = False
            device.open()
            with self._lock:
                self._status.update(connected=True, error=None)
            with device:
                while not self._stop.is_set():
                    line = device.readline().decode("ascii", errors="ignore").strip()
                    if not line:
                        continue
                    reading = parse_firmware_line(line)
                    with self._lock:
                        self._status["last_line"] = line
                    if reading is not None:
                        water_quality_service.record(reading)
        except Exception as exc:
            with self._lock:
                self._status.update(connected=False, error=str(exc))
        finally:
            with self._lock:
                self._status["connected"] = False

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2)
        with self._lock:
            self._thread = None
            self._status.update(connected=False)
            return dict(self._status)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)


serial_reader = Esp32SerialReader()