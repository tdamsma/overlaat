"""derive_name(): python-hosted inference servers get a meaningful, port-suffixed name."""

from overlaat import host_logger as hl


def test_uvicorn_in_server_dir_keeps_directory_name():
    cmd = "/srv/whisper-server/.venv/bin/python3 /srv/whisper-server/.venv/bin/uvicorn server:app --host 127.0.0.1 --port 8081"
    assert hl.derive_name(cmd) == "whisper-server-8081"


def test_mlx_lm_server_named_after_script():
    cmd = "/u/x/.local/share/uv/tools/mlx-lm/bin/python /u/x/.local/bin/mlx_lm.server --host 127.0.0.1 --port 8080 --decode-concurrency 8"
    assert hl.derive_name(cmd) == "mlx-lm-server-8080"


def test_rapid_mlx_named_after_script():
    cmd = "/u/x/.local/share/uv/tools/rapid-mlx/bin/python /u/x/.local/bin/rapid-mlx --no-telemetry serve org/model --host 127.0.0.1 --port 8087"
    assert hl.derive_name(cmd) == "rapid-mlx-8087"


def test_python_without_script_falls_back_to_exe_port():
    assert hl.derive_name("/usr/bin/python3 --port 9000") == "python3-9000"


def test_non_backend_exe_unchanged():
    assert hl.derive_name("/System/Library/CoreServices/WindowServer -daemon") == "WindowServer"


def test_mem_gb_prefers_footprint():
    assert hl.mem_gb({"rss_gb": 7.7, "footprint_gb": 19.9}) == 19.9
    assert hl.mem_gb({"rss_gb": 7.7, "footprint_gb": None}) == 7.7


def test_phys_footprint_of_self_is_plausible_or_none():
    import os

    fp = hl.phys_footprint_gb(os.getpid())
    assert fp is None or 0.0 < fp < 64.0
