"""Submit a finite batch into one already allocated runq, or deliver a native OBS batch."""

import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import zlib

import tide_cloud_relay as relay


def reply(status, reason):
    return {"status": status, "reason": reason}


def identity(node, claim, transport):
    relay.check_identity(claim)
    relay.require(claim["job"] == node["name"] and claim["transport"] == transport, "frozen job or transport differs")
    return {k: claim[k] for k in ("attempt", "job", "task", "fingerprint", "digest", "transport")}


def ticket(claim, job_id):
    return {**claim, "job_id": job_id, "schema": "tide-cloud-route/v1"}


def wire(obj):
    return (json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()


def cache_envelope(native, claim, body):
    target = relay.member(native, "envelopes/" + claim["attempt"] + ".json")
    target.parent.mkdir(mode=0o700, exist_ok=True)
    info = target.parent.stat()
    relay.require(info.st_uid == os.getuid() and not info.st_mode & 0o022, "untrusted SFS descriptor directory")
    if target.is_file():
        relay.require(target.read_bytes() == body, "SFS descriptor conflicts with existing attempt")
        return target
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as f:
        temp = Path(f.name)
        try:
            f.write(body)
            f.flush()
            os.fsync(f.fileno())
            os.chmod(temp, 0o400)
            try:
                os.link(temp, target)
            except FileExistsError:
                relay.require(target.read_bytes() == body, "concurrent SFS descriptor differs")
        finally:
            temp.unlink(missing_ok=True)
    return target


def resource_args(d):
    relay.require(d["backend"] in {"cpu", "npu"} and all(type(d[k]) is int and d[k] > 0 for k in ("threads", "ram_mib"))
                  and type(d["priority"]) is int, "invalid task resources")
    relay.require(d.get("estimated_seconds") is None or
                  type(d["estimated_seconds"]) is int and d["estimated_seconds"] > 0, "invalid runtime estimate")
    args = ["--worker", 0, "--key", d["attempt"], "--backend", d["backend"], "--threads", d["threads"],
            "--ram-mib", d["ram_mib"], "--priority", d["priority"]]
    if d.get("estimated_seconds"):
        args.extend(["--estimated-seconds", d["estimated_seconds"]])
    if d["backend"] == "npu":
        relay.require(type(d["hbm_mib"]) is int and d["hbm_mib"] > 0 and type(d["reserve_chip"]) is bool,
                      "invalid NPU resources")
        args.extend(["--hbm-mib", d["hbm_mib"]])
        if d["reserve_chip"]:
            args.append("--reserve-chip")
    return list(map(str, args))


def verified_queue(node, work, obs, shared, mount_check=os.path.ismount):
    native, release = relay.allocation_paths(node["name"], work, obs, shared, mount_check)
    expected = relay.read(relay.member(work, "job.json"))["runq"]
    installed = relay.read(relay.member(work, "tools/runq/.maflow-release.json"))
    relay.require(installed == {k: expected[k] for k in ("commit", "bytes", "sha256")} and
                  expected["sha256"] == release["runq_sha256"] == os.environ["MAFLOW_RUNQ_SHA256"]
                  and release["runq_commit"] == os.environ["MAFLOW_RUNQ_COMMIT"], "unexpected frozen runq")
    root = relay.member(work, "payload")
    relay.require(relay.digest(relay.member(root, "tide_cloud_node.py")) == release["files"]["tide_cloud_node.py"],
                  "frozen worker file differs")
    exe = relay.member(work, "bin/runq")
    relay.require(Path(os.environ["MAFLOW_RUNQ"]) == exe and exe.is_file(), "missing verified runq executable")
    wrapper = exe.read_text().splitlines()
    relay.require(len(wrapper) == 2 and wrapper[0] == "#!/bin/sh", "unrecognized frozen runq wrapper")
    parts = shlex.split(wrapper[1])
    relay.require(len(parts) == 6 and parts[0] == "exec"
                  and parts[2:] == ["-I", "-B", str(relay.member(work, "tools/runq/launcher.py")), "$@"]
                  and Path(parts[1]).is_absolute() and os.access(parts[1], os.X_OK), "unverified worker interpreter")
    queue = relay.member(native, "runq")
    registered = relay.read(relay.member(work, "runq-project.json"))
    cluster = relay.member(queue, ".runq-cluster")
    relay.require(registered["mode"] == "project" and registered["shared"] == str(queue)
                  and relay.read(relay.member(cluster, "allocation.json"))["workers"] == 1,
                  "project runq has not registered this allocation")
    return native, root, exe, parts[1], queue, cluster, float(registered["deadline"])


def remote_batch(node, claims, work, obs, shared, *, mount_check=os.path.ismount, runner=subprocess.run):
    results = {}
    try:
        native, root, exe, python, queue, cluster, deadline = verified_queue(node, work, obs, shared, mount_check)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {c["attempt"]: reply("retry", str(exc)) for c in claims}

    for claim in claims:
        attempt = claim["attempt"]
        try:
            ident = identity(node, claim, "ssh")
            delivery = claim.get("delivery")
            if delivery:
                route = delivery["route"]
                body = base64.b64decode(delivery["body"], validate=True)
            else:
                route = relay.read(relay.member(obs, "inputs/routes/" + attempt + ".json"))
                body = relay.member(obs, "inputs/authority/" + attempt + ".json").read_bytes()
            if route != ticket(ident, node["id"]):
                results[attempt] = reply("retry", "regional ticket does not identify this SSH request")
                continue
            if hashlib.sha256(body).hexdigest() != ident["digest"]:
                results[attempt] = reply("retry", "mounted authority has not exposed the authenticated descriptor bytes")
                continue
            d = relay.check_descriptor(body, claim)
            inputs = resource_args(d)
            if time.time() >= deadline and not (cluster / "requests" / (hashlib.sha256(attempt.encode()).hexdigest() + ".json")).is_file():
                results[attempt] = reply("retry", "registered allocation deadline reached")
                continue
            envelope = cache_envelope(native, claim, body)
            cmd = [str(exe), "--json", "cluster", "submit", "--shared", str(queue), "--cwd", str(root), *inputs, "--",
                   python, "-B", "-u", str(root / "tide_cloud_node.py"), "worker", str(envelope), claim["digest"]]
            run = runner(cmd, capture_output=True, text=True, timeout=20)
            if run.returncode:
                message = (run.stderr or run.stdout or "runq submit failed").strip()[-700:]
                status = "rejected" if "different cluster request" in message else "retry"
                results[attempt] = reply(status, message)
                continue
            result = json.loads(run.stdout)
            if isinstance(result, dict) and result.get("key") == attempt and result.get("worker") == 0:
                results[attempt] = reply("submitted", "cluster accepted the exact attempt key")
            else:
                results[attempt] = reply("retry", "runq acknowledgement has not confirmed this exact attempt")
        except (OSError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
            results[attempt] = reply("retry", str(exc))
        except (ValueError, KeyError, TypeError) as exc:
            results[attempt] = reply("rejected", str(exc))
    return results


def prepare(node, claims, region, transport):
    """Fence the two different immutable cluster argv forms, including after a lost response."""
    prefix = node["prefix"]
    ready, completed = [], {}

    def put_immutable(key, body):
        try:
            region.put_bytes(key, body)
        except (OSError, RuntimeError, http.client.HTTPException):
            # A conditional PUT can race another route or lose its response after commit.
            if not region.head(key):
                raise
            relay.require(region.get_bytes(key) == body, "different immutable object at " + key)

    def one(c):
        try:
            ident = identity(node, c, transport)
            dispatch = prefix + "/inputs/dispatch/" + ident["attempt"] + ".json"
            route = prefix + "/inputs/routes/" + ident["attempt"] + ".json"
            if transport == "ssh" and not region.head(route) and region.head(dispatch):
                relay.check_descriptor(region.get_bytes(dispatch), ident)
                return ident, reply("submitted", "identical descriptor already published through legacy OBS")
            route_ticket = ticket(ident, node["id"])
            put_immutable(route, wire(route_ticket))
            raw = region.get_bytes(prefix + "/inputs/authority/" + ident["attempt"] + ".json")
            relay.check_descriptor(raw, ident)
            if transport == "obs":
                put_immutable(dispatch, raw)
                return ident, reply("submitted", "native OBS descriptor is published for the frozen scanner")
            # The original ModelArts object mount can keep negative directory entries
            # after the regional API accepted them. Carry the fixed bytes over SSH.
            return {**ident, "delivery":{"route":route_ticket, "body":base64.b64encode(raw).decode()}}, None
        except (FileExistsError, ValueError, KeyError, TypeError) as exc:
            return c, reply("rejected", str(exc))
        except (OSError, RuntimeError, http.client.HTTPException) as exc:
            return c, reply("retry", str(exc))
    with ThreadPoolExecutor(max_workers=8) as pool:
        for c, result in pool.map(one, claims):
            if result:
                completed[c["attempt"]] = result
            else:
                ready.append(c)
    return ready, completed


def submit_native(node, claims, region):
    """Publish the authority bytes only for attempts initially assigned the native OBS route."""
    _, result = prepare(node, claims, region, "obs")
    return result


def remote_call(node, claims, *, probe=False):
    from tide_cloud_allocation import MAFLOW
    # The allocated payload remains immutable; this command lasts only for this batch.
    bootstrap = (
        "import base64,json,sys,types,zlib;"
        "v=json.loads(zlib.decompress(base64.b64decode(sys.argv[1] if sys.argv[1]!='-' else sys.stdin.read())));"
        "m=types.ModuleType('tide_cloud_relay');sys.modules[m.__name__]=m;exec(v['relay'],m.__dict__);"
        "p={'__name__':'tide_cloud_submit'};exec(v['submit'],p);"
        "c=v['node'];paths=m.runtime_paths(c['name'],c['id']);"
        "s=({'ready':bool(p['verified_queue'](c,*paths))} if v['probe'] else p['remote_batch'](c,v['claims'],*paths));"
        "print(json.dumps(s,ensure_ascii=False))"
    )
    package = {"relay": Path(relay.__file__).read_text(), "submit": Path(__file__).read_text(),
               "node": {k: node[k] for k in ("name", "id")}, "claims": claims, "probe": probe}
    encoded = base64.b64encode(zlib.compress(wire(package))).decode()
    inline = probe
    cmd = [sys.executable, "-B", str(MAFLOW), "job", "exec", "--json", "--node", "0", "guiyang/" + node["id"], "--",
           "python3", "-I", "-B", "-c", bootstrap, encoded if inline else "-"]
    result = subprocess.run(cmd, input=None if inline else encoded, capture_output=True, text=True,
                            timeout=35 if probe else 90 + 22 * len(claims))
    relay.require(result.stdout, "ModelArts SSH command unavailable: " + result.stderr[-400:])
    response = json.loads(result.stdout)
    relay.require(response["job_id"] == node["id"] and response.get("node") == 0,
                  "no authenticated execution result")
    relay.require(result.returncode == 0 and response["exit_code"] == 0,
                  "ModelArts SSH command failed: " + (response.get("stderr") or result.stderr)[-500:])
    result = json.loads(response["stdout"])
    relay.require(isinstance(result, dict), "invalid remote batch response")
    return result


def ssh_ready(node):
    """Check the actual frozen queue through this job's authenticated SSH environment before selecting its route."""
    try:
        return remote_call(node, [], probe=True) == {"ready": True}
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        return False


def submit_batch(node, claims):
    """Return per-attempt results. A lost SSH response retries the identical SFS argv and key."""
    from tide_cloud_allocation import Store
    try:
        ready, results = prepare(node, claims, Store.named("guiyang"), "ssh")
    except (OSError, ValueError, KeyError, RuntimeError, http.client.HTTPException) as exc:
        return {c["attempt"]: reply("retry", str(exc)) for c in claims}
    if ready:
        try:
            observed = remote_call(node, ready)
            for c in ready:
                val = observed.get(c["attempt"], {})
                results[c["attempt"]] = (val if isinstance(val, dict) and val.get("status") in {"submitted", "retry", "rejected"}
                                         else reply("retry", "no verified reply for this key"))
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
            results.update({c["attempt"]: reply("retry", str(exc)) for c in ready})
    return results
