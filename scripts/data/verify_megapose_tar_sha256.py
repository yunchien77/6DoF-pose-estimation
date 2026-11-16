"""Check every downloaded MegaPose shard (train_pbr_tar/shard-XXXXXX.tar) against the size and sha256 that Hugging Face lists for it.
A match means the file is byte for byte the shard on Hugging Face, i.e. the same file get_train_data.sh downloaded and extracted.
Standard library only; run from the repo root, inside tmux (it reads every byte: ~544 GB for GSO, ~567 GB for ShapeNetCore):

  python scripts/data/verify_megapose_tar_sha256.py --groups GSO; echo "exit code: $?"

  --jobs 2      shards hashed at the same time (more does not help on one hard disk)
  Interrupted? Run the same line again: shards already found correct (same size and modification time) are not hashed again
  (kept in MegaPose-<G>/train_pbr_tar_sha256_ok.txt).
Exit status 0 = every shard of the given groups is present and correct.
"""
import argparse
import hashlib
import json
import os
import os.path as osp
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API = "https://huggingface.co/api/datasets/bop-benchmark/megapose/tree/main/MegaPose-{g}"
N_SHARDS = 1040


def hf_listing(g):
    """-> {shard number: (size, sha256)} from the Hugging Face file listing (paginated, 1000 per page)"""
    out, url = {}, API.format(g=g)
    while url:
        req = urllib.request.Request(url, headers={"User-Agent": "megapose-tar-check"})
        with urllib.request.urlopen(req, timeout=60) as r:
            page = json.load(r)
            link = r.headers.get("Link") or ""
        for x in page:
            m = re.search(r"/shard-(\d{6})\.tar$", x.get("path", ""))
            if m and "lfs" in x:
                out[int(m.group(1))] = (int(x["lfs"]["size"]), x["lfs"]["oid"])
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(16 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def check_group(data_dir, g, jobs):
    want = hf_listing(g)
    if len(want) != N_SHARDS:
        print(f"{g}: Hugging Face lists {len(want)} shards, expected {N_SHARDS}")
        return False
    d = osp.join(data_dir, f"MegaPose-{g}", "train_pbr_tar")
    okfile = osp.join(data_dir, f"MegaPose-{g}", "train_pbr_tar_sha256_ok.txt")
    done = {}
    if osp.exists(okfile):
        for line in open(okfile):
            s, size, mt, h = line.split()
            done[int(s)] = (int(size), int(mt), h)
    missing, bad_size, todo = [], [], []
    for s, (size, h) in sorted(want.items()):
        p = osp.join(d, f"shard-{s:06d}.tar")
        if not osp.exists(p):
            missing.append(s)
            continue
        st = os.stat(p)
        if st.st_size != size:
            bad_size.append(s)
            print(f"  WRONG SIZE {g} {s:06d}: {st.st_size} bytes, Hugging Face {size}")
            continue
        if done.get(s) == (size, st.st_mtime_ns, h):
            continue
        todo.append((s, p, size, st.st_mtime_ns, h))
    print(f"{g}: {N_SHARDS} shards listed, {len(missing)} missing, {len(bad_size)} wrong size, "
          f"{N_SHARDS - len(missing) - len(bad_size) - len(todo)} already checked, {len(todo)} to hash", flush=True)
    bad_hash = []
    t0 = time.time()
    nbytes = 0
    with ThreadPoolExecutor(jobs) as ex, open(okfile, "a") as ok:
        for k, ((s, p, size, mt, h), got) in enumerate(zip(todo, ex.map(lambda t: sha256(t[1]), todo))):
            nbytes += size
            if got == h:
                ok.write(f"{s} {size} {mt} {h}\n")
                ok.flush()
            else:
                bad_hash.append(s)
                print(f"  WRONG SHA256 {g} {s:06d}")
            if (k + 1) % 20 == 0 or k + 1 == len(todo):
                dt = time.time() - t0
                left = (len(todo) - k - 1) * dt / (k + 1)
                print(f"  {g} {k + 1}/{len(todo)} hashed, {nbytes / dt / 1e6:.0f} MB/s, about {left / 3600:.1f} h left", flush=True)
    print(f"{g}: missing {len(missing)}, wrong size {len(bad_size)}, wrong sha256 {len(bad_hash)}"
          + (f"  (missing e.g. {missing[:5]})" if missing else ""))
    return not (missing or bad_size or bad_hash)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="datasets/MegaPose-Training-Data")
    ap.add_argument("--groups", nargs="+", default=["GSO", "ShapeNetCore"], choices=["GSO", "ShapeNetCore"])
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    ok = all([check_group(a.data_dir, g, a.jobs) for g in a.groups])
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
