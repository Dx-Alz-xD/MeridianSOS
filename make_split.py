import glob, re, random, shutil, collections
from pathlib import Path
SRC = Path("/home/claude/data/Multi-Weather Pothole Detection (MWPD)/MWPD")
DST = Path("/home/claude/pothole_ds")
MAX_TRAIN_COPIES = 2
def src(p): return re.sub(r"\.rf\..*", "", Path(p).name)
groups = collections.defaultdict(list)
for s in ["train", "valid", "test"]:
    for p in sorted(glob.glob(str(SRC / s / "images" / "*.jpg"))):
        groups[src(p)].append(Path(p))
ids = sorted(groups); random.Random(42).shuffle(ids)
n = len(ids); nv = int(.12*n); nt = int(.12*n)
split_of = {}
for i, g in enumerate(ids):
    split_of[g] = "val" if i < nv else "test" if i < nv+nt else "train"
if DST.exists(): shutil.rmtree(DST)
cnt = collections.Counter(); boxes = collections.Counter()
for g, files in groups.items():
    sp = split_of[g]
    keep = files[:MAX_TRAIN_COPIES] if sp == "train" else files[:1]
    for img in keep:
        lab = img.parent.parent / "labels" / (img.stem + ".txt")
        (DST/"images"/sp).mkdir(parents=True, exist_ok=True); (DST/"labels"/sp).mkdir(parents=True, exist_ok=True)
        shutil.copy(img, DST/"images"/sp/img.name); shutil.copy(lab, DST/"labels"/sp/lab.name)
        cnt[sp] += 1; boxes[sp] += len([l for l in open(lab) if l.strip()])
(DST/"data.yaml").write_text(f"path: {DST}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: pothole\n")
print("source groups:", n, "| images:", dict(cnt), "| boxes:", dict(boxes))
