#!/bin/zsh
# Stage the OSKI heroshot inputs onto `berkeley-usd-take`.
#
# Adapted from stage_berkeley_take.sh, which hardcodes hero_0454's rig and exactly one
# retarget, and whose tar carries only (usd textures tools rig shots) -- so a character
# style card had nowhere to live. place_rig.py:138 defaults --style-card to a path that
# exists only on the M5, so without one on the volume every non-hero_0454 render either
# dies at the json.load or silently renders against HER card.
#
# THE TAR IS WHAT WORKERS READ. The individual volume puts below exist for the FUSE
# fallback; the trainer prefers busd_take.tar and untars it once to /tmp/busd. Rebuild it
# whenever ANY input changes or the fleet renders yesterday's inputs while the manifest
# refuses to notice -- place_rig hashes what it READS, which is the untarred copy.
set -e
MODAL=/Users/molyneaux/hf-gpu-cluster-optimizer/.venv/bin/modal
[ -x "$MODAL" ] || MODAL=modal
BUSD=/Users/molyneaux/Desktop/berkeley-usd
# ONE DIRECTORY PER CHARACTER -- this is the root fix, and it generalises to any card.
# place_rig.py:935 resolves rig.json as a SIBLING of --glb, so a flat /rig/ dir can only
# ever serve one character: /rig/oski_walkfix.glb made it look for /rig/rig.json, which
# either does not exist (refusal) or belongs to somebody else (wrong stature, silently).
# Shipping <name>/rigged.glb + <name>/rig.json keeps the sibling convention true for as
# many characters as you like, with no code change.
CHAR=${CHAR:-oski}
GLBDIR=${GLBDIR:-/Users/molyneaux/3dmotions/out/oski_walkfix}
CARD=${CARD:-/Users/molyneaux/3dmotions/library/characters/oski/style.json}
[ -f "$GLBDIR/rigged.glb" ] || { echo "no rigged.glb in $GLBDIR"; exit 1; }
[ -f "$GLBDIR/rig.json" ]   || { echo "no rig.json in $GLBDIR -- place_rig now REFUSES without it"; exit 1; }

echo "== individual puts (FUSE fallback) =="
# SHIP THE WHOLE tools/heroshot DIR, not just place_rig.py. place_rig imports SIBLING
# modules -- frames (unconditional, line 612), lut_repair, and facade/object_harness
# conditionally -- and the original stager enumerated only place_rig + lut_repair. The
# missing `frames.py` killed all four chunks of oski_moonwalk_A10G_r1 with
# ModuleNotFoundError ~10 s in, after the full campus import. Enumerating imports by hand
# is a standing invitation to repeat that the next time a module is added; the directory
# is ~1 MB, so ship it whole.
$MODAL volume put --force berkeley-usd-take $BUSD/tools/heroshot /tools/heroshot >/dev/null
$MODAL volume put --force berkeley-usd-take $GLBDIR/rigged.glb /rig/$CHAR/rigged.glb >/dev/null
$MODAL volume put --force berkeley-usd-take $GLBDIR/rig.json   /rig/$CHAR/rig.json  >/dev/null
$MODAL volume put --force berkeley-usd-take $CARD /characters/$CHAR/style.json >/dev/null
for R in "$@"; do
  $MODAL volume put --force berkeley-usd-take "$R" /shots/$(basename $R) >/dev/null
done

echo "== rebuilding busd_take.tar (the one workers actually read) =="
ST=$(mktemp -d)
mkdir -p $ST/tools/heroshot $ST/tools/render $ST/rig/$CHAR $ST/shots $ST/characters/$CHAR
ln -s $BUSD/usd $ST/usd
ln -s $BUSD/textures $ST/textures
for _f in $BUSD/tools/heroshot/*.py; do ln -s $_f $ST/tools/heroshot/$(basename $_f); done
ln -s $BUSD/tools/render/lut_repair.py $ST/tools/render/lut_repair.py
ln -s $BUSD/tools/render/material_lut.json $ST/tools/render/material_lut.json
ln -s $GLBDIR/rigged.glb $ST/rig/$CHAR/rigged.glb
ln -s $GLBDIR/rig.json   $ST/rig/$CHAR/rig.json
ln -s $CARD $ST/characters/$CHAR/style.json
for R in "$@"; do ln -s "$R" $ST/shots/$(basename $R); done
# -h dereferences the symlinks; `characters` is the addition over the original
tar -chf $ST/busd_take.tar -C $ST usd textures tools rig shots characters
echo "  tar $(du -sm $ST/busd_take.tar | cut -f1) MB"
$MODAL volume put --force berkeley-usd-take $ST/busd_take.tar /busd_take.tar
rm -rf $ST
echo "staged."
