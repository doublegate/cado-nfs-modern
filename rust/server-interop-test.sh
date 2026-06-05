#!/usr/bin/env bash
# End-to-end test of the Rust work-unit server + Rust client over the wudb
# schema: seed a DB with one AVAILABLE work-unit, start cado-wu-server-rs, run
# cado-nfs-client-rs against it, and verify the DB row advanced
# AVAILABLE(0) -> RECEIVED_OK(3) with a recorded result file.
#
# Run from the repo root:  bash rust/server-interop-test.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
SERVER="rust/target/release/cado-wu-server-rs"
CLIENT="rust/target/release/cado-nfs-client-rs"
for b in "$SERVER" "$CLIENT"; do
  [ -x "$b" ] || { echo "build first: (cd rust && cargo build --release)"; exit 1; }
done
T="$(mktemp -d /tmp/cado-wu-srv.XXXXXX)"
mkdir -p "$T/filedir" "$T/uploads" "$T/dl" "$T/work"
DB="$T/wu.db"

# input file the work-unit downloads (checksum-verified by the client)
printf 'hello cado rust\n' > "$T/filedir/input.txt"
SHA1=$(sha1sum "$T/filedir/input.txt" | cut -d' ' -f1)

# seed the wudb schema + one AVAILABLE work-unit (status 0). The WU just cats the
# downloaded input file; its stdout (STDOUT0, upload:true) is the result.
python3 - "$DB" "$SHA1" <<'PY'
import sqlite3, json, sys
db, sha1 = sys.argv[1], sys.argv[2]
c = sqlite3.connect(db)
c.executescript("""
CREATE TABLE IF NOT EXISTS workunits(
  wurowid INTEGER PRIMARY KEY ASC UNIQUE NOT NULL, wuid VARCHAR(512) UNIQUE NOT NULL,
  submitter VARCHAR(512), status INTEGER NOT NULL, wu TEXT NOT NULL,
  timecreated TEXT, timeassigned TEXT, assignedclient TEXT, timeresult TEXT,
  resultclient TEXT, errorcode INTEGER, failedcommand INTEGER, timeverified TEXT,
  retryof INTEGER, priority INTEGER);
CREATE TABLE IF NOT EXISTS files(
  filesrowid INTEGER PRIMARY KEY ASC UNIQUE NOT NULL, filename TEXT,
  path VARCHAR(512) UNIQUE NOT NULL, type TEXT, command INTEGER, wurowid INTEGER);
""")
wu = {"id": "testwu1",
      "commands": ["/bin/cat ${FILE0}"],
      "files": {"FILE0": {"filename": "input.txt", "download": True,
                          "checksum": sha1, "algorithm": "sha1"},
                "STDOUT0": {"filename": "testwu1.out", "upload": True}}}
c.execute("INSERT INTO workunits(wuid,status,wu,timecreated) VALUES(?,?,?,?)",
          ("testwu1", 0, json.dumps(wu), "0"))
c.commit(); c.close()
print("seeded AVAILABLE work-unit testwu1")
PY

echo "## starting cado-wu-server-rs ..."
"$SERVER" --db "$DB" --filedir "$T/filedir" --uploaddir "$T/uploads" --port 0 >"$T/srv.log" 2>&1 &
SRV=$!; trap 'kill $SRV 2>/dev/null' EXIT
URL=""
for _ in $(seq 1 30); do
  URL=$(grep -oE 'http://[0-9.]+:[0-9]+' "$T/srv.log" | head -1)
  [ -n "$URL" ] && break
  kill -0 $SRV 2>/dev/null || { echo "server died"; cat "$T/srv.log"; exit 1; }
  sleep 0.3
done
echo "## server URL: ${URL:-NONE}"; [ -z "$URL" ] && { cat "$T/srv.log"; exit 1; }

echo "## running Rust client (--single) ..."
"$CLIENT" --server "$URL" --single --downloadretry 1 \
    --dldir "$T/dl" --workdir "$T/work" --clientid srvtest 2>&1 | sed 's/^/[client] /'

echo "## verifying DB state ..."
python3 - "$DB" "$T/uploads" <<'PY'
import sqlite3, sys, os, glob
db, up = sys.argv[1], sys.argv[2]
c = sqlite3.connect(db)
st = c.execute("SELECT status, resultclient FROM workunits WHERE wuid='testwu1'").fetchone()
nf = c.execute("SELECT COUNT(*), MIN(type) FROM files WHERE wurowid=1").fetchone()
print(f"workunit status={st[0]} (3=RECEIVED_OK) resultclient={st[1]}")
print(f"result files recorded={nf[0]} type={nf[1]}")
ups = glob.glob(os.path.join(up, '*'))
content = open(ups[0]).read().strip() if ups else '<none>'
print(f"uploaded result content={content!r}")
ok = st[0] == 3 and nf[0] >= 1 and content == 'hello cado rust'
print("## PASS" if ok else "## FAIL")
sys.exit(0 if ok else 1)
PY
echo "## done (rc=$?)"
