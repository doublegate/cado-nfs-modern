#!/usr/bin/env bash
#
# cluster-launch.sh -- fan out cado-nfs-client-rs across a cluster (Track 3.4).
#
# CADO-NFS distributes sieving by running many work-unit clients against one
# server. This helper starts the static Rust client (cado-nfs-client-rs) on a
# list of hosts (over SSH) or through Slurm (srun/sbatch), all pointed at the
# same server URL with the same TLS pinning -- so you do not hand-start clients
# one machine at a time.
#
# It assumes, like CADO's own distributed mode, SSH public-key auth to the hosts
# and that the client binary is reachable on each host (a shared filesystem, or
# pre-deployed to the same path -- see --client-bin). The server must already be
# running (cado-nfs.py prints "server.address=... server.port=..."; or run the
# Rust server directly). Cert pinning: pass the server's --certsha1 (cado-nfs.py
# prints the certificate SHA1) so clients trust it without a CA file.
#
# Examples:
#   # 4 clients each on two hosts, over SSH, with cert pinning:
#   scripts/cluster-launch.sh --server https://head:4242 --certsha1 AB12.. \
#       --hosts node01,node02 --clients-per-host 4
#
#   # one client per Slurm task across an allocation:
#   scripts/cluster-launch.sh --server https://head:4242 --certsha1 AB12.. \
#       --slurm --ntasks 16
#
#   # stop everything started over SSH:
#   scripts/cluster-launch.sh --hosts node01,node02 --stop
#
set -euo pipefail

PROG=$(basename "$0")
SERVER=""
CERTSHA1=""
HOSTS=""
HOSTFILE=""
SLURM=0
NTASKS=0
CLIENTS_PER_HOST=1
CLIENT_BIN=""
ARCH=""
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new"
STOP=0
DRYRUN=0
EXTRA=()

usage() {
    sed -n '3,33p' "$0" | sed 's/^# \{0,1\}//'
    cat <<EOF

Options:
  --server URL          work-unit server URL (required unless --stop)
  --certsha1 HEX        pin the server cert by SHA1 (recommended over TLS)
  --hosts h1,h2,...     comma-separated SSH hosts
  --hostfile FILE       file with one host per line (# comments allowed)
  --clients-per-host N  SSH clients to start per host (default 1)
  --slurm               launch via Slurm (srun) instead of SSH
  --ntasks N            Slurm: number of client tasks (srun -n N)
  --client-bin PATH     client binary path valid on the hosts
                        (default: this checkout's build .../cado-nfs-client-rs
                         or rust/target/release/cado-nfs-client-rs)
  --arch S              pass --arch S to the client
  --ssh-opts "..."      override ssh options
  --stop                kill cado-nfs-client-rs on the SSH hosts and exit
  --dry-run             print the commands without running them
  -h, --help            this help
  Anything after -- is passed verbatim to each client.
EOF
}

die() { echo "$PROG: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --server) SERVER="$2"; shift 2;;
        --certsha1) CERTSHA1="$2"; shift 2;;
        --hosts) HOSTS="$2"; shift 2;;
        --hostfile) HOSTFILE="$2"; shift 2;;
        --clients-per-host) CLIENTS_PER_HOST="$2"; shift 2;;
        --slurm) SLURM=1; shift;;
        --ntasks) NTASKS="$2"; shift 2;;
        --client-bin) CLIENT_BIN="$2"; shift 2;;
        --arch) ARCH="$2"; shift 2;;
        --ssh-opts) SSH_OPTS="$2"; shift 2;;
        --stop) STOP=1; shift;;
        --dry-run) DRYRUN=1; shift;;
        -h|--help) usage; exit 0;;
        --) shift; EXTRA=("$@"); break;;
        *) die "unknown argument $1 (try --help)";;
    esac
done

# Resolve the host list (from --hosts and/or --hostfile).
declare -a HOSTLIST=()
if [ -n "$HOSTS" ]; then
    IFS=',' read -r -a _h <<< "$HOSTS"
    HOSTLIST+=("${_h[@]}")
fi
if [ -n "$HOSTFILE" ]; then
    [ -r "$HOSTFILE" ] || die "cannot read hostfile $HOSTFILE"
    while IFS= read -r line; do
        line="${line%%#*}"; line="$(echo "$line" | tr -d '[:space:]')"
        [ -n "$line" ] && HOSTLIST+=("$line")
    done < "$HOSTFILE"
fi

run() {  # echo + (maybe) execute
    echo "+ $*"
    [ "$DRYRUN" -eq 1 ] || eval "$@"
}

# --stop: kill the client on each SSH host.
if [ "$STOP" -eq 1 ]; then
    [ ${#HOSTLIST[@]} -gt 0 ] || die "--stop needs --hosts/--hostfile"
    for h in "${HOSTLIST[@]}"; do
        run "ssh $SSH_OPTS $h 'pkill -f cado-nfs-client-rs || true'"
    done
    echo "$PROG: stop signal sent to ${#HOSTLIST[@]} host(s)."
    exit 0
fi

[ -n "$SERVER" ] || die "--server URL is required (try --help)"

# Default client binary: prefer this checkout's hostname build dir, else the
# Rust workspace release dir.
if [ -z "$CLIENT_BIN" ]; then
    here=$(cd "$(dirname "$0")/.." && pwd)
    for cand in \
        "$here/build/$(hostname)/rust/cado-nfs-client-rs" \
        "$here/rust/target/release/cado-nfs-client-rs"; do
        [ -x "$cand" ] && CLIENT_BIN="$cand" && break
    done
    [ -n "$CLIENT_BIN" ] || die "could not find cado-nfs-client-rs; pass --client-bin"
fi

# Assemble the common client argument string.
cargs="--server $SERVER"
[ -n "$CERTSHA1" ] && cargs="$cargs --certsha1 $CERTSHA1"
[ -n "$ARCH" ] && cargs="$cargs --arch $ARCH"
if [ ${#EXTRA[@]} -gt 0 ]; then
    cargs="$cargs ${EXTRA[*]}"
fi

echo "$PROG: server=$SERVER  client=$CLIENT_BIN"

if [ "$SLURM" -eq 1 ]; then
    [ "$DRYRUN" -eq 1 ] || command -v srun >/dev/null || die "--slurm given but srun not found"
    [ "$NTASKS" -gt 0 ] || die "--slurm needs --ntasks N"
    # one client per task; srun fans across the allocation's nodes.
    run "srun --ntasks=$NTASKS --kill-on-bad-exit=0 $CLIENT_BIN $cargs"
    exit 0
fi

[ ${#HOSTLIST[@]} -gt 0 ] || die "no hosts (use --hosts/--hostfile or --slurm)"
total=0
for h in "${HOSTLIST[@]}"; do
    for i in $(seq 1 "$CLIENTS_PER_HOST"); do
        # nohup + background so the SSH session returns immediately; clientid is
        # made unique per (host, slot) so the server distinguishes them.
        cid="${h}.${i}"
        remote="nohup $CLIENT_BIN $cargs --clientid $cid >/tmp/cado-client-$cid.log 2>&1 &"
        run "ssh $SSH_OPTS $h '$remote'"
        total=$((total + 1))
    done
done
echo "$PROG: launched $total client(s) across ${#HOSTLIST[@]} host(s)."
echo "  logs: /tmp/cado-client-<host>.<slot>.log on each host"
echo "  stop: $PROG --hosts <same> --stop"
