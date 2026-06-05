// cado-wu-server-rs -- an async work-unit server for CADO-NFS.
//
// Implements the api_server.py endpoints over the wudb SQLite schema
// (tables `workunits` + `files`; status 0=AVAILABLE 1=ASSIGNED 3=RECEIVED_OK
// 4=RECEIVED_ERROR), so it serves the stock Python clients and cado-nfs-client-rs:
//
//   GET  /            -> hello
//   GET  /workunit    -> assign an AVAILABLE work-unit (form body `clientid`);
//                        410 once serving is finished
//   GET  /file/<path> -> serve an input file from --filedir
//   GET  /files       -> list servable files
//   POST /upload      -> record a result (multipart)
//   POST /control     -> admin: finish | resume serving (optional --admin-token)
//
// Robustness: a connection pool (r2d2 + WAL); a background task that returns
// stale ASSIGNED rows to AVAILABLE after --wutimeout seconds (so a dead client's
// work is re-handed-out); a serving flag that makes /workunit answer 410 when the
// computation is finished; and optional TLS (--cert/--key) matching the Python
// self-signed-cert server.

use anyhow::{Context, Result};
use axum::{
    body::Bytes,
    extract::{Multipart, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use r2d2_sqlite::SqliteConnectionManager;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

type Pool = r2d2::Pool<SqliteConnectionManager>;

const AVAILABLE: i64 = 0;
const ASSIGNED: i64 = 1;
const RECEIVED_OK: i64 = 3;
const RECEIVED_ERROR: i64 = 4;

struct AppState {
    pool: Pool,
    filedir: PathBuf,
    uploaddir: PathBuf,
    serving: AtomicBool,
    admin_token: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cfg = parse_args()?;
    std::fs::create_dir_all(&cfg.uploaddir).ok();

    // pooled connections, WAL for concurrent readers + a single writer
    let manager = SqliteConnectionManager::file(&cfg.db).with_init(|c| {
        c.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")
    });
    let pool = r2d2::Pool::builder()
        .max_size(8)
        .build(manager)
        .context("building sqlite pool")?;
    {
        let conn = pool.get()?;
        ensure_schema(&conn)?;
    }

    let state = Arc::new(AppState {
        pool: pool.clone(),
        filedir: cfg.filedir,
        uploaddir: cfg.uploaddir,
        serving: AtomicBool::new(true),
        admin_token: cfg.admin_token,
    });

    // background: reclaim stale ASSIGNED work-units after --wutimeout
    if cfg.wutimeout > 0 {
        let pool2 = pool.clone();
        let wutimeout = cfg.wutimeout;
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(30.min(wutimeout.max(1))));
            loop {
                tick.tick().await;
                if let Ok(conn) = pool2.get() {
                    let cutoff = now_secs().saturating_sub(wutimeout);
                    match conn.execute(
                        "UPDATE workunits SET status=?1, assignedclient=NULL, timeassigned=NULL \
                         WHERE status=?2 AND CAST(timeassigned AS INTEGER) < ?3",
                        rusqlite::params![AVAILABLE, ASSIGNED, cutoff as i64],
                    ) {
                        Ok(n) if n > 0 => eprintln!("# reclaimed {n} stale work-unit(s) -> AVAILABLE"),
                        _ => {}
                    }
                }
            }
        });
    }

    let app = Router::new()
        .route("/", get(hello))
        .route("/workunit", get(get_workunit))
        .route("/files", get(list_files))
        .route("/file/*path", get(download_file))
        .route("/upload", post(upload))
        .route("/control", post(control))
        .with_state(state);

    let addr: std::net::SocketAddr = format!("{}:{}", cfg.addr, cfg.port).parse()?;
    let handle = axum_server::Handle::new();
    {
        // print the actually-bound URL (so --port 0 is usable)
        let handle = handle.clone();
        let scheme = if cfg.cert.is_some() { "https" } else { "http" };
        let db = cfg.db.clone();
        tokio::spawn(async move {
            if let Some(local) = handle.listening().await {
                println!("SERVER_URL {scheme}://{local}");
                eprintln!("# cado-wu-server-rs listening on {scheme}://{local}  db={db:?}");
            }
        });
    }

    match (cfg.cert, cfg.key) {
        (Some(cert), Some(key)) => {
            // rustls 0.23 needs a process-default crypto provider installed
            let _ = rustls::crypto::ring::default_provider().install_default();
            let tls = axum_server::tls_rustls::RustlsConfig::from_pem_file(&cert, &key)
                .await
                .with_context(|| format!("loading TLS cert/key {cert:?} {key:?}"))?;
            axum_server::bind_rustls(addr, tls)
                .handle(handle)
                .serve(app.into_make_service())
                .await?;
        }
        _ => {
            axum_server::bind(addr)
                .handle(handle)
                .serve(app.into_make_service())
                .await?;
        }
    }
    Ok(())
}

fn ensure_schema(conn: &rusqlite::Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS workunits (
            wurowid INTEGER PRIMARY KEY ASC UNIQUE NOT NULL,
            wuid VARCHAR(512) UNIQUE NOT NULL,
            submitter VARCHAR(512), status INTEGER NOT NULL, wu TEXT NOT NULL,
            timecreated TEXT, timeassigned TEXT, assignedclient TEXT,
            timeresult TEXT, resultclient TEXT,
            errorcode INTEGER, failedcommand INTEGER, timeverified TEXT,
            retryof INTEGER REFERENCES workunits(wurowid), priority INTEGER);
         CREATE TABLE IF NOT EXISTS files (
            filesrowid INTEGER PRIMARY KEY ASC UNIQUE NOT NULL,
            filename TEXT, path VARCHAR(512) UNIQUE NOT NULL,
            type TEXT, command INTEGER,
            wurowid INTEGER REFERENCES workunits(wurowid));",
    )
    .context("creating wudb schema")?;
    Ok(())
}

async fn hello() -> &'static str {
    "CADO-NFS work-unit server (Rust)\n"
}

async fn get_workunit(State(s): State<Arc<AppState>>, body: Bytes) -> Response {
    if !s.serving.load(Ordering::Relaxed) {
        return (StatusCode::GONE, "Distributed computation finished").into_response();
    }
    let form = parse_form(&body);
    let Some(clientid) = form.get("clientid").cloned() else {
        return (StatusCode::FORBIDDEN, "clientid must be provided").into_response();
    };
    let Ok(conn) = s.pool.get() else {
        return (StatusCode::INTERNAL_SERVER_ERROR, "db").into_response();
    };
    let row: Option<(i64, String)> = conn
        .query_row(
            "SELECT wurowid, wu FROM workunits WHERE status=?1 ORDER BY wurowid LIMIT 1",
            [AVAILABLE],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .ok();
    match row {
        None => (StatusCode::NOT_FOUND, "no work available").into_response(),
        Some((rowid, wu)) => {
            match conn.execute(
                "UPDATE workunits SET status=?1, assignedclient=?2, timeassigned=?3 \
                 WHERE wurowid=?4 AND status=?5",
                rusqlite::params![ASSIGNED, clientid, now_secs() as i64, rowid, AVAILABLE],
            ) {
                Ok(1) => {
                    eprintln!("# assigned wu {rowid} to {clientid}");
                    ([("content-type", "application/json")], wu).into_response()
                }
                _ => (StatusCode::NOT_FOUND, "no work available").into_response(),
            }
        }
    }
}

async fn download_file(State(s): State<Arc<AppState>>, Path(path): Path<String>) -> Response {
    if path.split('/').any(|c| c == "..") {
        return (StatusCode::BAD_REQUEST, "bad path").into_response();
    }
    match std::fs::read(s.filedir.join(&path)) {
        Ok(bytes) => ([("content-type", "application/octet-stream")], bytes).into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "file not registered").into_response(),
    }
}

async fn list_files(State(s): State<Arc<AppState>>) -> Response {
    let mut names = vec![];
    if let Ok(rd) = std::fs::read_dir(&s.filedir) {
        for e in rd.flatten() {
            if let Ok(n) = e.file_name().into_string() {
                names.push(n);
            }
        }
    }
    axum::Json(names).into_response()
}

// POST /control -- finish or resume serving work-units. Guarded by --admin-token
// when set (form field `token`). Lets a driver signal "computation done" -> 410.
async fn control(State(s): State<Arc<AppState>>, body: Bytes) -> Response {
    let form = parse_form(&body);
    if let Some(tok) = &s.admin_token {
        if form.get("token").map(|t| t != tok).unwrap_or(true) {
            return (StatusCode::FORBIDDEN, "bad token").into_response();
        }
    }
    match form.get("action").map(|a| a.as_str()) {
        Some("finish") => {
            s.serving.store(false, Ordering::Relaxed);
            (StatusCode::OK, "serving finished\n").into_response()
        }
        Some("resume") => {
            s.serving.store(true, Ordering::Relaxed);
            (StatusCode::OK, "serving resumed\n").into_response()
        }
        _ => (StatusCode::BAD_REQUEST, "action must be finish|resume").into_response(),
    }
}

async fn upload(State(s): State<Arc<AppState>>, mut mp: Multipart) -> Response {
    let mut fields: HashMap<String, String> = HashMap::new();
    let mut saved: Vec<(String, PathBuf)> = vec![];
    while let Some(field) = match mp.next_field().await {
        Ok(f) => f,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("multipart: {e}")).into_response(),
    } {
        let name = field.name().unwrap_or("").to_string();
        match field.file_name().map(|s| s.to_string()) {
            Some(fname) => {
                let data = match field.bytes().await {
                    Ok(b) => b,
                    Err(e) => return (StatusCode::BAD_REQUEST, format!("read: {e}")).into_response(),
                };
                let safe = sanitize(&fname);
                let dest = s.uploaddir.join(&safe);
                if dest.exists() {
                    return (StatusCode::FORBIDDEN, "File already exists").into_response();
                }
                if std::fs::write(&dest, &data).is_err() {
                    return (StatusCode::INTERNAL_SERVER_ERROR, "save").into_response();
                }
                saved.push((safe, dest));
            }
            None => {
                if let Ok(t) = field.text().await {
                    fields.insert(name, t);
                }
            }
        }
    }
    let (Some(clientid), Some(wuid)) = (fields.get("clientid"), fields.get("WUid")) else {
        return (StatusCode::BAD_REQUEST, "missing WUid and/or clientid").into_response();
    };
    let fileinfo: serde_json::Value =
        serde_json::from_str(fields.get("fileinfo").map(|s| s.as_str()).unwrap_or("{}"))
            .unwrap_or(serde_json::json!({}));
    let errorcode: Option<i64> = fields.get("errorcode").and_then(|v| v.parse().ok());
    record_result(&s, wuid, clientid, &saved, &fileinfo, errorcode)
}

fn record_result(
    s: &AppState,
    wuid: &str,
    clientid: &str,
    saved: &[(String, PathBuf)],
    fileinfo: &serde_json::Value,
    errorcode: Option<i64>,
) -> Response {
    let Ok(conn) = s.pool.get() else {
        return (StatusCode::INTERNAL_SERVER_ERROR, "db").into_response();
    };
    let row: Option<(i64, i64)> = conn
        .query_row(
            "SELECT wurowid, status FROM workunits WHERE wuid=?1",
            [wuid],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .ok();
    let Some((rowid, status)) = row else {
        return (StatusCode::NOT_FOUND, "unknown WUid").into_response();
    };
    if status != ASSIGNED {
        eprintln!("# warning: wu {wuid} not currently ASSIGNED (status {status})");
    }
    for (basename, destpath) in saved {
        let key = fileinfo
            .get(basename)
            .and_then(|v| v.get("key"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let _ = conn.execute(
            "INSERT INTO files (filename, path, type, wurowid) VALUES (?1,?2,?3,?4)",
            rusqlite::params![basename, destpath.to_string_lossy(), key, rowid],
        );
    }
    let new_status = if errorcode.unwrap_or(0) == 0 { RECEIVED_OK } else { RECEIVED_ERROR };
    let _ = conn.execute(
        "UPDATE workunits SET status=?1, resultclient=?2, errorcode=?3, timeresult=?4 \
         WHERE wurowid=?5",
        rusqlite::params![new_status, clientid, errorcode, now_secs() as i64, rowid],
    );
    eprintln!("# recorded result for wu {wuid} from {clientid}: status {new_status}, {} files", saved.len());
    (StatusCode::OK, "ok").into_response()
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() || "._-".contains(c) { c } else { '_' })
        .collect()
}

fn parse_form(body: &[u8]) -> HashMap<String, String> {
    let mut m = HashMap::new();
    let s = String::from_utf8_lossy(body);
    for pair in s.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            m.insert(urldecode(k), urldecode(v));
        }
    }
    m
}

fn urldecode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => { out.push(' '); i += 1; }
            b'%' if i + 2 < b.len() => match u8::from_str_radix(&s[i + 1..i + 3], 16) {
                Ok(byte) => { out.push(byte as char); i += 3; }
                Err(_) => { out.push('%'); i += 1; }
            },
            c => { out.push(c as char); i += 1; }
        }
    }
    out
}

struct Cfg {
    db: PathBuf,
    filedir: PathBuf,
    uploaddir: PathBuf,
    addr: String,
    port: u16,
    wutimeout: u64,
    cert: Option<PathBuf>,
    key: Option<PathBuf>,
    admin_token: Option<String>,
}

fn parse_args() -> Result<Cfg> {
    let mut c = Cfg {
        db: PathBuf::new(),
        filedir: PathBuf::from("."),
        uploaddir: std::env::temp_dir().join("cado-wu-uploads"),
        addr: "127.0.0.1".into(),
        port: 0,
        wutimeout: 3600,
        cert: None,
        key: None,
        admin_token: None,
    };
    let mut have_db = false;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--db" => { c.db = it.next().context("--db value")?.into(); have_db = true; }
            "--filedir" => c.filedir = it.next().context("--filedir")?.into(),
            "--uploaddir" => c.uploaddir = it.next().context("--uploaddir")?.into(),
            "--addr" => c.addr = it.next().context("--addr")?,
            "--port" => c.port = it.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            "--wutimeout" => c.wutimeout = it.next().and_then(|v| v.parse().ok()).unwrap_or(3600),
            "--cert" => c.cert = it.next().map(PathBuf::from),
            "--key" => c.key = it.next().map(PathBuf::from),
            "--admin-token" => c.admin_token = it.next(),
            "-h" | "--help" => {
                eprintln!(
                    "usage: cado-wu-server-rs --db <sqlite> [--filedir DIR] [--uploaddir DIR]\n\
                     [--addr 127.0.0.1] [--port N] [--wutimeout SECS] [--cert PEM --key PEM] [--admin-token T]"
                );
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown argument {other}"),
        }
    }
    anyhow::ensure!(have_db, "--db <sqlite path> is required");
    Ok(c)
}
