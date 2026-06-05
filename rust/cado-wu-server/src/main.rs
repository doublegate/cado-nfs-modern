// cado-wu-server-rs -- an async work-unit server for CADO-NFS.
//
// It implements the same HTTP endpoints as the stock Python server
// (scripts/cadofactor/api_server.py) over the same `wudb` SQLite schema
// (scripts/cadofactor/wudb.py: tables `workunits` and `files`), so it can serve
// the unmodified Python clients (and cado-nfs-client-rs):
//
//   GET  /            -> hello
//   GET  /workunit    -> assign an AVAILABLE work-unit (form body `clientid`)
//   GET  /file/<path> -> serve an input file from the file directory
//   GET  /files       -> list servable files
//   POST /upload      -> record a result (multipart: clientid, WUid, fileinfo, files)
//
// Work-unit lifecycle, matching wudb.WuAccess: a row's `status` goes
// AVAILABLE(0) -> ASSIGNED(1) on assign, -> RECEIVED_OK(3)/RECEIVED_ERROR(4) on
// upload; result files are inserted into the `files` table linked by wurowid.
// The Python driver (cadotask.py) populates AVAILABLE rows and reads results
// from the same DB, so the two interoperate.

use anyhow::{Context, Result};
use axum::{
    body::Bytes,
    extract::{Multipart, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use rusqlite::Connection;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

// wudb status codes (wudb.py STATUS_NAMES order)
const AVAILABLE: i64 = 0;
const ASSIGNED: i64 = 1;
const RECEIVED_OK: i64 = 3;
const RECEIVED_ERROR: i64 = 4;

struct AppState {
    db: Mutex<Connection>,
    filedir: PathBuf,
    uploaddir: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cfg = parse_args()?;
    std::fs::create_dir_all(&cfg.uploaddir).ok();

    let conn = Connection::open(&cfg.db).with_context(|| format!("open db {:?}", cfg.db))?;
    ensure_schema(&conn)?;
    let state = Arc::new(AppState {
        db: Mutex::new(conn),
        filedir: cfg.filedir,
        uploaddir: cfg.uploaddir,
    });

    let app = Router::new()
        .route("/", get(hello))
        .route("/workunit", get(get_workunit))
        .route("/files", get(list_files))
        .route("/file/*path", get(download_file))
        .route("/upload", post(upload))
        .with_state(state);

    let bindaddr = format!("{}:{}", cfg.addr, cfg.port);
    let listener = tokio::net::TcpListener::bind(&bindaddr)
        .await
        .with_context(|| format!("binding {bindaddr}"))?;
    let local = listener.local_addr().context("local_addr")?;
    // single, parseable line so a launcher can scrape the URL
    println!("SERVER_URL http://{local}");
    eprintln!("# cado-wu-server-rs listening on http://{local}  db={:?}", cfg.db);
    axum::serve(listener, app).await.context("serving")?;
    Ok(())
}

// The wudb schema (CREATE IF NOT EXISTS so we can both create a fresh DB and
// attach to one the Python driver already made).
fn ensure_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS workunits (
            wurowid INTEGER PRIMARY KEY ASC UNIQUE NOT NULL,
            wuid VARCHAR(512) UNIQUE NOT NULL,
            submitter VARCHAR(512),
            status INTEGER NOT NULL,
            wu TEXT NOT NULL,
            timecreated TEXT, timeassigned TEXT, assignedclient TEXT,
            timeresult TEXT, resultclient TEXT,
            errorcode INTEGER, failedcommand INTEGER,
            timeverified TEXT,
            retryof INTEGER REFERENCES workunits(wurowid),
            priority INTEGER);
         CREATE TABLE IF NOT EXISTS files (
            filesrowid INTEGER PRIMARY KEY ASC UNIQUE NOT NULL,
            filename TEXT,
            path VARCHAR(512) UNIQUE NOT NULL,
            type TEXT,
            command INTEGER,
            wurowid INTEGER REFERENCES workunits(wurowid));",
    )
    .context("creating wudb schema")?;
    Ok(())
}

async fn hello() -> &'static str {
    "CADO-NFS work-unit server (Rust)\n"
}

// GET /workunit -- the Python client sends `clientid` in a form-urlencoded body
// (even on GET). Assign one AVAILABLE work-unit to it.
async fn get_workunit(State(s): State<Arc<AppState>>, body: Bytes) -> Response {
    let form = parse_form(&body);
    let clientid = match form.get("clientid") {
        Some(c) => c.clone(),
        None => return (StatusCode::FORBIDDEN, "clientid must be provided").into_response(),
    };

    let db = s.db.lock().unwrap();
    // SELECT the first AVAILABLE wu, then mark it ASSIGNED (single-writer under
    // the mutex, so this is atomic enough for the assignment).
    let row: Option<(i64, String)> = db
        .query_row(
            "SELECT wurowid, wu FROM workunits WHERE status = ?1 ORDER BY wurowid LIMIT 1",
            [AVAILABLE],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .ok();

    match row {
        None => (StatusCode::NOT_FOUND, "no work available").into_response(),
        Some((rowid, wu)) => {
            let r = db.execute(
                "UPDATE workunits SET status=?1, assignedclient=?2, timeassigned=?3 \
                 WHERE wurowid=?4 AND status=?5",
                rusqlite::params![ASSIGNED, clientid, now(), rowid, AVAILABLE],
            );
            match r {
                Ok(1) => {
                    eprintln!("# assigned wu {rowid} to {clientid}");
                    ([("content-type", "application/json")], wu).into_response()
                }
                // lost a race (another client grabbed it): tell the client to retry
                _ => (StatusCode::NOT_FOUND, "no work available").into_response(),
            }
        }
    }
}

// GET /file/<path> -- serve an input file from the configured file directory.
async fn download_file(State(s): State<Arc<AppState>>, Path(path): Path<String>) -> Response {
    // prevent path escapes
    if path.split('/').any(|c| c == "..") {
        return (StatusCode::BAD_REQUEST, "bad path").into_response();
    }
    let full = s.filedir.join(&path);
    match std::fs::read(&full) {
        Ok(bytes) => (
            [("content-type", "application/octet-stream")],
            bytes,
        )
            .into_response(),
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

// POST /upload -- multipart: text fields clientid, WUid, [errorcode],
// [failedcommand], fileinfo (JSON {basename:{WUid,key}}); plus the result files.
async fn upload(State(s): State<Arc<AppState>>, mut mp: Multipart) -> Response {
    let mut fields: HashMap<String, String> = HashMap::new();
    let mut saved: Vec<(String, PathBuf)> = vec![]; // (basename, saved path)

    while let Some(field) = match mp.next_field().await {
        Ok(f) => f,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("multipart error: {e}")).into_response(),
    } {
        let name = field.name().unwrap_or("").to_string();
        match field.file_name().map(|s| s.to_string()) {
            Some(fname) => {
                // a file part
                let data = match field.bytes().await {
                    Ok(b) => b,
                    Err(e) => {
                        return (StatusCode::BAD_REQUEST, format!("read file: {e}")).into_response()
                    }
                };
                let safe = sanitize(&fname);
                let dest = s.uploaddir.join(&safe);
                if dest.exists() {
                    return (StatusCode::FORBIDDEN, "File already exists").into_response();
                }
                if std::fs::write(&dest, &data).is_err() {
                    return (StatusCode::INTERNAL_SERVER_ERROR, "save failed").into_response();
                }
                saved.push((safe, dest));
            }
            None => {
                // a text field
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
    let db = s.db.lock().unwrap();
    let row: Option<(i64, i64)> = db
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
        // matches the Python server: warn but accept (avoids losing a result that
        // raced a timeout-cancel). We still record the files.
        eprintln!("# warning: wu {wuid} not currently ASSIGNED (status {status})");
    }

    // insert result files (filename, path, type=key from fileinfo, wurowid)
    for (basename, destpath) in saved {
        let key = fileinfo
            .get(basename)
            .and_then(|v| v.get("key"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let _ = db.execute(
            "INSERT INTO files (filename, path, type, wurowid) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![basename, destpath.to_string_lossy(), key, rowid],
        );
    }

    let new_status = if errorcode.unwrap_or(0) == 0 { RECEIVED_OK } else { RECEIVED_ERROR };
    let _ = db.execute(
        "UPDATE workunits SET status=?1, resultclient=?2, errorcode=?3, timeresult=?4 \
         WHERE wurowid=?5",
        rusqlite::params![new_status, clientid, errorcode, now(), rowid],
    );
    eprintln!(
        "# recorded result for wu {wuid} from {clientid}: status {new_status}, {} files",
        saved.len()
    );
    (StatusCode::OK, "ok").into_response()
}

// --- helpers ---

fn now() -> String {
    let d = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    // wudb stores str(datetime.utcnow()); the happy path doesn't reparse it, so a
    // stable epoch-seconds string is sufficient and unambiguous.
    format!("{}", d.as_secs())
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
            b'+' => {
                out.push(' ');
                i += 1;
            }
            b'%' if i + 2 < b.len() => {
                if let Ok(byte) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                    out.push(byte as char);
                    i += 3;
                } else {
                    out.push('%');
                    i += 1;
                }
            }
            c => {
                out.push(c as char);
                i += 1;
            }
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
}

fn parse_args() -> Result<Cfg> {
    let mut db = None;
    let mut filedir = None;
    let mut uploaddir = None;
    let mut addr = "127.0.0.1".to_string();
    let mut port = 0u16; // 0 = let OS pick (printed at startup)
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--db" => db = it.next().map(PathBuf::from),
            "--filedir" => filedir = it.next().map(PathBuf::from),
            "--uploaddir" => uploaddir = it.next().map(PathBuf::from),
            "--addr" => addr = it.next().unwrap_or(addr),
            "--port" => port = it.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            "-h" | "--help" => {
                eprintln!(
                    "usage: cado-wu-server-rs --db <sqlite> [--filedir DIR] [--uploaddir DIR] \
                     [--addr 127.0.0.1] [--port N]"
                );
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown argument {other}"),
        }
    }
    Ok(Cfg {
        db: db.context("--db <sqlite path> is required")?,
        filedir: filedir.unwrap_or_else(|| PathBuf::from(".")),
        uploaddir: uploaddir.unwrap_or_else(|| std::env::temp_dir().join("cado-wu-uploads")),
        addr,
        port,
    })
}
