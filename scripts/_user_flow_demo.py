"""
Exercise the user-management lifecycle against a running API and render the
request/response transcript to an HTML file (for screenshotting).

Hits only the auth + history endpoints (no RAG stack required):
  register -> login -> /auth/me -> /me/queries

Run:  python scripts/_user_flow_demo.py [base_url] [out_html]
"""

from __future__ import annotations

import html
import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8077"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/screenshots/_user_flow.html"

DEMO_EMAIL = "admin.demo@example.com"
DEMO_PW = "S3curePass!"


def call(method: str, path: str, *, body=None, token=None):
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            status, payload = r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        status, payload = e.code, e.read().decode()
    try:
        payload = json.dumps(json.loads(payload), indent=2)
    except Exception:
        pass
    return status, payload


steps = []


def record(title, method, path, status, payload, body=None):
    steps.append({
        "title": title, "method": method, "path": path,
        "status": status, "payload": payload,
        "body": json.dumps(body, indent=2) if body else None,
    })


# 1. Register (idempotent for re-runs: tolerate 400 already-registered)
reg_body = {"email": DEMO_EMAIL, "password": DEMO_PW, "full_name": "Demo Admin"}
s, p = call("POST", "/auth/register", body=reg_body)
record("1. Crear usuario", "POST", "/auth/register", s, p, reg_body)

# 2. Login
login_body = {"email": DEMO_EMAIL, "password": DEMO_PW}
s, p = call("POST", "/auth/login", body=login_body)
record("2. Login (obtener JWT)", "POST", "/auth/login", s, p, login_body)
token = None
try:
    token = json.loads(p)["access_token"]
except Exception:
    pass

# 3. /auth/me (protected profile)
s, p = call("GET", "/auth/me", token=token)
record("3. Perfil del usuario autenticado", "GET", "/auth/me", s, p)

# 4. /me/queries (per-user history)
s, p = call("GET", "/me/queries", token=token)
record("4. Historial de consultas del usuario", "GET", "/me/queries", s, p)


def badge(status):
    color = "#16a34a" if 200 <= status < 300 else "#dc2626"
    return f'<span style="background:{color};color:#fff;padding:2px 10px;border-radius:6px;font-weight:700">{status}</span>'


def meth(m):
    colors = {"POST": "#49cc90", "GET": "#61affe"}
    return f'<span style="background:{colors.get(m,"#777")};color:#fff;padding:3px 12px;border-radius:5px;font-weight:700;font-size:13px">{m}</span>'


rows = []
for st in steps:
    req_block = (
        f'<div class="lbl">Request body</div><pre class="req">{html.escape(st["body"])}</pre>'
        if st["body"] else ""
    )
    rows.append(f"""
    <div class="card">
      <div class="hd">{meth(st['method'])} <code>{html.escape(st['path'])}</code>
        <span class="ttl">{html.escape(st['title'])}</span>
        <span class="st">{badge(st['status'])}</span></div>
      {req_block}
      <div class="lbl">Response</div>
      <pre class="res">{html.escape(st['payload'])}</pre>
    </div>""")

doc = f"""<!doctype html><html><head><meta charset="utf-8"><style>
body{{font-family:Segoe UI,Arial,sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:32px}}
h1{{font-size:24px;margin:0 0 4px}} .sub{{color:#94a3b8;margin-bottom:24px;font-size:14px}}
.card{{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:18px 20px;margin-bottom:18px}}
.hd{{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:12px}}
.hd code{{font-size:15px;color:#e2e8f0}} .ttl{{color:#cbd5e1;font-size:14px}}
.st{{margin-left:auto}}
.lbl{{text-transform:uppercase;font-size:11px;letter-spacing:.06em;color:#64748b;margin:8px 0 4px}}
pre{{background:#0b1120;border-radius:8px;padding:12px 14px;overflow:auto;font-size:13px;line-height:1.45;
     font-family:Consolas,Monaco,monospace;margin:0;color:#a5f3fc}}
pre.req{{color:#fde68a}}
</style></head><body>
<h1>Administración de usuarios — flujo en vivo</h1>
<div class="sub">Knowledge Assistant API · endpoints de usuario ejercitados contra el servidor en vivo ({html.escape(BASE)})</div>
{''.join(rows)}
</body></html>"""

import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(doc)
print("WROTE", OUT)
for st in steps:
    print(f"{st['method']:4} {st['path']:18} -> {st['status']}")
