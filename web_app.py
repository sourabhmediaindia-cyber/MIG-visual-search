#!/usr/bin/env python3
"""Run a local browser interface for searching an existing CLIP image index."""

import argparse
import hmac
import io
import html
import json
import mimetypes
import os
import secrets
import threading
import time
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from email import policy
from email.parser import BytesParser
from pathlib import Path
from urllib.parse import parse_qs, quote, urlencode, urlsplit

import numpy as np
import torch
from PIL import Image

from index import ImageDS, collate, path_terms, save, scan
import open_clip
from search import build_text_embedding, lexical_match_score

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
SESSION_TTL_SECONDS = 12 * 60 * 60


def render_login_page(error=""):
  message = f'<p class="error" role="alert">{html.escape(error)}</p>' if error else ""
  return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sign in - MIG Visual Search</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      min-height: 100vh;
      margin: 0;
      display: grid;
      place-items: center;
      padding: 24px;
      color: #20251f;
      font-family: Inter, "Segoe UI", sans-serif;
      background: #f4f5f2;
    }}
    .login-card {{
      width: min(100%, 420px);
      padding: 32px;
      border: 1px solid #e3e6df;
      border-radius: 16px;
      background: #fff;
      box-shadow: 0 12px 40px rgba(30, 45, 30, .08);
    }}
    h1 {{ margin: 0; font-size: 1.6rem; }}
    .intro {{ margin: 8px 0 24px; color: #727a70; }}
    label {{ display: block; margin: 16px 0 6px; font-weight: 600; }}
    input {{
      width: 100%;
      padding: 12px;
      border: 1px solid #d9ded5;
      border-radius: 9px;
      font: inherit;
    }}
    input:focus {{ border-color: #477a55; outline: 2px solid rgba(71, 122, 85, .16); }}
    button {{
      width: 100%;
      margin-top: 22px;
      padding: 12px;
      border: 0;
      border-radius: 9px;
      color: #fff;
      background: #477a55;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }}
    button:hover {{ background: #315c3d; }}
    .error {{ color: #a12622; }}
  </style>
</head>
<body>
  <main class="login-card">
    <h1>MIG Visual Search</h1>
    <p class="intro">Sign in to continue.</p>
    {message}
    <form action="/login" method="post">
      <label for="username">Username</label>
      <input id="username" name="username" autocomplete="username" required autofocus>
      <label for="password">Password</label>
      <input id="password" name="password" type="password" autocomplete="current-password" required>
      <button type="submit">Sign in</button>
    </form>
  </main>
</body>
</html>"""


def render_page(
    query="",
    results=(),
    error="",
    folders=(),
    folder="",
    page=1,
    page_count=1,
    result_total=0,
    image_query="",
    page_size=20,
    scan_folder="",
    broad_matches=False,
):
    cards = []
    start_rank = (page - 1) * page_size + 1
    for rank, (image_id, path, score) in enumerate(results, start_rank):
        relative_path = f"{path.parent.name}/{path.name}"
        escaped_path = html.escape(relative_path, quote=True)
        cards.append(
            '<article class="card">'
            f'<a href="/image/{image_id}" target="_blank" rel="noopener">'
            f'<div class="thumb-wrap"><img src="/image/{image_id}" '
            f'alt="{escaped_path}" loading="lazy"></div></a>'
            '<div class="card-body"><div class="meta-row">'
            f'<span class="rank">#{rank}</span>'
            f'<span class="score">{score:.3f}</span></div>'
            f'<div class="path">{escaped_path}</div>'
            f'<a class="download-link" href="/download/{image_id}">Download image</a>'
            '</div></article>'
        )
    folder_options = ['<option value="">All folders</option>']
    for folder_path in folders:
        value = str(folder_path)
        selected = " selected" if value == folder else ""
        folder_options.append(
            f'<option value="{html.escape(value, quote=True)}"{selected}>'
            f'{html.escape(value)}</option>'
        )

    message = (
        f'<p class="message">{html.escape(error)}</p>'
        if error
        else (
            '<p class="message">No images matched. Try another query or folder.</p>'
            if (query or image_query or folder) and not cards
            else ""
        )
    )
    heading = (
        f'<p class="results-meta">{result_total} results'
        f'{f" for <strong>{html.escape(query)}</strong>" if query else ""}'
        f'{" visually similar to your image" if image_query else ""}'
        f' <span>Page {page} of {page_count}</span></p>'
        if result_total
        else '<p class="results-meta empty-results-meta">Search indexed images by text or example image</p>'
    )

    def page_link(target_page, label, disabled=False):
        params = {"page": target_page}
        if query:
            params["q"] = query
        if folder:
            params["folder"] = folder
        if image_query:
            params["image_query"] = image_query
        if broad_matches:
            params["broaden"] = "1"
        if disabled:
            return f'<span class="page-link disabled">{label}</span>'
        return f'<a class="page-link" href="?{html.escape(urlencode(params), quote=True)}">{label}</a>'

    pagination = ""
    if page_count > 1:
        pagination = (
            '<nav class="pagination" aria-label="Search result pages">'
            + page_link(page - 1, "Previous", page <= 1)
            + f'<span>Page {page} of {page_count}</span>'
            + page_link(page + 1, "Next", page >= page_count)
            + '</nav>'
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MIG Visual Search</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f5f2;
      --panel: #fff;
      --border: #e3e6df;
      --text: #20251f;
      --muted: #727a70;
      --accent: #477a55;
      --accent-dark: #315c3d;
      --surface: #fff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      min-height: 100vh;
      margin: 0;
      padding: 0 28px 64px;
      color: var(--text);
      font-family: Inter, "Segoe UI", sans-serif;
      background:
        linear-gradient(rgba(17, 27, 38, .28), rgba(17, 27, 38, .42)),
        url("/visual-search-background.png") center / cover fixed no-repeat;
    }}
    .page {{ max-width: 1380px; margin: 0 auto; }}
    .hero {{
      max-width: 960px;
      margin: 0 auto 38px;
      padding: 22px 30px;
      border-bottom: 1px solid var(--border);
      background: rgba(255, 255, 255, .94);
      backdrop-filter: blur(8px);
      text-align: center;
    }}
    .hero-header {{ display: flex; align-items: center; justify-content: center; gap: 12px; max-width: 1380px; margin: 0 auto 30px; }}
    .brand {{ display: flex; align-items: center; gap: 10px; color: var(--text); font-size: 1.31rem; font-weight: 750; letter-spacing: .12em; }}
    .brand-mark {{ width: 9px; height: 9px; border-radius: 50%; background: var(--accent); }}
    .brand-logo {{ width: 75px; height: auto; }}
    .logout-row {{ display: flex; justify-content: flex-end; margin: -18px auto 14px; }}
    .logout-button {{ padding: 7px 11px; border: 1px solid var(--border); border-radius: 8px; color: var(--text); background: #fff; font: inherit; cursor: pointer; }}
    .eyebrow {{ color: var(--muted); font-size: .82rem; }}
    h1 {{ max-width: 1380px; margin: 0 auto; font-size: clamp(1.8rem, 3vw, 2.5rem); letter-spacing: -.045em; line-height: 1.12; }}
    .intro {{ max-width: 1380px; margin: 10px auto 22px; color: var(--muted); line-height: 1.55; }}
    .search-form {{ display: flex; flex-wrap: wrap; gap: 10px; max-width: 900px; margin: 0 auto; }}
    .search-form .search-input {{ min-height: 50px; }}
    .search-input {{
      flex: 1;
      min-width: 0;
      padding: 13px 16px;
      border: 1px solid #d9ded5;
      border-radius: 10px;
      outline: none;
      color: var(--text);
      background: var(--surface);
      font: inherit;
      transition: border-color .18s, box-shadow .18s;
    }}
    .search-input:focus, .folder-input:focus, .folder-select:focus, .image-input:focus {{ border-color: var(--accent); box-shadow: 0 0 0 3px rgba(71, 122, 85, .13); outline: none; }}
    .folder-select, .image-input, .folder-input {{
      min-width: 0;
      padding: 11px 13px;
      border: 1px solid #d9ded5;
      border-radius: 10px;
      color: var(--text);
      background: var(--surface);
      font: inherit;
    }}
    .folder-select {{ flex: 1; }}
    .search-button {{
      padding: 12px 22px;
      border: 0;
      border-radius: 10px;
      color: #fff;
      background: var(--accent);
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      transition: filter .18s, transform .18s;
    }}
    .search-button:hover {{ background: var(--accent-dark); }}
    .image-button {{
      padding: 10px 14px;
      border: 1px solid #d9ded5;
      border-radius: 10px;
      color: var(--text);
      background: #fff;
      font: inherit;
      cursor: pointer;
      transition: background .18s, border-color .18s;
    }}
    .image-button:hover {{ background: #f1f5ef; border-color: var(--accent); }}
    .search-options {{ display: flex; flex-basis: 100%; align-items: center; justify-content: space-between; gap: 14px; margin-top: 3px; }}
    .broad-search {{ display: flex; align-items: center; gap: 8px; color: var(--muted); font-size: .88rem; cursor: pointer; }}
    .broad-search input {{ width: 16px; height: 16px; accent-color: var(--accent); }}
    .image-search {{ max-width: 900px; margin: 18px auto 0; padding-top: 16px; border-top: 1px solid var(--border); }}
    .image-search-title {{ margin: 0 0 10px; font-size: .87rem; font-weight: 650; }}
    .image-form {{ display: flex; align-items: center; gap: 10px; }}
    .image-input {{ flex: 1; padding: 10px; }}
    .image-preview {{ display: flex; align-items: center; gap: 10px; min-width: 0; margin-top: 10px; color: var(--muted); font-size: .82rem; }}
    .image-preview[hidden] {{ display: none; }}
    .image-preview img {{ width: 52px; height: 52px; border-radius: 9px; object-fit: cover; }}
    .index-tools {{ max-width: 900px; margin: 16px auto 0; padding-top: 14px; border-top: 1px solid var(--border); color: var(--muted); }}
    .index-tools summary {{ width: fit-content; cursor: pointer; font-size: .87rem; }}
    .index-form {{ display: flex; gap: 10px; margin-top: 14px; }}
    .folder-input {{ flex: 1; }}
    .index-status {{ min-height: 1.4em; margin: 10px 0 0; color: var(--muted); }}
    .results-meta {{ display: flex; align-items: baseline; gap: 8px; margin: 0 0 20px; color: var(--muted); }}
    .empty-results-meta {{ justify-content: center; }}
    .results-meta strong {{ color: var(--text); }}
    .results-meta span {{ margin-left: auto; font-size: .86rem; }}
    .filters {{ display: flex; justify-content: center; gap: 10px; max-width: 1380px; margin: 0 auto 22px; }}
    .filters .folder-select {{ max-width: 340px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(245px, 1fr));
      gap: 18px;
    }}
    .card {{
      overflow: hidden;
      border: 1px solid var(--border);
      border-radius: 13px;
      background: var(--panel);
      box-shadow: 0 3px 12px rgba(30, 45, 30, .045);
      transition: transform .18s, border-color .18s, box-shadow .18s;
    }}
    .card:hover {{ transform: translateY(-3px); border-color: #c7d7c6; box-shadow: 0 10px 24px rgba(30, 45, 30, .1); }}
    .thumb-wrap {{ aspect-ratio: 4 / 3; background: #ecefe9; }}
    .thumb-wrap img {{ width: 100%; height: 100%; object-fit: contain; }}
    .card-body {{ padding: 13px 14px 15px; }}
    .meta-row {{ display: flex; justify-content: space-between; margin-bottom: 7px; }}
    .rank, .path {{ color: var(--muted); font-size: .82rem; }}
    .score {{ color: var(--accent); font-size: .8rem; font-weight: 700; }}
    .path {{ overflow-wrap: anywhere; line-height: 1.4; }}
    .download-link {{ display: inline-block; margin-top: 12px; color: var(--accent-dark); font-size: .84rem; font-weight: 650; text-decoration: none; }}
    .download-link:hover {{ text-decoration: underline; }}
    .message {{ color: var(--muted); }}
    .pagination {{ display: flex; align-items: center; justify-content: center; gap: 18px; margin: 24px 0; color: var(--muted); }}
    .page-link {{ padding: 9px 12px; border: 1px solid var(--border); border-radius: 8px; color: var(--text); background: #fff; text-decoration: none; }}
    .page-link.disabled {{ opacity: .45; }}
    button:disabled {{ opacity: .58; cursor: wait; }}
    button:focus-visible, a:focus-visible, summary:focus-visible {{ outline: 3px solid rgba(71, 122, 85, .35); outline-offset: 2px; }}
    @media (prefers-reduced-motion: reduce) {{ *, *::before, *::after {{ scroll-behavior: auto !important; transition-duration: .01ms !important; }} }}
    @media (max-width: 700px) {{
      body {{ padding: 0 16px 40px; }}
      .hero {{ margin: 0 -16px 28px; padding: 18px 16px; }}
      .hero-header {{ margin-bottom: 25px; }}
      .eyebrow {{ display: none; }}
      .search-form {{ flex-direction: column; }}
      .search-button {{ width: 100%; }}
      .search-options {{ align-items: flex-start; flex-direction: column; }}
      .image-form, .filters, .index-form {{ align-items: stretch; flex-direction: column; }}
      .image-button {{ width: 100%; }}
      .filters .folder-select {{ max-width: none; }}
      .grid {{ grid-template-columns: repeat(auto-fill, minmax(155px, 1fr)); gap: 12px; }}
      .card-body {{ padding: 10px; }}
      .results-meta {{ flex-wrap: wrap; }}
      .results-meta span {{ width: 100%; margin: 0; }}
    }}
  </style>
</head>
<body>
  <div class="page">
  <br><br>
    <section class="hero">
      <header class="hero-header">
        <div class="brand"><img class="brand-logo" src="https://mediaindia.eu/wp-content/themes/mig_theme/assets/images/mig-logo-200x140.png" alt="MIG"><span class="brand-mark" aria-hidden="true"></span>MIG VISUAL SEARCH</div>
         <span class="eyebrow"></span>
      </header>
      <form class="logout-row" action="/logout" method="post">
        <button class="logout-button" type="submit">Sign out</button>
      </form>
      <h1>Your visual library, searchable.</h1>
      <p class="intro">Find images by name, place, or what’s in the scene — or search with a reference image.</p>
      <form class="search-form" action="/" method="get">
        <input class="search-input" type="search" name="q"
          placeholder="Try &quot;Abu Dhabi&quot; or &quot;mosque at sunset&quot;"
          value="{html.escape(query, quote=True)}" aria-label="Image name or tag">
        <input type="hidden" name="folder" value="{html.escape(folder, quote=True)}">
        <button class="search-button" type="submit">Search</button>
        <div class="search-options">
          <span class="eyebrow">Describe the subject, location, or mood</span>
          <label class="broad-search">
            <input type="checkbox" name="broaden" value="1"{" checked" if broad_matches else ""}>
            Include more loosely related matches
          </label>
        </div>
      </form>
      <section class="image-search" aria-labelledby="image-search-title">
        <h2 class="image-search-title" id="image-search-title">Search with a reference image</h2>
        <form class="image-form" action="/search-image" method="post" enctype="multipart/form-data">
          <input type="hidden" name="folder" value="{html.escape(folder, quote=True)}">
          <input class="image-input" id="reference-image" type="file" name="image" accept="image/*" required
            aria-label="Choose an example image">
          <button class="image-button" type="submit">Find similar</button>
        </form>
        <div class="image-preview" id="image-preview" hidden>
          <img id="image-preview-thumb" alt="">
          <span id="image-preview-name"></span>
        </div>
      </section>
      <details class="index-tools">
        <summary>Add pictures to your index</summary>
        <form class="index-form" id="index-form">
          <input class="folder-input" id="scan-folder" type="text"
            value="{html.escape(scan_folder, quote=True)}"
            placeholder="Folder path containing pictures" aria-label="Folder path containing pictures" required>
          <button class="image-button" id="index-button" type="submit">Index folder</button>
        </form>
        <p class="index-status" id="index-status" role="status" aria-live="polite"></p>
      </details>
    </section>
    <form class="filters" action="/" method="get">
      <input type="hidden" name="q" value="{html.escape(query, quote=True)}">
      <input type="hidden" name="image_query" value="{html.escape(image_query, quote=True)}">
      <select class="folder-select" name="folder" aria-label="Filter by folder">
        {''.join(folder_options)}
      </select>
      <button class="image-button" type="submit">Filter</button>
    </form>
    {heading}
    {message}
    {pagination}
    <main class="grid">{''.join(cards)}</main>
    {pagination}
  </div>
  <script>
    const referenceImage = document.getElementById("reference-image");
    const imagePreview = document.getElementById("image-preview");
    const imagePreviewThumb = document.getElementById("image-preview-thumb");
    const imagePreviewName = document.getElementById("image-preview-name");
    let previewUrl;
    referenceImage.addEventListener("change", () => {{
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      const [file] = referenceImage.files;
      imagePreview.hidden = !file;
      if (file) {{
        previewUrl = URL.createObjectURL(file);
        imagePreviewThumb.src = previewUrl;
        imagePreviewName.textContent = file.name;
      }}
    }});

    const indexForm = document.getElementById("index-form");
    const indexButton = document.getElementById("index-button");
    const indexStatus = document.getElementById("index-status");
    let pollTimer;

    async function updateIndexStatus() {{
      try {{
        const response = await fetch("/index-status");
        const job = await response.json();
        if (!response.ok) throw new Error(job.error || "Could not load indexing status.");
        if (job.status === "scanning" || job.status === "indexing") {{
          indexStatus.textContent = job.status === "scanning"
            ? "Scanning folder..."
            : `Indexing ${{job.new_count}} pictures...`;
          pollTimer = window.setTimeout(updateIndexStatus, 700);
        }} else {{
          indexButton.disabled = false;
          indexStatus.textContent = job.message || "";
        }}
      }} catch (error) {{
        indexButton.disabled = false;
        indexStatus.textContent = error.message;
      }}
    }}

    indexForm.addEventListener("submit", async (event) => {{
      event.preventDefault();
      indexButton.disabled = true;
      indexStatus.textContent = "Scanning folder...";
      try {{
        const response = await fetch("/index", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ folder: document.getElementById("scan-folder").value }})
        }});
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Could not start indexing.");
        pollTimer = window.setTimeout(updateIndexStatus, 300);
      }} catch (error) {{
        indexButton.disabled = false;
        indexStatus.textContent = error.message;
      }}
    }});
  </script>
</body>
</html>"""


def make_handler(
  paths,
  vectors,
  labels,
  model,
  tokenizer,
  preprocess,
  min_score,
  result_count,
  login_username,
  login_password,
  index_dir=None,
  scan_folder="",
):
  image_queries = {}
  sessions = {}
  sessions_lock = threading.Lock()
  state = {
    "paths": paths,
    "vectors": vectors,
    "labels": labels,
    "job": {"status": "idle", "new_count": 0, "message": ""},
  }
  state_lock = threading.Lock()
  index_meta = {}
  if index_dir is not None and (index_dir / "meta.json").exists():
    index_meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))

  def session_for_request(handler):
    cookie = SimpleCookie()
    try:
      cookie.load(handler.headers.get("Cookie", ""))
    except CookieError:
      return None
    morsel = cookie.get("MIGVisualSession")
    if morsel is None:
      return None
    token = morsel.value
    now = time.monotonic()
    with sessions_lock:
      expired = [key for key, expiry in sessions.items() if expiry <= now]
      for key in expired:
        del sessions[key]
      if sessions.get(token, 0) <= now:
        return None
    return token

  def index_folder(folder):
    try:
      with state_lock:
        state["job"] = {"status": "scanning", "new_count": 0, "message": ""}
      found_paths = scan(folder)
      with state_lock:
        known_paths = {str(Path(path).resolve()).casefold() for path in state["paths"]}
      new_paths = [
        path for path in found_paths
        if str(Path(path).resolve()).casefold() not in known_paths
      ]
      with state_lock:
        state["job"] = {"status": "indexing", "new_count": len(new_paths), "message": ""}
      if not new_paths:
        with state_lock:
          state["job"] = {"status": "complete", "new_count": 0, "message": "No new pictures to index."}
        return

      loader = torch.utils.data.DataLoader(
        ImageDS(new_paths, preprocess),
        batch_size=32,
        num_workers=0,
        collate_fn=collate,
      )
      feature_batches = []
      indexed_paths = []
      indexed_labels = []
      with torch.no_grad():
        for batch in loader:
          if batch is None:
            continue
          indices, images = batch
          features = model.encode_image(images)
          features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
          feature_batches.append(features.cpu().numpy())
          batch_paths = [new_paths[index] for index in indices.tolist()]
          indexed_paths.extend(batch_paths)
          indexed_labels.extend(path_terms(path) for path in batch_paths)

      with state_lock:
        old_paths = state["paths"]
        old_vectors = state["vectors"]
        old_labels = state["labels"]
      combined_paths = old_paths + indexed_paths
      if feature_batches:
        new_vectors = np.concatenate(feature_batches, axis=0)
        combined_vectors = (
          new_vectors if old_vectors.size == 0
          else np.concatenate([old_vectors, new_vectors], axis=0)
        )
      else:
        combined_vectors = old_vectors
      combined_labels = old_labels + indexed_labels
      if indexed_paths and index_dir is not None:
        save(index_dir, combined_paths, combined_vectors, index_meta, combined_labels)

      with state_lock:
        state["paths"] = combined_paths
        state["vectors"] = combined_vectors
        state["labels"] = combined_labels
        state["job"] = {
          "status": "complete",
          "new_count": len(indexed_paths),
          "message": f"Indexed {len(indexed_paths)} new pictures.",
        }
    except Exception as exc:
      with state_lock:
        state["job"] = {"status": "error", "new_count": 0, "message": f"Indexing failed: {exc}"}

  class SearchHandler(BaseHTTPRequestHandler):
    def redirect(self, location):
      self.send_response(303)
      self.send_header("Location", location)
      self.send_header("Cache-Control", "no-store")
      self.send_header("Content-Length", "0")
      self.end_headers()

    def do_login(self):
      if session_for_request(self):
        self.redirect("/")
        return
      try:
        content_length = int(self.headers.get("Content-Length", "0"))
      except ValueError:
        content_length = 0
      if content_length < 1 or content_length > 8192:
        self.send_page(render_login_page("Invalid login request."), 400)
        return
      form = parse_qs(
        self.rfile.read(content_length).decode("utf-8", errors="replace"),
        keep_blank_values=True,
      )
      supplied_username = form.get("username", [""])[0]
      supplied_password = form.get("password", [""])[0]
      username_matches = hmac.compare_digest(
        supplied_username.encode("utf-8"), login_username.encode("utf-8")
      )
      password_matches = hmac.compare_digest(
        supplied_password.encode("utf-8"), login_password.encode("utf-8")
      )
      if not (username_matches and password_matches):
        self.send_page(
          render_login_page("Wrong credentials, please try again."),
          401,
        )
        return

      token = secrets.token_urlsafe(32)
      with sessions_lock:
        sessions[token] = time.monotonic() + SESSION_TTL_SECONDS
      self.send_response(303)
      self.send_header("Location", "/")
      self.send_header(
        "Set-Cookie",
        f"MIGVisualSession={token}; Path=/; Max-Age={SESSION_TTL_SECONDS}; "
        "HttpOnly; SameSite=Strict",
      )
      self.send_header("Cache-Control", "no-store")
      self.send_header("Content-Length", "0")
      self.end_headers()

    def do_logout(self):
      token = session_for_request(self)
      if token:
        with sessions_lock:
          sessions.pop(token, None)
      self.send_response(303)
      self.send_header("Location", "/login")
      self.send_header(
        "Set-Cookie",
        "MIGVisualSession=; Path=/; Max-Age=0; HttpOnly; SameSite=Strict",
      )
      self.send_header("Cache-Control", "no-store")
      self.send_header("Content-Length", "0")
      self.end_headers()

    def do_GET(self):
      request = urlsplit(self.path)
      authenticated = session_for_request(self) is not None
      if request.path == "/login":
        if authenticated:
          self.redirect("/")
        else:
          self.send_page(render_login_page())
        return
      if request.path == "/logout":
        self.send_error(405)
        return
      if not authenticated:
        if request.path == "/index-status":
          self.send_json({"error": "Please sign in."}, 401)
        else:
          self.redirect("/login")
        return
      if request.path == "/index-status":
        with state_lock:
          job = dict(state["job"])
        self.send_json(job)
        return
      if request.path == "/visual-search-background.png":
        self.serve_background()
        return
      if request.path.startswith("/download/"):
        self.serve_image(request.path.removeprefix("/download/"), download=True)
        return
      if request.path.startswith("/image/"):
        self.serve_image(request.path.removeprefix("/image/"))
        return
      if request.path != "/":
        self.send_error(404)
        return

      params = parse_qs(request.query, keep_blank_values=True)
      query = params.get("q", [""])[0].strip()
      folder = params.get("folder", [""])[0]
      image_query = params.get("image_query", [""])[0]
      if len(query) > 300:
        self.send_page(
          render_page(error="Search text must be 300 characters or fewer.", scan_folder=scan_folder),
          400,
        )
        return
      try:
        page = max(1, int(params.get("page", ["1"])[0]))
      except ValueError:
        page = 1

      with state_lock:
        current_paths = state["paths"]
        current_vectors = state["vectors"]
        current_labels = state["labels"]
      folders = sorted({Path(path).parent for path in current_paths}, key=lambda item: str(item).casefold())
      eligible = np.arange(len(current_paths))
      if folder:
        eligible = np.array(
          [index for index in eligible if str(Path(current_paths[index]).parent) == folder],
          dtype=np.int64,
        )

      results = []
      result_total = 0
      page_count = 1
      if query or image_query or folder:
        if query:
          with torch.no_grad():
            embedding = build_text_embedding(model, tokenizer, query)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
          query_vector = embedding[0].cpu().numpy()
        elif image_query:
          query_vector = image_queries.get(image_query)
          if query_vector is None:
            self.send_page(
              render_page(error="This image search has expired. Please upload the image again."),
              410,
            )
            return
        else:
          query_vector = None

        if query_vector is None:
          clip_scores = np.zeros(len(current_paths), dtype=np.float32)
          ranked = eligible
        else:
          clip_scores = current_vectors @ query_vector
          final_scores = clip_scores.copy()
          if query:
            lexical_scores = np.array(
              [
                lexical_match_score(
                  query,
                  " ".join(current_labels[index]) if index < len(current_labels) else current_paths[index],
                )
                for index in range(len(current_paths))
              ],
              dtype=np.float32,
            )
            final_scores += 0.18 * lexical_scores
            threshold = max(0.0, min_score - 0.04) if params.get("broaden", [""])[0] == "1" else min_score
            eligible = eligible[clip_scores[eligible] >= threshold]
          ranked = eligible[np.argsort(-final_scores[eligible], kind="stable")]

        result_total = len(ranked)
        page_count = max(1, (result_total + result_count - 1) // result_count)
        page = min(page, page_count)
        selected = ranked[(page - 1) * result_count : page * result_count]
        results = [
          (int(index), Path(current_paths[index]), float(clip_scores[index]))
          for index in selected
        ]

      self.send_page(
        render_page(
          query=query,
          results=results,
          folders=folders,
          folder=folder,
          page=page,
          page_count=page_count,
          result_total=result_total,
          image_query=image_query,
          page_size=result_count,
          scan_folder=scan_folder,
          broad_matches=(params.get("broaden", [""])[0] == "1"),
        )
      )

    def do_POST(self):
      request_path = urlsplit(self.path).path
      if request_path == "/login":
        self.do_login()
        return
      if request_path == "/logout":
        self.do_logout()
        return
      if session_for_request(self) is None:
        if request_path == "/index":
          self.send_json({"error": "Please sign in."}, 401)
        else:
          self.redirect("/login")
        return
      if request_path == "/index":
        self.index_new_folder()
        return
      if request_path != "/search-image":
        self.send_error(404)
        return
      try:
        content_length = int(self.headers.get("Content-Length", "0"))
      except ValueError:
        content_length = 0
      if content_length < 1 or content_length > MAX_UPLOAD_BYTES:
        self.send_page(render_page(error="Upload an image smaller than 15 MB."), 413)
        return

      content_type = self.headers.get("Content-Type", "")
      headers = (
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n"
      ).encode("latin-1", errors="replace")
      message = BytesParser(policy=policy.default).parsebytes(
        headers + self.rfile.read(content_length)
      )
      image_bytes = None
      folder_value = ""
      if message.is_multipart():
        for part in message.iter_parts():
          if part.get_content_disposition() != "form-data":
            continue
          field_name = part.get_param("name", header="content-disposition")
          if  field_name == "image":
            image_bytes = part.get_payload(decode=True)
          elif field_name == "folder":
            folder_value = (part.get_payload(decode=True) or b"").decode(
              "utf-8", errors="replace"
            )
      if not image_bytes:
        self.send_page(render_page(error="Choose a valid image to search."), 400)
        return

      try:
        with Image.open(io.BytesIO(image_bytes)) as uploaded:
          if uploaded.width * uploaded.height > 100_000_000:
            raise ValueError("Image dimensions are too large.")
          image = uploaded.convert("RGB")
        with torch.no_grad():
          embedding = model.encode_image(preprocess(image).unsqueeze(0))
          embedding = embedding / embedding.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        token = secrets.token_urlsafe(24)
        image_queries[token] = embedding[0].cpu().numpy()
        while len(image_queries) > 32:
          image_queries.pop(next(iter(image_queries)))
      except Exception:
        self.send_page(render_page(error="The uploaded file could not be read as an image."), 400)
        return

      self.send_response(303)
      query_params = {"image_query": token}
      if folder_value:
        query_params["folder"] = folder_value
      self.send_header("Location", f"/?{urlencode(query_params)}")
      self.send_header("Content-Length", "0")
      self.end_headers()

    def serve_image(self, image_id, download=False):
      if not image_id.isdigit():
        self.send_error(404)
        return
      with state_lock:
        current_paths = state["paths"]
      if int(image_id) >= len(current_paths):
        self.send_error(404)
        return
      image_path = Path(current_paths[int(image_id)])
      content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
      with image_path.open("rb") as image_file:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(image_file.seek(0, 2)))
        self.send_header("Cache-Control", "private, no-store")
        if download:
          self.send_header(
            "Content-Disposition",
            f"attachment; filename*=UTF-8''{quote(image_path.name, safe='')}",
          )
        self.end_headers()
        image_file.seek(0)
        while chunk := image_file.read(64 * 1024):
          self.wfile.write(chunk)

    def serve_background(self):
      image_path = Path(__file__).parent / "static" / "visual-search-background.png"
      with image_path.open("rb") as image_file:
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(image_path.stat().st_size))
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        while chunk := image_file.read(64 * 1024):
         self.wfile.write(chunk)

    def send_page(self, page, status=200):
      content = page.encode("utf-8")
      self.send_response(status)
      self.send_header("Content-Type", "text/html; charset=utf-8")
      self.send_header("Content-Length", str(len(content)))
      self.send_header("Cache-Control", "no-store")
      self.end_headers()
      self.wfile.write(content)

    def send_json(self, payload, status=200):
      content = json.dumps(payload).encode("utf-8")
      self.send_response(status)
      self.send_header("Content-Type", "application/json; charset=utf-8")
      self.send_header("Content-Length", str(len(content)))
      self.send_header("Cache-Control", "no-store")
      self.end_headers()
      self.wfile.write(content)

    def index_new_folder(self):
      try:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length < 1 or content_length > 16_384:
          raise ValueError("Request body must be between 1 byte and 16 KB.")
        payload = json.loads(self.rfile.read(content_length))
        if not isinstance(payload, dict):
          raise ValueError("Request body must be a JSON object.")
        folder_value = payload.get("folder")
        if not isinstance(folder_value, str) or not folder_value.strip():
          raise ValueError("Enter a folder path to scan.")
        folder_path = Path(folder_value.strip()).expanduser()
        if not folder_path.is_dir():
          raise ValueError(f"Folder does not exist: {folder_path}")
      except (ValueError, json.JSONDecodeError) as exc:
        self.send_json({"error": str(exc)}, 400)
        return

      with state_lock:
        if state["job"]["status"] in ("scanning", "indexing"):
          self.send_json({"error": "Indexing is already in progress."}, 409)
          return
        state["job"] = {"status": "scanning", "new_count": 0, "message": ""}
        thread = threading.Thread(
          target=index_folder,
          args=(str(folder_path.resolve()),),
          daemon=True,
        )
        thread.start()
      self.send_json({"status": "started"}, 202)

    def log_message(self, format_string, *args):
      print(f"{self.address_string()} - {format_string % args}")

  return SearchHandler


def main():
    parser = argparse.ArgumentParser(description="Start the local image search web interface.")
    parser.add_argument("--index", default="index_data", help="folder containing the image index")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="host interface to bind (default: localhost)",
    )
    parser.add_argument("--port", type=int, default=8000, help="web server port")
    parser.add_argument(
      "--scan-folder",
      default=r"C:\MIG media hub test\test folder",
      help="default folder to scan for new pictures",
    )
    parser.add_argument("-k", type=int, default=20, help="maximum results to show")
    parser.add_argument("--min-score", type=float, default=0.28, help="minimum CLIP similarity")
    args = parser.parse_args()
    if args.port < 1 or args.port > 65535:
        parser.error("--port must be between 1 and 65535")
    if args.k < 1:
        parser.error("-k must be at least 1")
    if not 0.0 <= args.min_score <= 1.0:
        parser.error("--min-score must be between 0 and 1")
    login_username = os.environ.get("MIG_VISUAL_SEARCH_USERNAME")
    login_password = os.environ.get("MIG_VISUAL_SEARCH_PASSWORD")
    if not login_username or not login_password:
        parser.error(
            "Set both MIG_VISUAL_SEARCH_USERNAME and MIG_VISUAL_SEARCH_PASSWORD "
            "environment variables before starting the app"
        )

    index = Path(args.index)
    metadata = json.loads((index / "meta.json").read_text(encoding="utf-8"))
    paths = json.loads((index / "paths.json").read_text(encoding="utf-8"))
    vectors = np.load(index / "vectors.npy")
    labels = (
        json.loads((index / "labels.json").read_text(encoding="utf-8"))
        if (index / "labels.json").exists()
        else [[path] for path in paths]
    )
    print(f"Loading {metadata['model']} ({metadata['pretrained']})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        metadata["model"], pretrained=metadata["pretrained"]
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(metadata["model"])

    server = ThreadingHTTPServer(
        (args.host, args.port),
      make_handler(
        paths,
        vectors,
        labels,
        model,
        tokenizer,
        preprocess,
        args.min_score,
        args.k,
        login_username,
        login_password,
        index,
        args.scan_folder,
      ),
    )
    if args.host == "0.0.0.0":
        print(f"Image search ready on all network interfaces at port {args.port}.")
        print(f"On this PC: http://127.0.0.1:{args.port}")
        print(f"On another device: http://<this-PC's IPv4 address>:{args.port}")
    else:
        print(f"Image search ready: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping image search.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()