/**
 * YouTube Transcript Fetcher — Cloudflare Worker
 *
 * Serves an HTML UI at `/` and scoped APIs at:
 *   GET /api/transcript?v=VIDEO_ID
 *   GET /api/playlist?list=PLAYLIST_ID
 *
 * Only fetches youtube.com — not an open proxy.
 *
 * Deploy:
 *   npm install -g wrangler
 *   wrangler login
 *   wrangler deploy
 */

const ALLOWED_HOSTS = new Set([
  'www.youtube.com',
  'youtube.com',
  'm.youtube.com',
]);

// InnerTube clients to try, in order. ANDROID first because it has lax rate-limiting
// and avoids PoToken requirements that increasingly affect the WEB client.
// API keys are hardcoded into YouTube's official apps — they're not secrets, but YouTube
// could rotate them. If transcripts mysteriously start failing, refresh these from a fresh
// watch-page scrape: curl https://www.youtube.com/watch?v=dQw4w9WgXcQ | grep -o 'INNERTUBE_API_KEY":"[^"]*'
const INNERTUBE_CLIENTS = [
  {
    name: 'ANDROID',
    apiKey: 'AIzaSyA8eiZmM1FaDVjRy-df2KTyQ_vz_yYM39w',
    context: { client: { clientName: 'ANDROID', clientVersion: '20.10.38' } },
    userAgent: 'com.google.android.youtube/20.10.38 (Linux; U; Android 13) gzip',
  },
  {
    name: 'WEB',
    apiKey: 'AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8',
    context: { client: { clientName: 'WEB', clientVersion: '2.20240101.00.00' } },
    userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
  },
];

const PLAYLIST_CONCURRENCY = 5;

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    if (url.pathname === '/' || url.pathname === '/index.html') {
      return serveHTML(request);
    }
    if (url.pathname === '/api/transcript') {
      return handleTranscript(url.searchParams.get('v'), request);
    }
    if (url.pathname === '/api/playlist') {
      return handlePlaylist(url.searchParams.get('list'), request);
    }
    return new Response('Not found', { status: 404 });
  },
};

// --- API: single video transcript ---

async function handleTranscript(videoId, request) {
  if (!videoId || !/^[a-zA-Z0-9_-]{11}$/.test(videoId)) {
    return json({ error: 'Invalid or missing video ID' }, 400);
  }
  const user = request.headers.get('Cf-Access-Authenticated-User-Email');
  if (user) console.log(`Transcript: ${videoId} by ${user}`);

  try {
    const result = await fetchTranscriptForVideo(videoId);
    return json(result);
  } catch (e) {
    return json({ error: e.message }, 500);
  }
}

async function fetchTranscriptForVideo(videoId) {
  const playerData = await callInnerTube('/youtubei/v1/player', { videoId });

  const tracks = playerData?.captions?.playerCaptionsTracklistRenderer?.captionTracks || [];
  const title = playerData?.videoDetails?.title || videoId;
  const author = playerData?.videoDetails?.author || '';

  if (!tracks.length) {
    throw new Error(`No captions available for "${title}"`);
  }

  const track = tracks.find(t => t.languageCode === 'en' && t.kind !== 'asr')
             || tracks.find(t => t.languageCode?.startsWith('en'))
             || tracks[0];

  const captionUrl = setFmtJson3(track.baseUrl);
  const segments = await fetchCaptions(captionUrl);

  return {
    videoId,
    title,
    author,
    language: track.languageCode,
    kind: track.kind === 'asr' ? 'auto' : 'manual',
    segments,
  };
}

// --- API: playlist ---

async function handlePlaylist(playlistId, request) {
  if (!playlistId || !/^[a-zA-Z0-9_-]{10,}$/.test(playlistId)) {
    return json({ error: 'Invalid or missing playlist ID' }, 400);
  }
  const user = request.headers.get('Cf-Access-Authenticated-User-Email');
  if (user) console.log(`Playlist: ${playlistId} by ${user}`);

  try {
    const { title: playlistTitle, videos } = await getPlaylistVideos(playlistId);
    if (!videos.length) {
      return json({ error: 'Playlist is empty or unavailable' }, 404);
    }

    // Cloudflare Workers free tier allows 50 subrequests per request.
    // Playlist enumeration uses 1 subrequest per page (WEB client, up to ~100 videos per page).
    // Each video then uses 2 subrequests in the best case (ANDROID player + caption fetch),
    // or up to 4 in the worst case if we retry with WEB.
    // Typical budget: 1 + 2×N. Worst case: 1 + 4×N.
    // Conservative threshold: warn over 20 videos.
    const truncated = videos.length > 20;
    const toProcess = videos;  // Don't truncate; let the user see what happened

    const results = await parallelMap(toProcess, PLAYLIST_CONCURRENCY, async (video) => {
      try {
        const t = await fetchTranscriptForVideo(video.videoId);
        return { ok: true, ...t };
      } catch (e) {
        return {
          ok: false,
          videoId: video.videoId,
          title: video.title,
          error: e.message,
        };
      }
    });

    return json({
      playlistId,
      playlistTitle,
      count: results.length,
      truncated,
      videos: results,
    });
  } catch (e) {
    return json({ error: e.message }, 500);
  }
}

// Run async fn across items with at most `concurrency` in flight. Preserves order.
async function parallelMap(items, concurrency, fn) {
  const results = new Array(items.length);
  let index = 0;
  async function worker() {
    while (true) {
      const i = index++;
      if (i >= items.length) return;
      results[i] = await fn(items[i], i);
    }
  }
  const workers = Array.from({ length: Math.min(concurrency, items.length) }, () => worker());
  await Promise.all(workers);
  return results;
}

async function getPlaylistVideos(playlistId) {
  // Use the WEB client specifically for playlist enumeration. WEB returns more
  // videos per page than ANDROID and its response shape is more predictable.
  // Playlists aren't rate-limit-sensitive the way player calls are.
  const webClient = INNERTUBE_CLIENTS.find(c => c.name === 'WEB');

  const allVideos = [];
  let title = playlistId;
  let payload = { browseId: 'VL' + playlistId };
  let safetyLimit = 10; // up to ~1000 videos via continuation pagination

  while (safetyLimit-- > 0) {
    const data = await callInnerTubeWithClient(webClient, '/youtubei/v1/browse', payload);

    // Extract the playlist title (only present on the first response)
    if (title === playlistId) {
      title = data?.metadata?.playlistMetadataRenderer?.title
           || data?.header?.playlistHeaderRenderer?.title?.simpleText
           || data?.header?.pageHeaderRenderer?.pageTitle
           || playlistId;
    }

    // Recursively find every playlistVideoRenderer anywhere in the response.
    // More resilient than walking specific paths, which vary across clients
    // and YouTube layout changes.
    const videoNodes = findAll(data, 'playlistVideoRenderer');
    for (const r of videoNodes) {
      if (!r?.videoId) continue;
      allVideos.push({
        videoId: r.videoId,
        title: r.title?.runs?.[0]?.text || r.title?.simpleText || r.videoId,
      });
    }

    // Look for a continuation token to fetch the next page
    const continuations = findAll(data, 'continuationCommand');
    const token = continuations.find(c => c.token)?.token;
    if (!token) break;
    payload = { continuation: token };
  }

  return { title, videos: allVideos };
}

// Recursively walk any JSON value and return every value found at a property named `keyName`.
// Preserves natural array order, which matters for playlist video ordering.
function findAll(node, keyName) {
  const out = [];
  function walk(v) {
    if (!v || typeof v !== 'object') return;
    if (Array.isArray(v)) {
      for (const item of v) walk(item);
      return;
    }
    for (const k in v) {
      if (k === keyName) out.push(v[k]);
      walk(v[k]);
    }
  }
  walk(node);
  return out;
}

// --- InnerTube helper ---

// Single attempt with a specific client. Throws on failure.
async function callInnerTubeWithClient(client, path, payload) {
  const url = 'https://www.youtube.com' + path + '?key=' + client.apiKey + '&prettyPrint=false';
  const res = await safeFetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'User-Agent': client.userAgent,
      'X-Youtube-Client-Name': client.context.client.clientName,
      'X-Youtube-Client-Version': client.context.client.clientVersion,
    },
    body: JSON.stringify({ context: client.context, ...payload }),
  });
  if (!res.ok) throw new Error(`${client.name}: HTTP ${res.status}`);
  const data = await res.json();
  const playabilityStatus = data?.playabilityStatus?.status;
  if (playabilityStatus && playabilityStatus !== 'OK') {
    throw new Error(`${client.name}: ${data.playabilityStatus.reason || playabilityStatus}`);
  }
  return data;
}

// Try each client in order until one succeeds. Used for player calls where
// ANDROID-first-then-WEB fallback is important.
async function callInnerTube(path, payload) {
  const errors = [];
  for (const client of INNERTUBE_CLIENTS) {
    try {
      return await callInnerTubeWithClient(client, path, payload);
    } catch (e) {
      errors.push(e.message);
    }
  }
  throw new Error(`All InnerTube clients failed. ${errors.join('; ')}`);
}

async function fetchCaptions(captionUrl) {
  const res = await safeFetch(captionUrl, {
    headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36' },
  });
  if (!res.ok) throw new Error(`Caption fetch returned ${res.status}`);
  const body = await res.text();
  const trimmed = body.trimStart();

  // JSON3 path (preferred)
  if (trimmed.startsWith('{')) {
    const data = JSON.parse(trimmed);
    const out = [];
    for (const ev of data.events || []) {
      if (!ev.segs) continue;
      const text = ev.segs.map(s => s.utf8 || '').join('').replace(/\n/g, ' ').trim();
      if (text) out.push({ start: (ev.tStartMs || 0) / 1000, text });
    }
    return out;
  }

  // XML fallback (srv1/srv3) — older format YouTube still returns sometimes
  if (trimmed.startsWith('<')) {
    return parseCaptionXml(trimmed);
  }

  throw new Error('Caption response was neither JSON nor XML');
}

function parseCaptionXml(xml) {
  const out = [];
  const re = /<text[^>]*\bstart="([^"]+)"[^>]*>([\s\S]*?)<\/text>/g;
  let m;
  while ((m = re.exec(xml)) !== null) {
    const text = decodeXmlEntities(stripTags(m[2])).replace(/\n/g, ' ').trim();
    if (text) out.push({ start: parseFloat(m[1]), text });
  }
  return out;
}

// Strip HTML/XML-like tags. Applied iteratively until stable so that nested
// patterns like "<scr<script>ipt>" don't leak a tag through a single pass.
function stripTags(s) {
  let prev;
  do {
    prev = s;
    s = s.replace(/<[^>]*>/g, '');
  } while (s !== prev);
  return s;
}

// Decode XML entities in a single pass so that "&amp;lt;" stays as "&lt;"
// (rather than being re-decoded into "<"). Chained .replace() calls would
// double-unescape, which CodeQL correctly flags.
const NAMED_ENTITIES = {
  amp: '&',
  lt: '<',
  gt: '>',
  quot: '"',
  apos: "'",
};

function decodeXmlEntities(s) {
  return s.replace(/&(#x[0-9a-fA-F]+|#[0-9]+|[a-zA-Z]+);/g, (match, body) => {
    if (body[0] === '#') {
      const code = body[1] === 'x' || body[1] === 'X'
        ? parseInt(body.slice(2), 16)
        : parseInt(body.slice(1), 10);
      if (Number.isFinite(code) && code >= 0 && code <= 0x10FFFF) {
        try { return String.fromCodePoint(code); } catch (e) { return match; }
      }
      return match;
    }
    return NAMED_ENTITIES[body] !== undefined ? NAMED_ENTITIES[body] : match;
  });
}

async function safeFetch(targetUrl, options) {
  const u = new URL(targetUrl);
  if (!ALLOWED_HOSTS.has(u.hostname)) {
    throw new Error(`Refusing to fetch ${u.hostname} — not on allowlist`);
  }
  return fetch(targetUrl, options);
}

function appendParam(url, key, value) {
  return url + (url.includes('?') ? '&' : '?') + key + '=' + encodeURIComponent(value);
}

// YouTube caption baseUrl often already has &fmt=srv3 (XML format). The caption
// server honours the first fmt= parameter it sees, so we must strip any existing
// one before appending fmt=json3.
function setFmtJson3(url) {
  const stripped = url.replace(/[?&]fmt=[^&]*/g, '');
  return appendParam(stripped, 'fmt', 'json3');
}

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

// --- UI ---

function serveHTML(request) {
  const user = request.headers.get('Cf-Access-Authenticated-User-Email') || '';
  return new Response(HTML.replace('__USER_EMAIL__', escapeHtml(user)), {
    headers: { 'Content-Type': 'text/html; charset=utf-8' },
  });
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

const HTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>YouTube Transcript Fetcher</title>
<style>
  :root {
    --bg: #ffffff; --surface: #f5f5f4; --border: rgba(0,0,0,0.12);
    --text: #1a1a19; --text-muted: #6b6b66;
    --accent: #c84a23; --danger: #b91c1c; --success: #15803d;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #1a1a19; --surface: #2a2a28; --border: rgba(255,255,255,0.15);
      --text: #f5f5f4; --text-muted: #a8a8a3;
    }
  }
  * { box-sizing: border-box; }
  body {
    font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
    background: var(--bg); color: var(--text);
    max-width: 820px; margin: 0 auto; padding: 2rem 1.25rem; line-height: 1.5;
  }
  header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.5rem; }
  h1 { font-size: 22px; font-weight: 500; margin: 0; }
  .user { font-size: 12px; color: var(--text-muted); font-family: ui-monospace, monospace; }
  p.lede { color: var(--text-muted); font-size: 14px; margin: 0 0 1.5rem; }
  p.lede .lede-note { display: block; font-size: 12px; margin-top: 4px; }
  .row { display: flex; gap: 8px; margin-bottom: 1rem; }
  input[type="text"] {
    flex: 1; height: 38px; padding: 0 12px; border-radius: 8px;
    border: 1px solid var(--border); background: var(--bg); color: var(--text);
    font-size: 14px; font-family: inherit;
  }
  input[type="text"]:focus {
    outline: none; border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(200, 74, 35, 0.15);
  }
  button {
    height: 32px; padding: 0 12px; border-radius: 6px;
    border: 1px solid var(--border); background: var(--surface); color: var(--text);
    font-size: 13px; font-family: inherit; cursor: pointer; transition: background 0.15s;
    display: inline-flex; align-items: center; gap: 4px;
  }
  button:hover:not(:disabled) { background: var(--border); }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  button.primary { background: var(--accent); color: white; border-color: var(--accent); height: 38px; padding: 0 16px; font-size: 14px; }
  button.primary:hover:not(:disabled) { background: #a83d1c; }
  .status { font-size: 13px; color: var(--text-muted); margin-bottom: 12px; min-height: 18px; font-family: ui-monospace, monospace; }
  .status.error { color: var(--danger); }
  .status.success { color: var(--success); }

  /* Single video view */
  textarea {
    width: 100%; min-height: 320px; padding: 12px; border-radius: 8px;
    border: 1px solid var(--border); background: var(--surface); color: var(--text);
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: 13px; line-height: 1.6; resize: vertical;
  }
  .actions { display: flex; gap: 8px; margin-top: 12px; align-items: center; flex-wrap: wrap; }
  .toggle { display: flex; align-items: center; gap: 6px; font-size: 13px; color: var(--text-muted); cursor: pointer; margin-left: 4px; }
  .meta { font-size: 12px; color: var(--text-muted); margin-left: auto; font-family: ui-monospace, monospace; text-align: right; }

  /* Playlist view */
  .playlist-header {
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 12px 16px; margin-bottom: 16px;
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
    position: sticky; top: 0; z-index: 10;
  }
  .playlist-title { font-weight: 500; flex: 1; min-width: 200px; }
  .playlist-title .sub { font-size: 12px; color: var(--text-muted); font-weight: 400; }
  .playlist-actions { display: flex; gap: 6px; align-items: center; }
  .playlist-options { margin: 0 4px 14px; }

  .video-card {
    border: 1px solid var(--border); border-radius: 8px; margin-bottom: 12px;
    background: var(--bg);
  }
  .video-card.failed { opacity: 0.6; }
  .video-card-header {
    padding: 8px 14px; display: flex; align-items: center; gap: 8px;
    border-bottom: 1px solid var(--border); user-select: none;
  }
  .video-card.collapsed .video-card-header { border-bottom: none; }
  .video-num {
    font-family: ui-monospace, monospace; font-size: 12px; color: var(--text-muted);
    min-width: 28px;
  }
  .video-title-area { flex: 1; min-width: 0; cursor: pointer; }
  .video-title-area:hover .video-title { color: var(--accent); }
  .video-title {
    font-size: 14px; font-weight: 500;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .video-sub { font-size: 11px; color: var(--text-muted); font-family: ui-monospace, monospace; }
  .video-card-actions { display: flex; gap: 6px; align-items: center; flex-shrink: 0; }
  .video-card-actions button { height: 28px; padding: 0 10px; font-size: 12px; }
  .chevron-btn {
    background: transparent !important; border: none !important;
    color: var(--text-muted); padding: 4px 6px !important; height: 28px;
    cursor: pointer;
  }
  .chevron-btn:hover { color: var(--text); }
  .chevron { display: inline-block; font-size: 14px; transition: transform 0.15s; }
  .video-card.collapsed .chevron { transform: rotate(-90deg); }
  .video-card-body { padding: 8px 14px 12px; }
  .video-card.collapsed .video-card-body { display: none; }
  .video-transcript {
    background: var(--surface); border-radius: 6px; padding: 10px 12px;
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: 12px; line-height: 1.6;
    max-height: 240px; overflow-y: auto; white-space: pre-wrap;
  }
  .video-error {
    background: var(--surface); border-radius: 6px; padding: 10px 12px;
    font-size: 13px; color: var(--danger);
  }

  .toast {
    position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
    background: var(--text); color: var(--bg); padding: 8px 16px; border-radius: 6px;
    font-size: 13px; opacity: 0; pointer-events: none; transition: opacity 0.2s; z-index: 100;
  }
  .toast.show { opacity: 1; }
  .hidden { display: none !important; }
</style>
</head>
<body>
  <header>
    <h1>YouTube transcript fetcher</h1>
    <span class="user">__USER_EMAIL__</span>
  </header>
  <p class="lede">
    Paste a YouTube video URL, playlist URL, video ID, or playlist ID.
    <span class="lede-note">Playlists up to ~20 videos work reliably on Cloudflare's free tier; larger needs the paid plan.</span>
  </p>

  <div class="row">
    <input type="text" id="url" placeholder="https://www.youtube.com/watch?v=... or ?list=..." />
    <button id="fetchBtn" class="primary">Fetch</button>
  </div>

  <div class="status" id="status">Ready.</div>

  <!-- Single video view -->
  <div id="singleView" class="hidden">
    <textarea id="output" placeholder="Transcript will appear here..." readonly></textarea>
    <div class="actions">
      <button id="copyBtn"><span>\u29C9</span> Copy</button>
      <button id="downloadBtn"><span>\u2193</span> Download</button>
      <label class="toggle"><input type="checkbox" id="timestamps" /> Include timestamps</label>
      <span class="meta" id="meta"></span>
    </div>
  </div>

  <!-- Playlist view -->
  <div id="playlistView" class="hidden">
    <div class="playlist-header">
      <div class="playlist-title">
        <div id="playlistName">Playlist</div>
        <div class="sub" id="playlistSub"></div>
      </div>
      <div class="playlist-actions">
        <button id="copyAllBtn"><span>\u29C9</span> Copy all</button>
        <button id="downloadAllBtn"><span>\u2193</span> Download all</button>
      </div>
    </div>
    <div class="playlist-options">
      <label class="toggle">
        <input type="checkbox" id="timestampsPlaylist" />
        Include timestamps in copied and downloaded transcripts
      </label>
    </div>
    <div id="videoList"></div>
  </div>

  <div class="toast" id="toast"></div>

<script>
(function(){
  const urlInput = document.getElementById('url');
  const fetchBtn = document.getElementById('fetchBtn');
  const status = document.getElementById('status');
  const toast = document.getElementById('toast');

  const singleView = document.getElementById('singleView');
  const playlistView = document.getElementById('playlistView');

  // Single video elements
  const output = document.getElementById('output');
  const copyBtn = document.getElementById('copyBtn');
  const downloadBtn = document.getElementById('downloadBtn');
  const tsToggle = document.getElementById('timestamps');
  const meta = document.getElementById('meta');

  // Playlist elements
  const playlistName = document.getElementById('playlistName');
  const playlistSub = document.getElementById('playlistSub');
  const videoList = document.getElementById('videoList');
  const copyAllBtn = document.getElementById('copyAllBtn');
  const downloadAllBtn = document.getElementById('downloadAllBtn');
  const tsTogglePlaylist = document.getElementById('timestampsPlaylist');

  let current = null;

  function setStatus(msg, kind) {
    status.textContent = msg;
    status.className = 'status' + (kind ? ' ' + kind : '');
  }

  function showToast(msg) {
    toast.textContent = msg;
    toast.classList.add('show');
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => toast.classList.remove('show'), 1800);
  }

  function parseInput(input) {
    input = input.trim();
    if (/^[a-zA-Z0-9_-]{11}$/.test(input)) return { kind: 'video', id: input };
    if (/^(PL|UU|LL|FL|RD|OL)[a-zA-Z0-9_-]{10,}$/.test(input)) return { kind: 'playlist', id: input };

    try {
      const u = new URL(input);
      const list = u.searchParams.get('list');
      const v = u.searchParams.get('v');
      if (list && (u.pathname === '/playlist' || !v)) return { kind: 'playlist', id: list };
      if (v) return { kind: 'video', id: v };
      if (list) return { kind: 'playlist', id: list };
      if (u.hostname === 'youtu.be') return { kind: 'video', id: u.pathname.slice(1) };
      const m = u.pathname.match(/\\/(embed|shorts|live)\\/([a-zA-Z0-9_-]{11})/);
      if (m) return { kind: 'video', id: m[2] };
    } catch (e) {}

    const m = input.match(/([a-zA-Z0-9_-]{11})/);
    if (m) return { kind: 'video', id: m[1] };
    return null;
  }

  function formatTime(seconds) {
    const s = Math.floor(seconds);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    if (h > 0) return h + ':' + String(m).padStart(2,'0') + ':' + String(sec).padStart(2,'0');
    return m + ':' + String(sec).padStart(2,'0');
  }

  function renderSegments(segments, withTimestamps) {
    if (withTimestamps) {
      return segments.map(s => '[' + formatTime(s.start) + '] ' + s.text).join('\\n');
    }
    return segments.map(s => s.text).join(' ').replace(/\\s+/g, ' ').trim();
  }

  function sanitizeFilename(s) {
    return s.replace(/[\\\\/:*?"<>|]/g, '_').replace(/\\s+/g, ' ').trim().slice(0, 80) || 'transcript';
  }

  function downloadText(filename, content) {
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      URL.revokeObjectURL(a.href);
      a.remove();
    }, 100);
  }

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (e) {
      return false;
    }
  }

  // --- Single video render ---

  function renderSingle() {
    if (!current || current.kind !== 'video') return;
    output.value = renderSegments(current.data.segments, tsToggle.checked);
  }

  function showSingle(data) {
    current = { kind: 'video', data };
    singleView.classList.remove('hidden');
    playlistView.classList.add('hidden');
    renderSingle();
    meta.textContent = data.videoId + ' \u00b7 ' + data.language + ' (' + data.kind + ')';
  }

  // --- Playlist render ---

  function buildPlaylistText(withTimestamps) {
    const d = current.data;
    const parts = [];
    parts.push('# ' + (d.playlistTitle || d.playlistId));
    parts.push('');
    for (const v of d.videos) {
      parts.push('---');
      parts.push('## ' + (v.title || v.videoId));
      if (v.author) parts.push(v.author);
      parts.push('https://youtu.be/' + v.videoId);
      parts.push('');
      if (v.ok) {
        parts.push(renderSegments(v.segments, withTimestamps));
      } else {
        parts.push('[no transcript: ' + v.error + ']');
      }
      parts.push('');
    }
    return parts.join('\\n');
  }

  function renderPlaylistVideoBodies() {
    if (!current || current.kind !== 'playlist') return;
    const withTimestamps = tsTogglePlaylist.checked;
    const cards = videoList.querySelectorAll('.video-card');
    current.data.videos.forEach((v, i) => {
      const card = cards[i];
      if (!card || !v.ok) return;
      const transcriptEl = card.querySelector('.video-transcript');
      if (transcriptEl) transcriptEl.textContent = renderSegments(v.segments, withTimestamps);
    });
  }

  function showPlaylist(data) {
    current = { kind: 'playlist', data };
    singleView.classList.add('hidden');
    playlistView.classList.remove('hidden');

    playlistName.textContent = data.playlistTitle || data.playlistId;
    const ok = data.videos.filter(v => v.ok).length;
    const failed = data.videos.length - ok;
    let sub = ok + ' of ' + data.videos.length + ' videos have captions'
      + (failed > 0 ? ' \u00b7 ' + failed + ' missing' : '');
    if (data.truncated) {
      sub += ' \u00b7 \u26A0 ' + data.videos.length + ' videos exceeds free-tier limit (~20); later videos may have failed';
    }
    playlistSub.textContent = sub;

    const withTimestamps = tsTogglePlaylist.checked;
    videoList.innerHTML = '';
    data.videos.forEach((v, i) => {
      const card = document.createElement('div');
      card.className = 'video-card collapsed' + (v.ok ? '' : ' failed');

      const header = document.createElement('div');
      header.className = 'video-card-header';

      const numEl = document.createElement('span');
      numEl.className = 'video-num';
      numEl.textContent = (i + 1) + '.';

      const titleArea = document.createElement('div');
      titleArea.className = 'video-title-area';
      titleArea.title = 'Click to toggle transcript';
      const titleEl = document.createElement('div');
      titleEl.className = 'video-title';
      titleEl.textContent = v.title || v.videoId;
      const subEl = document.createElement('div');
      subEl.className = 'video-sub';
      subEl.textContent = v.ok
        ? (v.videoId + ' \u00b7 ' + v.segments.length + ' segments \u00b7 ' + (v.language || '?') + ' (' + v.kind + ')')
        : (v.videoId + ' \u00b7 no captions');
      titleArea.appendChild(titleEl);
      titleArea.appendChild(subEl);
      // Only the title area toggles — buttons get their own clicks
      titleArea.addEventListener('click', () => card.classList.toggle('collapsed'));

      const actions = document.createElement('div');
      actions.className = 'video-card-actions';

      if (v.ok) {
        const copy = document.createElement('button');
        copy.innerHTML = '<span>\u29C9</span> Copy';
        copy.title = 'Copy transcript to clipboard';
        copy.addEventListener('click', async (e) => {
          e.stopPropagation();
          const ok = await copyText(renderSegments(v.segments, tsTogglePlaylist.checked));
          showToast(ok ? 'Copied "' + (v.title || v.videoId) + '"' : 'Copy failed');
        });

        const dl = document.createElement('button');
        dl.innerHTML = '<span>\u2193</span> Download';
        dl.title = 'Download transcript as .txt';
        dl.addEventListener('click', (e) => {
          e.stopPropagation();
          const fname = sanitizeFilename(v.title || v.videoId) + '.txt';
          downloadText(fname, renderSegments(v.segments, tsTogglePlaylist.checked));
          showToast('Downloaded ' + fname);
        });

        const open = document.createElement('button');
        open.innerHTML = '\u2197';
        open.title = 'Open on YouTube';
        open.addEventListener('click', (e) => {
          e.stopPropagation();
          window.open('https://youtu.be/' + v.videoId, '_blank');
        });

        actions.appendChild(copy);
        actions.appendChild(dl);
        actions.appendChild(open);
      }

      const chevronBtn = document.createElement('button');
      chevronBtn.className = 'chevron-btn';
      chevronBtn.innerHTML = '<span class="chevron">\u25BE</span>';
      chevronBtn.title = 'Toggle transcript';
      chevronBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        card.classList.toggle('collapsed');
      });
      actions.appendChild(chevronBtn);

      header.appendChild(numEl);
      header.appendChild(titleArea);
      header.appendChild(actions);

      const body = document.createElement('div');
      body.className = 'video-card-body';

      if (v.ok) {
        const transcript = document.createElement('div');
        transcript.className = 'video-transcript';
        transcript.textContent = renderSegments(v.segments, withTimestamps);
        body.appendChild(transcript);
      } else {
        const err = document.createElement('div');
        err.className = 'video-error';
        err.textContent = v.error;
        body.appendChild(err);
      }

      card.appendChild(header);
      card.appendChild(body);
      videoList.appendChild(card);
    });
  }

  // --- Fetch ---

  fetchBtn.addEventListener('click', async () => {
    const parsed = parseInput(urlInput.value);
    if (!parsed) { setStatus('Could not parse a video or playlist from that input.', 'error'); return; }

    fetchBtn.disabled = true;
    singleView.classList.add('hidden');
    playlistView.classList.add('hidden');
    current = null;

    try {
      if (parsed.kind === 'video') {
        setStatus('Fetching video transcript...');
        const res = await fetch('/api/transcript?v=' + encodeURIComponent(parsed.id));
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Request failed');
        showSingle(data);
        setStatus('Loaded ' + data.segments.length + ' segments.', 'success');
      } else {
        setStatus('Fetching playlist (parallel, this is fast)...');
        const t0 = Date.now();
        const res = await fetch('/api/playlist?list=' + encodeURIComponent(parsed.id));
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Request failed');
        const ok = data.videos.filter(v => v.ok).length;
        const failed = data.videos.length - ok;
        const failMsg = failed > 0 ? ' (' + failed + ' without captions)' : '';
        const secs = ((Date.now() - t0) / 1000).toFixed(1);
        showPlaylist(data);
        setStatus('Loaded ' + ok + ' of ' + data.videos.length + ' transcripts' + failMsg + ' in ' + secs + 's.', 'success');
      }
    } catch (e) {
      setStatus(e.message, 'error');
    } finally {
      fetchBtn.disabled = false;
    }
  });

  // Single video controls
  copyBtn.addEventListener('click', async () => {
    if (!output.value) { showToast('Nothing to copy'); return; }
    const ok = await copyText(output.value);
    showToast(ok ? 'Copied to clipboard' : 'Copy failed');
  });

  downloadBtn.addEventListener('click', () => {
    if (!output.value || !current || current.kind !== 'video') { showToast('Nothing to download'); return; }
    const fname = sanitizeFilename(current.data.title || current.data.videoId) + '.txt';
    downloadText(fname, output.value);
    showToast('Downloaded ' + fname);
  });

  tsToggle.addEventListener('change', renderSingle);

  // Playlist controls
  copyAllBtn.addEventListener('click', async () => {
    if (!current || current.kind !== 'playlist') return;
    const ok = await copyText(buildPlaylistText(tsTogglePlaylist.checked));
    showToast(ok ? 'Copied entire playlist' : 'Copy failed');
  });

  downloadAllBtn.addEventListener('click', () => {
    if (!current || current.kind !== 'playlist') return;
    const fname = sanitizeFilename(current.data.playlistTitle || current.data.playlistId) + '.txt';
    downloadText(fname, buildPlaylistText(tsTogglePlaylist.checked));
    showToast('Downloaded ' + fname);
  });

  tsTogglePlaylist.addEventListener('change', renderPlaylistVideoBodies);

  urlInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') fetchBtn.click(); });
})();
</script>
</body>
</html>`;
