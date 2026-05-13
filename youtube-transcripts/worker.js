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

// InnerTube clients to try, in order. Order matters: we put clients that tolerate
// anonymous datacenter IPs (like Cloudflare Worker egress) first.
// - TVHTML5_SIMPLY_EMBEDDED_PLAYER: designed for TV apps that fetch unauthenticated.
//   Most reliable for bypassing LOGIN_REQUIRED from datacenter IPs.
// - IOS: signed differently from WEB/ANDROID; sometimes succeeds when others fail.
// - ANDROID: traditionally laxest rate-limiting, but increasingly hit by bot detection.
// - WEB: last resort. Most likely to demand PoToken on datacenter IPs.
//
// API keys are hardcoded into the official YouTube apps — they're not secrets, but
// YouTube could rotate them. Refresh via:
//   curl -A 'Mozilla/5.0' https://www.youtube.com/watch?v=dQw4w9WgXcQ | grep -o 'INNERTUBE_API_KEY":"[^"]*'
const INNERTUBE_CLIENTS = [
  {
    name: 'TVHTML5_EMBED',
    apiKey: 'AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8',
    context: {
      client: {
        clientName: 'TVHTML5_SIMPLY_EMBEDDED_PLAYER',
        clientVersion: '2.0',
        clientScreen: 'EMBED',
      },
      thirdParty: { embedUrl: 'https://www.youtube.com/' },
    },
    userAgent: 'Mozilla/5.0 (PlayStation; PlayStation 4/12.00) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0',
  },
  {
    name: 'IOS',
    apiKey: 'AIzaSyB-63vPrdThhKuerbB2N_l7Kwwcxj6yUAc',
    context: { client: { clientName: 'IOS', clientVersion: '20.10.4', deviceModel: 'iPhone16,2' } },
    userAgent: 'com.google.ios.youtube/20.10.4 (iPhone16,2; U; CPU iOS 18_3_2 like Mac OS X;)',
  },
  {
    name: 'ANDROID',
    apiKey: 'AIzaSyA8eiZmM1FaDVjRy-df2KTyQ_vz_yYM39w',
    context: { client: { clientName: 'ANDROID', clientVersion: '20.10.38', androidSdkVersion: 30 } },
    userAgent: 'com.google.android.youtube/20.10.38 (Linux; U; Android 11) gzip',
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
      return serveHTML(request, env);
    }
    if (url.pathname === '/api/transcript') {
      return handleTranscript(url.searchParams.get('v'), request, env, ctx);
    }
    if (url.pathname === '/api/playlist') {
      return handlePlaylist(url.searchParams.get('list'), request, env, ctx);
    }
    if (url.pathname === '/api/credits') {
      return handleCredits(env);
    }
    return new Response('Not found', { status: 404 });
  },
};

// Proxy to Supadata's /me endpoint. Returns null fields if no API key is configured
// (development mode), letting the UI hide the credits display.
async function handleCredits(env) {
  if (!env || !env.SUPADATA_API_KEY) {
    return json({ provider: 'innertube' });
  }
  try {
    await paceSupadata();  // Share the rate-limit gate with transcript calls
    const res = await fetch('https://api.supadata.ai/v1/me', {
      headers: { 'x-api-key': env.SUPADATA_API_KEY },
    });
    if (!res.ok) {
      return json({ provider: 'supadata', error: `HTTP ${res.status}` });
    }
    const data = await res.json();
    return json({
      provider: 'supadata',
      plan: data.plan,
      used: data.usedCredits,
      max: data.maxCredits,
    });
  } catch (e) {
    return json({ provider: 'supadata', error: e.message });
  }
}

// Pick the provider based on whether the Supadata API key is configured.
// - Production (deployed Worker with secret set): Supadata, which has residential
//   IPs that YouTube doesn't bot-block.
// - Local development (no secret set): direct InnerTube calls, which work from
//   home IPs and cost nothing.
function getProvider(env) {
  if (env && env.SUPADATA_API_KEY) {
    return {
      name: 'supadata',
      fetchTranscriptForVideo: (id) => supadataTranscript(id, env.SUPADATA_API_KEY),
      getPlaylistVideos: (id) => supadataPlaylist(id, env.SUPADATA_API_KEY),
    };
  }
  return {
    name: 'innertube',
    fetchTranscriptForVideo: innertubeTranscript,
    getPlaylistVideos: innertubePlaylist,
  };
}

// --- KV cache wrapper ---
// Cache key conventions:
//   transcript:<videoId>      — never expires (transcripts are immutable)
//   playlist:<playlistId>     — 24h TTL (playlists change as videos are added)
//
// If env.CACHE is not bound (e.g. local dev without KV namespace), this is
// a transparent no-op: every request goes straight to the provider.

async function cachedOrFetch(env, ctx, key, ttlSeconds, fetchFn) {
  const kv = env && env.CACHE;
  if (kv) {
    try {
      const hit = await kv.get(key, 'json');
      if (hit) {
        console.log(`[cache] HIT ${key}`);
        return { data: hit, cached: true };
      }
      console.log(`[cache] MISS ${key}`);
    } catch (e) {
      console.log(`[cache] read failed for ${key}: ${e.message}`);
    }
  }

  const data = await fetchFn();

  if (kv && ctx) {
    // Register the write with waitUntil so the Worker doesn't terminate
    // before the put completes. Without this, the write often gets cancelled
    // after the response is sent and the cache stays empty.
    const writeOptions = ttlSeconds ? { expirationTtl: ttlSeconds } : {};
    ctx.waitUntil(
      kv.put(key, JSON.stringify(data), writeOptions)
        .then(() => console.log(`[cache] wrote ${key}`))
        .catch((e) => console.log(`[cache] write failed for ${key}: ${e.message}`))
    );
  }

  return { data, cached: false };
}

// --- API: single video transcript ---

async function handleTranscript(videoId, request, env, ctx) {
  if (!videoId || !/^[a-zA-Z0-9_-]{11}$/.test(videoId)) {
    return json({ error: 'Invalid or missing video ID' }, 400);
  }
  const provider = getProvider(env);
  const user = request.headers.get('Cf-Access-Authenticated-User-Email');
  if (user) console.log(`Transcript [${provider.name}]: ${videoId} by ${user}`);

  try {
    const { data, cached } = await cachedOrFetch(
      env,
      ctx,
      `transcript:${videoId}`,
      null,  // no TTL — transcripts are immutable
      () => provider.fetchTranscriptForVideo(videoId)
    );
    return json({ ...data, cached });
  } catch (e) {
    return json({ error: e.message }, 500);
  }
}

async function innertubeTranscript(videoId) {
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

async function handlePlaylist(playlistId, request, env, ctx) {
  if (!playlistId || !/^[a-zA-Z0-9_-]{10,}$/.test(playlistId)) {
    return json({ error: 'Invalid or missing playlist ID' }, 400);
  }
  const provider = getProvider(env);
  const user = request.headers.get('Cf-Access-Authenticated-User-Email');
  if (user) console.log(`Playlist [${provider.name}]: ${playlistId} by ${user}`);

  try {
    // Cache the playlist enumeration for 24h — playlists change as videos are added/removed.
    const { data: playlistData, cached: playlistCached } = await cachedOrFetch(
      env,
      ctx,
      `playlist:${playlistId}`,
      86400,  // 24h TTL
      () => provider.getPlaylistVideos(playlistId)
    );
    const { title: playlistTitle, videos } = playlistData;

    if (!videos.length) {
      return json({ error: 'Playlist is empty or unavailable' }, 404);
    }

    // Subrequest budget concerns differ by provider:
    // - InnerTube: Workers free tier allows 50 subrequests. Each video uses 2-8.
    //   Conservative threshold: ~15.
    // - Supadata: 1 outbound subrequest per video, no client-retry chains.
    //   Workers limit allows ~45 videos; Supadata's free tier is the harder cap.
    // Caching reduces real cost: only fresh videos hit the provider.
    const limit = provider.name === 'supadata' ? 45 : 15;
    const truncated = videos.length > limit;
    const toProcess = videos;

    // Concurrency depends on the provider:
    // - Supadata free tier rate-limits concurrent requests, so process one at a time.
    // - InnerTube has no such constraint; parallel is much faster.
    // For both, the cache short-circuits already-fetched videos at near-zero cost.
    const concurrency = provider.name === 'supadata' ? 1 : PLAYLIST_CONCURRENCY;

    let cacheHits = 0;
    const results = await parallelMap(toProcess, concurrency, async (video) => {
      try {
        const { data, cached } = await cachedOrFetch(
          env,
          ctx,
          `transcript:${video.videoId}`,
          null,
          () => provider.fetchTranscriptForVideo(video.videoId)
        );
        if (cached) cacheHits++;
        // Preserve playlist title if the provider didn't return one
        return { ok: true, ...data, title: data.title || video.title };
      } catch (e) {
        return {
          ok: false,
          videoId: video.videoId,
          title: video.title,
          error: e.message,
        };
      }
    });

    console.log(`Playlist ${playlistId}: ${cacheHits}/${results.length} from cache (playlist enum: ${playlistCached ? 'cached' : 'fresh'})`);

    return json({
      playlistId,
      playlistTitle,
      count: results.length,
      truncated,
      videos: results,
      provider: provider.name,
      cacheHits,
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

async function innertubePlaylist(playlistId) {
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

// --- Provider: Supadata ---
// Used in production where Cloudflare Worker IPs get LOGIN_REQUIRED from YouTube.
// Supadata fetches from residential IPs and returns clean JSON.
// Pricing: 1 credit per transcript, 100 free per month, paid plans scale up.
// Free tier has a strict rate limit (~1 req/sec); all Supadata calls go through
// sequential paths only, with retry-on-429 below.
// https://docs.supadata.ai

const SUPADATA_BASE = 'https://api.supadata.ai/v1';

// Sleep helper for backoff delays
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Free tier has a strict rate limit of 1 request per second. We enforce this
// deterministically by tracking when the next call is allowed to fire and
// reserving slots atomically before awaiting. This is module-scoped, so within
// a single Worker isolate (which handles one request at a time per async task
// but can serve many concurrent requests via await interleaving), all Supadata
// calls are properly paced relative to each other. Across separate isolates
// (e.g. users in different regions) pacing is best-effort — the retry-on-429
// below catches any leakage.
let supadataNextCallAt = 0;
const SUPADATA_MIN_INTERVAL_MS = 1100;  // 1s limit + 100ms safety margin

async function paceSupadata() {
  // Reserve our slot BEFORE awaiting. By computing and writing supadataNextCallAt
  // synchronously, two concurrent callers in the same isolate will see different
  // reserved times: the first reserves now+1.1s, the second sees that and reserves
  // now+2.2s, etc. Each then awaits until its own reserved time.
  const now = Date.now();
  const fireAt = Math.max(now, supadataNextCallAt);
  supadataNextCallAt = fireAt + SUPADATA_MIN_INTERVAL_MS;

  const waitMs = fireAt - now;
  if (waitMs > 0) {
    console.log(`[supadata] pacing: waiting ${waitMs}ms`);
    await sleep(waitMs);
  }
}

async function supadataRequest(path, apiKey, retries = 3) {
  let lastErr;
  for (let attempt = 0; attempt < retries; attempt++) {
    await paceSupadata();

    const res = await fetch(SUPADATA_BASE + path, {
      headers: { 'x-api-key': apiKey },
    });

    if (res.ok) return res.json();

    // Capture body for diagnosis
    let body = '';
    try { body = await res.text(); } catch (e) {}
    const detail = body ? ': ' + body.slice(0, 200) : '';

    // Retry on 429 (rate limit slipped through) and 5xx (transient). Don't retry on other 4xx.
    if (res.status === 429 || res.status >= 500) {
      const wait = 1000 * Math.pow(2, attempt) + Math.random() * 250;  // 1s, 2s, 4s + jitter
      console.log(`[supadata] ${path} returned ${res.status}, retrying in ${Math.round(wait)}ms (attempt ${attempt + 1}/${retries})`);
      lastErr = new Error(`Supadata ${path} returned ${res.status}${detail}`);
      await sleep(wait);
      continue;
    }

    throw new Error(`Supadata ${path} returned ${res.status}${detail}`);
  }
  throw lastErr;
}

async function supadataTranscript(videoId, apiKey) {
  console.log(`[supadata] transcript ${videoId}`);
  // Sequential calls — free tier rate-limits concurrent requests.
  const transcript = await supadataRequest(`/youtube/transcript?videoId=${videoId}&text=false`, apiKey);
  // Metadata is best-effort; if it fails (e.g. rate limit), we still return the transcript.
  let video = null;
  try {
    video = await supadataRequest(`/youtube/video?id=${videoId}`, apiKey);
  } catch (e) {
    console.log(`[supadata] metadata fetch failed for ${videoId}: ${e.message}`);
  }

  // Supadata uses `offset` (ms) and `duration` (ms); convert to our `start` (seconds).
  const segments = (transcript.content || []).map((s) => ({
    start: (s.offset || 0) / 1000,
    text: (s.text || '').replace(/\n/g, ' ').trim(),
  })).filter((s) => s.text);

  if (!segments.length) throw new Error('Supadata returned an empty transcript');

  return {
    videoId,
    title: video?.title || videoId,
    author: video?.channel?.name || '',
    language: transcript.lang || 'unknown',
    kind: 'unknown',  // Supadata doesn't distinguish auto vs manual
    segments,
  };
}

async function supadataPlaylist(playlistId, apiKey) {
  console.log(`[supadata] playlist ${playlistId}`);
  // Sequential — same rate-limit constraint as transcripts.
  // Fetch the playlist title first, then enumerate video IDs.
  let meta = null;
  try {
    meta = await supadataRequest(`/youtube/playlist?id=${playlistId}`, apiKey);
  } catch (e) {
    console.log(`[supadata] playlist metadata failed: ${e.message}`);
  }

  const vids = await supadataRequest(`/youtube/playlist/videos?id=${playlistId}&limit=100`, apiKey);
  const ids = vids?.videoIds || vids?.video_ids || [];
  const videos = ids.map((id) => ({ videoId: id, title: id }));

  return {
    title: meta?.title || playlistId,
    videos,
  };
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
  const requestId = (payload.videoId || payload.browseId || 'continuation').toString().slice(0, 20);
  console.log(`[${client.name}] ${path} → ${requestId}`);

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

  if (!res.ok) {
    // Log a short snippet of body for diagnosis (often HTML error page from YouTube edge)
    const snippet = (await res.text()).slice(0, 200).replace(/\s+/g, ' ');
    console.log(`[${client.name}] HTTP ${res.status}: ${snippet}`);
    throw new Error(`${client.name}: HTTP ${res.status}`);
  }

  const data = await res.json();

  // Did we even get the field we expect?
  const hasCaptions = !!data?.captions?.playerCaptionsTracklistRenderer?.captionTracks;
  const trackCount = data?.captions?.playerCaptionsTracklistRenderer?.captionTracks?.length || 0;
  const playabilityStatus = data?.playabilityStatus?.status;
  const reason = data?.playabilityStatus?.reason;
  const videoTitle = data?.videoDetails?.title;

  if (path.includes('/player')) {
    console.log(`[${client.name}] ${requestId} status=${playabilityStatus} title="${videoTitle?.slice(0, 40)}" captions=${hasCaptions} tracks=${trackCount}`);
  }

  if (playabilityStatus && playabilityStatus !== 'OK') {
    throw new Error(`${client.name}: ${reason || playabilityStatus}`);
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
  if (!res.ok) {
    console.log(`[captions] HTTP ${res.status} from ${new URL(captionUrl).pathname}`);
    throw new Error(`Caption fetch returned ${res.status}`);
  }
  const body = await res.text();
  const trimmed = body.trimStart();
  console.log(`[captions] ${body.length} bytes, starts with "${trimmed.slice(0, 20)}"`);

  // JSON3 path (preferred)
  if (trimmed.startsWith('{')) {
    const data = JSON.parse(trimmed);
    const out = [];
    for (const ev of data.events || []) {
      if (!ev.segs) continue;
      const text = ev.segs.map(s => s.utf8 || '').join('').replace(/\n/g, ' ').trim();
      if (text) out.push({ start: (ev.tStartMs || 0) / 1000, text });
    }
    console.log(`[captions] parsed ${out.length} JSON segments`);
    return out;
  }

  // XML fallback (srv1/srv3) — older format YouTube still returns sometimes
  if (trimmed.startsWith('<')) {
    const out = parseCaptionXml(trimmed);
    console.log(`[captions] parsed ${out.length} XML segments`);
    return out;
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

function serveHTML(request, env) {
  const user = request.headers.get('Cf-Access-Authenticated-User-Email') || '';
  const provider = getProvider(env).name;
  const html = HTML
    .replace('__USER_EMAIL__', escapeHtml(user))
    .replace('__PROVIDER__', escapeHtml(provider));
  return new Response(html, {
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
  .header-meta { display: flex; flex-direction: column; align-items: flex-end; gap: 2px; }
  .credits { font-size: 11px; color: var(--text-muted); font-family: ui-monospace, monospace; }
  .credits.low { color: var(--danger); }
  .credits:empty { display: none; }
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
    <div class="header-meta">
      <span class="credits" id="credits"></span>
      <span class="user">__USER_EMAIL__</span>
    </div>
  </header>
  <p class="lede">
    Paste a YouTube video URL, playlist URL, video ID, or playlist ID.
    <span class="lede-note" id="ledeNote" data-provider="__PROVIDER__"></span>
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
  // Provider-specific guidance shown below the input box
  const ledeNote = document.getElementById('ledeNote');
  const provider = ledeNote.dataset.provider;
  if (provider === 'supadata') {
    ledeNote.textContent = 'Production mode: fetching via Supadata. Free-tier rate limit is 1 transcript per second. Cached videos return instantly.';
  } else {
    ledeNote.textContent = 'Development mode: fetching YouTube directly. Playlists up to ~15 videos on free tier (may fail on Cloudflare IPs).';
  }

  // Credits display in the header — only meaningful when running against Supadata
  const creditsEl = document.getElementById('credits');
  async function refreshCredits() {
    try {
      const res = await fetch('/api/credits');
      const data = await res.json();
      if (data.provider !== 'supadata' || data.error) {
        creditsEl.textContent = '';
        return;
      }
      const remaining = data.max - data.used;
      creditsEl.textContent = remaining + ' of ' + data.max + ' credits left';
      creditsEl.title = data.used + ' used this month on ' + (data.plan || 'current') + ' plan';
      // Visually warn when we're under 10% of monthly allowance
      if (remaining < data.max * 0.1) creditsEl.classList.add('low');
      else creditsEl.classList.remove('low');
    } catch (e) {
      creditsEl.textContent = '';
    }
  }
  refreshCredits();

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
    const cacheLabel = data.cached ? ' \u00b7 \u26A1 cached' : '';
    meta.textContent = data.videoId + ' \u00b7 ' + data.language + ' (' + data.kind + ')' + cacheLabel;
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
    if (typeof data.cacheHits === 'number' && data.cacheHits > 0) {
      sub += ' \u00b7 \u26A1 ' + data.cacheHits + ' from cache';
    }
    if (data.truncated) {
      const cap = data.provider === 'supadata' ? '~45' : '~15';
      sub += ' \u00b7 \u26A0 ' + data.videos.length + ' videos exceeds limit (' + cap + '); later videos may have failed';
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
        // The provider mode is already in scope from the dataset attribute; reuse it
        // to set an honest expectation about timing.
        if (provider === 'supadata') {
          setStatus('Fetching playlist (1 transcript per second on free tier)...');
        } else {
          setStatus('Fetching playlist (parallel, this is fast)...');
        }
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
      // Refresh credits — a fetch may have consumed some
      refreshCredits();
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
