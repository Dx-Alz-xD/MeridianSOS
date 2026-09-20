/* ============================================================
   MeridianSOS — Vercel serverless port of meridian-sos-local/server.js

   The local server is a single long-lived Node process with two things
   Vercel serverless cannot provide:

     a writable filesystem   it persists to meridian-data.json on every
                             mutation. On Vercel the filesystem is read-only
                             apart from /tmp, and /tmp is neither shared
                             between invocations nor durable, so every
                             check-in and message would silently vanish.

     in-process broadcast    /api/stream is Server-Sent Events pushed from a
                             Set of open responses. Serverless invocations do
                             not share memory, so two users would land on
                             different instances and never see each other.

   This port replaces both:

     store   -> Vercel Blob, one JSON document (state.json), read-modify-write
     stream  -> GET /api/poll?since=<rev>, long-poll returning the state when
                the revision advances

   Everything else is the same routing and the same shapes, deliberately, so
   the two implementations stay comparable. All handlers live in ONE catch-all
   function rather than nine files: nine copies of the store helpers is how
   the local and deployed versions drift apart.

   CONCURRENCY, stated plainly: a mutation is read-modify-write on a single
   blob with no compare-and-swap. Two writes landing inside the same ~200 ms
   window can lose one of them. That is acceptable for a demo with a handful
   of users and is NOT acceptable for real dispatch. Fixing it properly means
   a store with atomic operations (Redis/Postgres); see HANDOFF.md.

   WITHOUT a Blob token the whole thing still serves: it reports
   storage:"none" and the client drops to a local-only mode, so the page is
   never broken, just single-user.
   ============================================================ */
"use strict";

const STATE_KEY = "meridian/state.json";
const PRESENCE_TTL_MS = 25000;
const POLL_TIMEOUT_MS = 20000;
const POLL_INTERVAL_MS = 700;

const ADMIN_ONLY_PREFIXES = ["meta/", "alerts/", "shelters/", "members/"];
const ADMIN_CODE = process.env.MERIDIAN_ADMIN_CODE || "742519";

let blob = null;
function blobApi() {
  if (blob === null) {
    try { blob = require("@vercel/blob"); } catch (e) { blob = false; }
  }
  return blob || null;
}
const hasStorage = () => !!process.env.BLOB_READ_WRITE_TOKEN && !!blobApi();

/* ---------------- seed ---------------- */
let SEED = null;
function seed() {
  if (SEED) return SEED;
  try { SEED = require("./seed.json"); } catch (e) { SEED = {}; }
  return SEED;
}
function freshState() {
  return { store: JSON.parse(JSON.stringify(seed())), people: {},
           presence: {}, rev: 1, updatedAt: Date.now() };
}

/* ---------------- state in Blob ----------------
   A short in-process cache makes a burst of reads on one warm instance cheap
   without making writes stale: every mutation refetches first.            */
let cache = null, cacheAt = 0;
const CACHE_MS = 400;

async function readState(force) {
  if (!hasStorage()) {
    // No blob token: fall back to per-instance memory. This must still read
    // back its own writes, otherwise every mutation is silently dropped and
    // even claiming admin does not stick. It is NOT shared between
    // invocations, which is exactly why the client shows the single-user
    // banner -- but within a warm instance the API behaves correctly, and
    // `vercel dev` works without any storage configured.
    if (!cache) cache = freshState();
    return cache;
  }
  if (!force && cache && Date.now() - cacheAt < CACHE_MS) return cache;
  const { list } = blobApi();
  try {
    const { blobs } = await list({ prefix: STATE_KEY, limit: 1 });
    if (!blobs.length) {
      const s = freshState();
      await writeState(s);
      return s;
    }
    // cache-bust: the blob CDN will happily serve a stale copy otherwise
    const r = await fetch(blobs[0].url + "?t=" + Date.now(), { cache: "no-store" });
    if (!r.ok) throw new Error("blob fetch " + r.status);
    const s = await r.json();
    cache = s; cacheAt = Date.now();
    return s;
  } catch (e) {
    const s = freshState();
    s.error = String(e && e.message || e);
    return s;
  }
}

async function writeState(s) {
  s.rev = (s.rev || 0) + 1;
  s.updatedAt = Date.now();
  cache = s; cacheAt = Date.now();
  if (!hasStorage()) return s;
  const { put } = blobApi();
  await put(STATE_KEY, JSON.stringify(s), {
    access: "public", contentType: "application/json",
    addRandomSuffix: false, allowOverwrite: true, cacheControlMaxAge: 0,
  });
  return s;
}

/* ---------------- helpers ---------------- */
function prune(s) {
  const cut = Date.now() - PRESENCE_TTL_MS;
  let changed = false;
  for (const id of Object.keys(s.presence || {})) {
    if ((s.presence[id].updatedAt || 0) < cut) { delete s.presence[id]; changed = true; }
  }
  return changed;
}
const presenceList = (s) => Object.keys(s.presence || {}).map(id => ({
  id, presence: s.presence[id].presence, updatedAt: s.presence[id].updatedAt }));

const isAdmin = (s, id) => { const m = s.store["members/" + id]; return !!(m && m.admin); };
const requiresAdmin = (p) => ADMIN_ONLY_PREFIXES.some(pre => p.startsWith(pre));
const merge = (base, patch) => Object.assign({}, base || {}, patch);

function send(res, code, obj) {
  res.setHeader("Content-Type", "application/json");
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Cache-Control", "no-store");
  res.status(code).end(JSON.stringify(obj));
}
async function body(req) {
  if (req.body && typeof req.body === "object") return req.body;
  if (typeof req.body === "string") { try { return JSON.parse(req.body); } catch (e) { return {}; } }
  const chunks = [];
  for await (const c of req) chunks.push(c);
  if (!chunks.length) return {};
  try { return JSON.parse(Buffer.concat(chunks).toString("utf8")); } catch (e) { return {}; }
}
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

/* ---------------- handler ---------------- */
module.exports = async function handler(req, res) {
  const u = new URL(req.url, "http://x");
  const seg = (u.pathname.replace(/^\/api\/?/, "").split("/").filter(Boolean));
  const route = seg.join("/");
  const q = u.searchParams;
  const id = String(req.headers["x-device-id"] || q.get("id") || "").slice(0, 80);

  if (req.method === "OPTIONS") {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type,x-device-id");
    return res.status(204).end();
  }

  try {
    /* ---- info ---- */
    if (route === "info") {
      return send(res, 200, {
        storage: hasStorage() ? "vercel-blob" : "none",
        mode: hasStorage() ? "shared" : "local-only",
        transport: "poll",
        adminCodeHint: hasStorage()
          ? "set MERIDIAN_ADMIN_CODE in the Vercel project"
          : "storage not connected; running single-user",
        urls: [],
      });
    }

    /* ---- poll: the SSE replacement ----
       Long-polls up to POLL_TIMEOUT_MS. Returns as soon as the revision
       moves, so a message lands on other clients in well under a second in
       the common case, and the request costs nothing while it waits.      */
    if (route === "poll") {
      const since = Number(q.get("since") || 0);
      const deadline = Date.now() + POLL_TIMEOUT_MS;
      let s = await readState(true);
      while (s.rev <= since && Date.now() < deadline) {
        await sleep(POLL_INTERVAL_MS);
        s = await readState(true);
      }
      if (prune(s) && hasStorage()) await writeState(s);
      return send(res, 200, {
        rev: s.rev, changed: s.rev > since, store: s.store,
        people: s.people, peers: presenceList(s),
        storage: hasStorage() ? "vercel-blob" : "none",
      });
    }

    /* ---- identity ---- */
    if (route === "identity" && req.method === "POST") {
      const b = await body(req);
      if (!b.id) return send(res, 400, { code: "bad-request" });
      const s = await readState(true);
      s.people[b.id] = { name: String(b.name || "").slice(0, 60),
                         color: String(b.color || "").slice(0, 20) };
      await writeState(s);
      return send(res, 200, { ok: true });
    }

    /* ---- people ---- */
    if (route === "people" && req.method === "GET") {
      const s = await readState();
      const ids = String(q.get("ids") || "").split(",").filter(Boolean);
      const out = {};
      ids.forEach(i => { if (s.people[i]) out[i] = s.people[i]; });
      return send(res, 200, out);
    }
    if (route === "people/search" && req.method === "GET") {
      const s = await readState();
      const needle = String(q.get("q") || "").toLowerCase().trim();
      const rows = [];
      if (needle) {
        for (const i of Object.keys(s.people)) {
          const p = s.people[i];
          if ((p.name || "").toLowerCase().includes(needle)) rows.push({ id: i, ...p });
        }
      }
      return send(res, 200, rows.slice(0, 40));
    }

    /* ---- admin claim ---- */
    if (route === "claim-admin" && req.method === "POST") {
      const b = await body(req);
      if (String(b.code || "") !== ADMIN_CODE) return send(res, 403, { ok: false });
      const s = await readState(true);
      s.store["members/" + b.id] = merge(s.store["members/" + b.id], { admin: true });
      await writeState(s);
      return send(res, 200, { ok: true });
    }

    /* ---- presence ---- */
    if (route === "presence" && req.method === "POST") {
      const b = await body(req);
      if (!b.id) return send(res, 400, { code: "bad-request" });
      const s = await readState(true);
      s.presence[b.id] = { presence: b.patch || {}, updatedAt: Date.now() };
      prune(s);
      await writeState(s);
      return send(res, 200, { ok: true, peers: presenceList(s) });
    }

    /* ---- emit (ephemeral fan-out; persisted so pollers see it) ---- */
    if (route === "emit" && req.method === "POST") {
      const b = await body(req);
      const s = await readState(true);
      s.store["events/" + Date.now() + "-" + Math.random().toString(36).slice(2, 8)] =
        { from: b.id, topic: b.topic, data: b.data, at: Date.now() };
      const keys = Object.keys(s.store).filter(k => k.startsWith("events/")).sort();
      while (keys.length > 60) delete s.store[keys.shift()];   // keep it bounded
      await writeState(s);
      return send(res, 200, { ok: true });
    }

    /* ---- doc ---- */
    if (route === "doc" && req.method === "GET") {
      const s = await readState();
      const p = String(q.get("path") || "");
      if (p.startsWith("addresses/") && !isAdmin(s, id))
        return send(res, 403, { code: "forbidden" });
      return send(res, 200, { path: p, data: s.store[p] || null });
    }
    if (route === "doc" && (req.method === "POST" || req.method === "DELETE")) {
      const s = await readState(true);
      const b = req.method === "POST" ? await body(req) : {};
      const p = String(req.method === "POST" ? (b.path || "") : (q.get("path") || ""));
      if (!p) return send(res, 400, { code: "bad-request" });
      if ((requiresAdmin(p) || p.startsWith("addresses/")) && !isAdmin(s, id))
        return send(res, 403, { code: "forbidden" });
      if (req.method === "DELETE") delete s.store[p];
      else s.store[p] = b.merge ? merge(s.store[p], b.data) : b.data;
      await writeState(s);
      return send(res, 200, { ok: true, rev: s.rev });
    }

    /* ---- collection ---- */
    if (route === "collection" && req.method === "GET") {
      const s = await readState();
      const prefix = String(q.get("path") || "").replace(/\/?$/, "/");
      if (prefix.startsWith("addresses/") && !isAdmin(s, id))
        return send(res, 403, { code: "forbidden" });
      const rows = Object.keys(s.store)
        .filter(k => k.startsWith(prefix) && k.slice(prefix.length).indexOf("/") === -1)
        .map(k => ({ id: k.slice(prefix.length), ...s.store[k] }));
      return send(res, 200, rows);
    }

    /* ---- stream: gone on purpose, tell the client why ---- */
    if (route === "stream") {
      return send(res, 410, {
        code: "gone",
        message: "Server-Sent Events do not work across serverless instances. Use GET /api/poll?since=<rev>.",
      });
    }

    return send(res, 404, { code: "not-found", route });
  } catch (e) {
    return send(res, 500, { code: "unavailable", message: String(e && e.message || e) });
  }
};
