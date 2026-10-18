// Minimal static server with the headers wasm threads need.
// SharedArrayBuffer (and therefore pthreads) only works when the page is
// cross-origin isolated, which requires COOP + COEP on every response.
//
//   node web/serve.mjs [directory] [port] [--game <game dir>]
//   node web/serve.mjs build/web-release 8080 --game game   (its runtime/ folder is served)
//
// With --game, the local game folder (files/ and scan/) is served under /game/ with
// HTTP range support, plus /game/manifest.txt listing every file. The page then streams
// game data on demand (open it with ?stream) instead of importing ~7 GB into OPFS.
// Development only: it serves to 127.0.0.1 and nothing leaves the machine.

import { createServer } from 'node:http';
import { createReadStream, existsSync } from 'node:fs';
import { readdir, stat } from 'node:fs/promises';
import { extname, join, normalize, relative, resolve, sep } from 'node:path';

const args = process.argv.slice(2);
const gameIndex = args.indexOf('--game');
const gameRoot = gameIndex >= 0 ? resolve(args.splice(gameIndex, 2)[1]) : null;
// A build folder (build/web-release) holds the page in its runtime/ subfolder.
let root = resolve(args[0] ?? 'web/public');
if (!existsSync(join(root, 'index.html')) && existsSync(join(root, 'runtime', 'index.html')))
  root = join(root, 'runtime');
if (!existsSync(join(root, 'index.html')))
  console.warn(`Warning: no index.html in ${root}. Build the web target first (scripts\\build-web.ps1).`);
const port = Number(args[1] ?? 8080);

const types = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.wasm': 'application/wasm',
  '.json': 'application/json',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.txt': 'text/plain; charset=utf-8',
};

async function listFiles(dir) {
  const out = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) out.push(...await listFiles(full));
    else if (entry.isFile()) out.push(full);
  }
  return out;
}

let manifest = null;
async function gameManifest() {
  if (!manifest) {
    const files = [];
    for (const sub of ['files', 'scan']) {
      try {
        files.push(...await listFiles(join(gameRoot, sub)));
      } catch { /* missing folder */ }
    }
    manifest = files.map((f) => relative(gameRoot, f).split(sep).join('/')).join('\n') + '\n';
  }
  return manifest;
}

// Serves one file, honouring HEAD and a single "bytes=a-b" range.
async function sendFile(req, res, filePath) {
  const info = await stat(filePath);
  if (!info.isFile()) throw new Error('not a file');
  const headers = {
    'Content-Type': types[extname(filePath).toLowerCase()] ?? 'application/octet-stream',
    'Accept-Ranges': 'bytes',
  };
  const range = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range ?? '');
  if (range && req.method !== 'HEAD' && (range[1] || range[2])) {
    let start = range[1] ? Number(range[1]) : info.size - Number(range[2]);
    let end = range[1] && range[2] ? Number(range[2]) : info.size - 1;
    end = Math.min(end, info.size - 1);
    if (start > end || start >= info.size) {
      res.writeHead(416, { 'Content-Range': `bytes */${info.size}` }).end();
      return;
    }
    res.writeHead(206, { ...headers, 'Content-Range': `bytes ${start}-${end}/${info.size}`, 'Content-Length': end - start + 1 });
    createReadStream(filePath, { start, end }).pipe(res);
    return;
  }
  res.writeHead(200, { ...headers, 'Content-Length': info.size });
  if (req.method === 'HEAD') res.end();
  else createReadStream(filePath).pipe(res);
}

function inside(base, path) {
  return path === base || path.startsWith(base + sep);
}

createServer(async (req, res) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
  res.setHeader('Cache-Control', 'no-cache');

  const urlPath = decodeURIComponent(new URL(req.url, 'http://localhost').pathname).replace(/\/{2,}/g, '/');
  try {
    if (gameRoot && (urlPath === '/game' || urlPath.startsWith('/game/'))) {
      const rest = urlPath.slice('/game'.length).replace(/^\//, '');
      if (rest === 'manifest.txt') {
        const body = await gameManifest();
        res.writeHead(200, { 'Content-Type': types['.txt'], 'Content-Length': Buffer.byteLength(body), 'Accept-Ranges': 'bytes' });
        res.end(req.method === 'HEAD' ? undefined : body);
        return;
      }
      const filePath = normalize(join(gameRoot, rest));
      if (!inside(gameRoot, filePath)) throw new Error('outside');
      await sendFile(req, res, filePath);
      return;
    }
    const filePath = normalize(join(root, urlPath.endsWith('/') ? urlPath + 'index.html' : urlPath));
    if (!inside(root, filePath)) {
      res.writeHead(403).end();
      return;
    }
    await sendFile(req, res, filePath);
  } catch {
    if (!res.headersSent) res.writeHead(404, { 'Content-Type': 'text/plain' }).end('Not found');
  }
}).listen(port, '127.0.0.1', () => {
  console.log(`Serving ${root} at http://localhost:${port} (cross-origin isolated)`);
  if (gameRoot) console.log(`Streaming game data from ${gameRoot}: open http://localhost:${port}/?stream`);
});
