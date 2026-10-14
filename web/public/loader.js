// Web shell: capability checks, importing the user's game files into OPFS,
// and launching the Emscripten module (ttt2.js / ttt2.wasm).
//
// OPFS layout (mounted at /game inside the runtime):
//   files/...            extracted disc contents
//   scan/image.bin       decrypted XEX image
//   scan/image_info.toml entry point etc.

const $ = (id) => document.getElementById(id);
const logEl = $('log');

function log(line) {
  logEl.textContent += line + '\n';
  logEl.scrollTop = logEl.scrollHeight;
}

window.ttt2Status = (text) => log(text);

// Smallest valid module declaring a 64-bit memory: validates only with memory64 support.
const MEMORY64_PROBE = new Uint8Array([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x05, 0x03, 0x01, 0x04, 0x00]);

async function runChecks() {
  const checks = [
    ['Cross-origin isolated (needed for threads)', self.crossOriginIsolated === true],
    ['SharedArrayBuffer', typeof SharedArrayBuffer !== 'undefined'],
    ['WebAssembly memory64', WebAssembly.validate(MEMORY64_PROBE)],
    ['WebGPU', !!navigator.gpu && !!(await navigator.gpu.requestAdapter().catch(() => null))],
    ['Origin Private File System', !!navigator.storage?.getDirectory],
  ];
  const list = $('check-list');
  list.replaceChildren(...checks.map(([label, ok]) => {
    const li = document.createElement('li');
    li.className = ok ? 'ok' : 'bad';
    li.textContent = label;
    return li;
  }));
  if (!checks[0][1]) {
    log('Not cross-origin isolated: serve with `node web/serve.mjs <build dir>` (sets COOP/COEP headers).');
  }
  return checks.every(([, ok]) => ok);
}

async function opfsRoot() {
  return navigator.storage.getDirectory();
}

async function fileExists(root, path) {
  try {
    let dir = root;
    const parts = path.split('/');
    for (const part of parts.slice(0, -1)) dir = await dir.getDirectoryHandle(part);
    await dir.getFileHandle(parts.at(-1));
    return true;
  } catch {
    return false;
  }
}

async function refreshImportStatus() {
  const root = await opfsRoot();
  const ready = await fileExists(root, 'scan/image.bin') && await fileExists(root, 'files/default.xex');
  $('import-status').textContent = ready
    ? 'Game data imported.'
    : 'No game data imported yet.';
  return ready;
}

// Recursively collects [relativePath, FileSystemFileHandle] under a picked directory.
async function* walk(dir, prefix = '') {
  for await (const [name, handle] of dir.entries()) {
    const path = prefix ? `${prefix}/${name}` : name;
    if (handle.kind === 'directory') yield* walk(handle, path);
    else yield [path, handle];
  }
}

async function importGameFolder() {
  if (!window.showDirectoryPicker) {
    alert('This browser cannot pick folders. Use a recent Chrome or Edge.');
    return;
  }
  const picked = await window.showDirectoryPicker({ mode: 'read' });
  const entries = [];
  for await (const entry of walk(picked)) {
    if (entry[0].startsWith('files/') || entry[0].startsWith('scan/')) entries.push(entry);
  }
  if (!entries.some(([p]) => p === 'scan/image.bin')) {
    alert('That folder has no scan/image.bin. Pick the repo\'s game folder after running scripts/scan-xex.ps1.');
    return;
  }

  const files = await Promise.all(entries.map(async ([path, handle]) => [path, await handle.getFile()]));
  const total = files.reduce((sum, [, f]) => sum + f.size, 0);
  const progress = $('progress');
  progress.hidden = false;
  progress.max = total;
  progress.value = 0;

  const root = await opfsRoot();
  let done = 0;
  for (const [path, file] of files) {
    const parts = path.split('/');
    let dir = root;
    for (const part of parts.slice(0, -1)) dir = await dir.getDirectoryHandle(part, { create: true });
    const out = await (await dir.getFileHandle(parts.at(-1), { create: true })).createWritable();
    await file.stream().pipeTo(out); // closes `out`
    done += file.size;
    progress.value = done;
    $('import-status').textContent = `Importing… ${(done / 2 ** 30).toFixed(2)} / ${(total / 2 ** 30).toFixed(2)} GB`;
  }
  progress.hidden = true;
  log(`Imported ${files.length} files.`);
  await refreshImportStatus();
}

async function clearImported() {
  if (!confirm('Delete all imported game data from this browser?')) return;
  const root = await opfsRoot();
  for await (const name of root.keys()) await root.removeEntry(name, { recursive: true });
  log('Imported data deleted.');
  await refreshImportStatus();
}

async function launch() {
  $('launch').disabled = true;
  const canvas = $('screen');
  try {
    const { default: createTTT2 } = await import('./ttt2.js');
    await createTTT2({
      canvas,
      print: (t) => log(t),
      printErr: (t) => log(t),
    });
    $('fullscreen').disabled = false;
  } catch (err) {
    log(`Launch failed: ${err?.message ?? err}`);
    $('launch').disabled = false;
  }
}

$('pick').addEventListener('click', () => importGameFolder().catch((e) => log(`Import failed: ${e.message}`)));
$('clear').addEventListener('click', () => clearImported().catch((e) => log(e.message)));
$('launch').addEventListener('click', launch);
$('fullscreen').addEventListener('click', () => $('screen').requestFullscreen());

const supported = await runChecks();
await refreshImportStatus().catch(() => {});
// The placeholder runtime runs without game data, so only browser support gates launching.
$('launch').disabled = !supported;
