#!/usr/bin/env node
/**
 * Ensure electron/dist/electron exists. Official install.js sometimes leaves an
 * empty dist/ (cache/extract glitches). Falls back to unzipping the release zip.
 */
const { execFileSync } = require("child_process");
const fs = require("fs");
const https = require("https");
const http = require("http");
const path = require("path");
const { pipeline } = require("stream/promises");

const electronDir = path.join(__dirname, "..", "node_modules", "electron");
const version = require(path.join(electronDir, "package.json")).version;
const distElectron = path.join(electronDir, "dist", "electron");
const pathTxt = path.join(electronDir, "path.txt");

function ok() {
  return fs.existsSync(distElectron) && fs.statSync(distElectron).size > 1_000_000;
}

function writeMeta() {
  fs.writeFileSync(pathTxt, "electron");
  fs.writeFileSync(path.join(electronDir, "dist", "version"), version);
}

async function download(url, dest) {
  const mod = url.startsWith("https") ? https : http;
  await new Promise((resolve, reject) => {
    const follow = (u, redirects = 0) => {
      mod.get(u, (res) => {
        if (
          res.statusCode >= 300 &&
          res.statusCode < 400 &&
          res.headers.location &&
          redirects < 5
        ) {
          follow(res.headers.location, redirects + 1);
          return;
        }
        if (res.statusCode !== 200) {
          reject(new Error(`HTTP ${res.statusCode} for ${u}`));
          return;
        }
        const out = fs.createWriteStream(dest);
        pipeline(res, out).then(resolve).catch(reject);
      }).on("error", reject);
    };
    follow(url);
  });
}

async function main() {
  if (!fs.existsSync(electronDir)) {
    console.warn("[ensure-electron] electron package missing; run npm install first");
    return;
  }

  try {
    execFileSync(process.execPath, [path.join(electronDir, "install.js")], {
      cwd: electronDir,
      stdio: "inherit",
      env: { ...process.env, force_no_cache: "true" },
    });
  } catch (_) {
    /* fall through to manual unzip */
  }

  if (ok()) {
    writeMeta();
    console.log("[ensure-electron] ok:", distElectron);
    return;
  }

  const zipUrl = `https://github.com/electron/electron/releases/download/v${version}/electron-v${version}-linux-x64.zip`;
  const zipPath = path.join(electronDir, "electron-download.zip");
  console.log("[ensure-electron] downloading", zipUrl);
  await download(zipUrl, zipPath);

  const dist = path.join(electronDir, "dist");
  fs.rmSync(dist, { recursive: true, force: true });
  fs.mkdirSync(dist, { recursive: true });
  execFileSync("unzip", ["-qo", zipPath, "-d", dist], { stdio: "inherit" });
  fs.rmSync(zipPath, { force: true });
  writeMeta();

  if (!ok()) {
    console.error("[ensure-electron] still missing binary after unzip");
    process.exit(1);
  }
  console.log("[ensure-electron] installed:", distElectron);
}

main().catch((err) => {
  console.error("[ensure-electron]", err);
  process.exit(1);
});
