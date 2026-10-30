import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the wave tank simulator shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /<title>RPHYS Wave Tank Field Lab<\/title>/i);
  assert.match(html, /Wave Tank/);
  assert.match(html, /STEREO RIG/);
  assert.match(html, /Synthetic surface visualization/);
  assert.match(html, /aria-label="Pause simulation"/);
});

test("ships the CAD asset and scientific limitation copy", async () => {
  const [asset, scene, page, packageJson] = await Promise.all([
    access(new URL("../public/full-assembly.stl", import.meta.url)),
    readFile(new URL("../components/SimulationScene.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);
  assert.equal(asset, undefined);
  assert.match(scene, /STLLoader/);
  assert.match(scene, /full-assembly\.stl/);
  assert.match(scene, /updateRayGeometry/);
  assert.match(page, /not CFD/);
  assert.match(packageJson, /"three"/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
});
