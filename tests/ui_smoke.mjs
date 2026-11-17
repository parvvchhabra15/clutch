import assert from "node:assert/strict";
import {spawn} from "node:child_process";
import {existsSync} from "node:fs";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import {fileURLToPath, pathToFileURL} from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const candidates = [
  process.env.BROWSER_PATH,
  process.env["ProgramFiles(x86)"] && path.join(process.env["ProgramFiles(x86)"], "Microsoft", "Edge", "Application", "msedge.exe"),
  process.env.ProgramFiles && path.join(process.env.ProgramFiles, "Microsoft", "Edge", "Application", "msedge.exe"),
].filter(Boolean);
const executable = candidates.find(existsSync);
if (!executable) throw new Error("Set BROWSER_PATH to an installed Chromium or Edge executable.");
const profile = await fs.mkdtemp(path.join(os.tmpdir(), "clutch-ui-"));
const output = path.join(root, "build", "ui-review");
await fs.mkdir(output, {recursive: true});
const browser = spawn(executable, [
  "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
  "--disable-background-networking", "--disable-component-update",
  "--remote-debugging-port=0", `--user-data-dir=${profile}`, "about:blank",
], {stdio: ["ignore", "ignore", "pipe"], windowsHide: true});
let browserLog = "";
browser.stderr.on("data", chunk => { browserLog += chunk.toString(); });
let socket;
let session;
let nextId = 0;
const pending = new Map();
const errors = [];
const network = [];
const checks = [];
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function request(method, params = {}, targetSession = session) {
  const id = ++nextId;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error(`Timed out: ${method}`));
    }, 20000);
    pending.set(id, {resolve, reject, timer});
    socket.send(JSON.stringify({id, method, params, ...(targetSession ? {sessionId: targetSession} : {})}));
  });
}

async function evaluate(expression) {
  const result = await request("Runtime.evaluate", {expression, returnByValue: true, awaitPromise: true});
  if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}

async function waitFor(expression, timeout = 15000) {
  const until = Date.now() + timeout;
  while (Date.now() < until) {
    if (await evaluate(`Boolean(${expression})`)) return;
    await sleep(100);
  }
  throw new Error(`Page condition did not become true: ${expression}`);
}

async function screenshot(name) {
  const image = await request("Page.captureScreenshot", {format: "png", captureBeyondViewport: false});
  await fs.writeFile(path.join(output, name + ".png"), Buffer.from(image.data, "base64"));
}

async function viewport(width, height) {
  await request("Emulation.setDeviceMetricsOverride", {width, height, deviceScaleFactor: 1, mobile: false});
  await sleep(350);
}

async function checkFit(label) {
  const result = await evaluate(`({
    width: document.documentElement.clientWidth,
    scroll: document.documentElement.scrollWidth,
    pink: getComputedStyle(document.body).backgroundColor,
    chart: document.querySelector('#component-chart').getBoundingClientRect().width
  })`);
  if (result.scroll > result.width + 2) {
    const overflow = await evaluate(`([...document.querySelectorAll('body *')].filter(e => {
      const r=e.getBoundingClientRect();return r.width && r.right>document.documentElement.clientWidth+2;
    }).map(e=>({tag:e.tagName,id:e.id,class:e.getAttribute('class'),right:e.getBoundingClientRect().right,width:e.getBoundingClientRect().width})).slice(0,25))`);
    await fs.writeFile(path.join(output, label+"-overflow.json"), JSON.stringify({result, overflow}, null, 2));
  }
  assert.ok(result.scroll <= result.width + 2, `${label}: document overflows horizontally`);
  assert.equal(result.pink, "rgb(255, 243, 248)");
  assert.ok(result.chart > 200, `${label}: chart is too narrow`);
  checks.push({check: label, ...result});
}

try {
  const portFile = path.join(profile, "DevToolsActivePort");
  for (let i = 0; i < 150 && !existsSync(portFile); i++) {
    if (browser.exitCode !== null) throw new Error(`Browser exited: ${browserLog.slice(-2000)}`);
    await sleep(100);
  }
  const [port, endpoint] = (await fs.readFile(portFile, "utf8")).trim().split(/\r?\n/);
  socket = new WebSocket(`ws://127.0.0.1:${port}${endpoint}`);
  await new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, {once: true});
    socket.addEventListener("error", reject, {once: true});
  });
  socket.addEventListener("message", event => {
    const message = JSON.parse(event.data);
    if (message.id) {
      const task = pending.get(message.id);
      if (!task) return;
      clearTimeout(task.timer);
      pending.delete(message.id);
      if (message.error) task.reject(new Error(JSON.stringify(message.error)));
      else task.resolve(message.result);
    } else if (message.method === "Runtime.exceptionThrown") {
      errors.push(message.params.exceptionDetails);
    } else if (message.method === "Runtime.consoleAPICalled" && message.params.type === "error") {
      errors.push(message.params.args.map(item => item.value || item.description).join(" "));
    } else if (message.method === "Network.requestWillBeSent") {
      network.push(message.params.request.url);
    }
  });
  const target = await request("Target.createTarget", {url: "about:blank"}, null);
  session = (await request("Target.attachToTarget", {targetId: target.targetId, flatten: true}, null)).sessionId;
  await request("Page.enable");
  await request("Runtime.enable");
  await request("Network.enable");
  await request("Emulation.setEmulatedMedia", {features: [{name: "prefers-reduced-motion", value: "reduce"}]});
  await viewport(1440, 900);
  await request("Page.navigate", {url: pathToFileURL(path.join(root, "site", "index.html")).href});
  await waitFor("document.querySelector('#component-chart')?._fullLayout && document.querySelector('#match-chart')?._fullLayout && document.querySelector('#component-fallback').hidden && document.querySelector('#match-fallback').hidden");
  assert.equal(await evaluate("document.querySelectorAll('.factor-list li').length"), 135);
  assert.equal(await evaluate("document.querySelector('#profile-name').textContent"), "Rajat Patidar");
  assert.equal(await evaluate("document.querySelectorAll('.role-table:not([hidden]) tbody tr:not([hidden])').length"), 10);
  assert.equal(await evaluate("document.querySelector('#component-chart').data.length"), 3);
  assert.equal(await evaluate("document.querySelector('.report-note').textContent"), "For the detailed algorithm, refer to the report.");
  await checkFit("desktop");
  await screenshot("desktop");

  await evaluate("document.querySelector('[data-role-button=bowling]').click()");
  await waitFor("document.querySelector('#profile-name').textContent === 'Rashid Khan'");
  await evaluate("document.querySelector('#player-search').value='Bumrah';document.querySelector('#player-search').dispatchEvent(new Event('input',{bubbles:true}))");
  await waitFor("document.querySelector('#profile-name').textContent === 'Jasprit Bumrah'");
  assert.equal(await evaluate("document.querySelectorAll('.role-table:not([hidden]) tbody tr:not([hidden])').length"), 1);
  assert.equal(await evaluate("document.querySelector('#profile-score').textContent"), "52.98");
  const score = await evaluate("document.querySelector('#profile-score').textContent");
  await evaluate("(()=>{const s=document.querySelector('#match-season');s.value=s.options[1].value;s.dispatchEvent(new Event('change',{bubbles:true}));})()");
  const filtered = await evaluate("({year:document.querySelector('#match-season').value,dates:[...document.querySelectorAll('#match-rows tr td:first-child')].map(e=>e.textContent),score:document.querySelector('#profile-score').textContent})");
  assert.ok(filtered.dates.length > 0 && filtered.dates.every(date => date.startsWith(filtered.year)));
  assert.equal(filtered.score, score, "Filtering match evidence must not change the ranking score");
  checks.push({check: "player_search_and_season_filter", result: "passed"});
  await evaluate("document.querySelector('#player').scrollIntoView()");
  await screenshot("player-evidence");

  await evaluate("(()=>{const p=window.CLUTCH_DATA.players.find(p=>p.role==='bowling'&&p.rank===11);const s=document.querySelector('#player-select');s.value=p.key;s.dispatchEvent(new Event('change',{bubbles:true}));})()");
  assert.equal(await evaluate("document.querySelector('#match-unavailable').hidden"), false);
  assert.equal(await evaluate("document.querySelector('#match-evidence').hidden"), true);
  await evaluate("document.querySelector('#player-search').value='no such player qqq';document.querySelector('#player-search').dispatchEvent(new Event('input',{bubbles:true}))");
  assert.equal(await evaluate("document.querySelector('#empty-ranking').hidden"), false);
  await evaluate("document.querySelector('#player-search').value='';document.querySelector('#player-search').dispatchEvent(new Event('input',{bubbles:true}));document.querySelector('#ranking-scope').value='top';document.querySelector('#ranking-scope').dispatchEvent(new Event('change',{bubbles:true}));document.querySelector('[data-role-button=batting]').click()");
  await waitFor("document.querySelector('#profile-name').textContent === 'Rajat Patidar'");
  checks.push({check: "unpackaged_match_detail_and_empty_search", result: "passed"});
  await evaluate("document.querySelector('#expand-factors').click()");
  assert.equal(await evaluate("document.querySelectorAll('.factor-group[open]').length"), 9);
  await evaluate("document.querySelector('#factors').scrollIntoView()");
  await screenshot("factor-checklist");

  for (const [width, height, name] of [[900, 800, "notebook"], [390, 844, "mobile"]]) {
    await viewport(width, height);
    await evaluate("window.scrollTo(0,0)");
    await screenshot(name);
    await checkFit(name);
  }
  assert.deepEqual(errors, [], "Normal interactions must not produce console errors");
  assert.deepEqual(network.filter(url => /^https?:/.test(url)), [], "The dashboard must not fetch external assets");
  checks.push({check: "offline_rendering_and_console", result: "passed"});

  await viewport(1440, 900);
  await request("Network.setBlockedURLs", {urls: ["*plotly-basic.min.js"]});
  await request("Page.reload", {ignoreCache: true});
  await waitFor("document.documentElement.classList.contains('js') && !document.querySelector('#notice').hidden");
  assert.equal(await evaluate("document.querySelector('#component-fallback').hidden"), false);
  assert.equal(await evaluate("document.querySelectorAll('.role-table:not([hidden]) tbody tr:not([hidden])').length"), 10);
  await evaluate("document.querySelector('[data-role-button=bowling]').click()");
  assert.ok((await evaluate("document.querySelector('#component-fallback').src")).endsWith("bowling-scores.svg"));
  checks.push({check: "chart_library_failure_fallback", result: "passed"});
  await screenshot("static-fallback");

  await request("Network.setBlockedURLs", {urls: []});
  await request("Emulation.setScriptExecutionDisabled", {value: true});
  await request("Page.reload", {ignoreCache: true});
  await sleep(600);
  const noScript = await evaluate("({enhanced:document.documentElement.classList.contains('js'),tables:document.querySelectorAll('.role-table:not([hidden])').length,factors:document.querySelectorAll('[data-factor]').length,rows:document.querySelectorAll('.role-table tbody tr').length})");
  assert.equal(noScript.enhanced, false);
  assert.equal(noScript.tables, 2);
  assert.equal(noScript.factors, 135);
  assert.equal(noScript.rows, 153);
  checks.push({check: "javascript_disabled", result: "passed"});
  await fs.writeFile(path.join(output, "checks.json"), JSON.stringify({status: "passed", checks}, null, 2));
  console.log(JSON.stringify({status: "passed", checks, screenshots: output}, null, 2));
} finally {
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({id: ++nextId, method: "Browser.close"}));
  }
  for (let i = 0; i < 100 && browser.exitCode === null; i++) await sleep(100);
  if (browser.exitCode === null) browser.kill();
  socket?.close();
  for (const task of pending.values()) clearTimeout(task.timer);
  await fs.rm(profile, {recursive: true, force: true, maxRetries: 5, retryDelay: 300});
}
