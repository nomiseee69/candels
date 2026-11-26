const assert = require("node:assert/strict");
const {readFileSync} = require("node:fs");
const {join} = require("node:path");
const {test} = require("node:test");
const vm = require("node:vm");

const script = readFileSync(join(__dirname, "../miniapp/app.js"), "utf8");

function fixture() {
  const nodes = new Map();
  const timers = [];
  const requests = [];
  let closed = 0;
  let respond;
  function element(id) {
    if (!nodes.has(id)) {
      nodes.set(id, {
        hidden: true, disabled: false, open: false, textContent: "",
        replaceChildren() {},
        addEventListener(name, handler) { this[name] = () => handler({currentTarget: this}); }
      });
    }
    return nodes.get(id);
  }
  vm.runInNewContext(script, {
    window: {Telegram: {WebApp: {
      initData: "signed", ready() {}, expand() {}, onEvent() {}, close() { closed++; }
    }}},
    document: {documentElement: {dataset: {}}, getElementById: element, querySelectorAll: () => []},
    setTimeout: fn => timers.push(fn),
    fetch: (path, options) => {
      if (path === "/api/state") {
        return Promise.resolve({ok: true, json: async () => ({plans: [], affiliate_active: true})});
      }
      requests.push([path, options]);
      return new Promise(resolve => { respond = resolve; });
    }
  });
  return {element, timers, requests, closed: () => closed,
          respond: (ok, body) => respond({ok, json: async () => body})};
}

test("successful delivery closes Mini App and rapid presses send one request", async () => {
  const ui = fixture();
  const button = ui.element("affiliate-open");
  const first = button.click();
  await button.click();
  assert.equal(button.disabled, true);
  assert.equal(ui.requests.length, 1);
  assert.equal(ui.requests[0][0], "/api/affiliate-message");
  assert.equal(ui.requests[0][1].headers["X-Telegram-Init-Data"], "signed");
  assert.equal(ui.requests[0][1].body, "{}");
  assert.equal(ui.closed(), 0);
  ui.respond(true, {sent: true});
  await first;
  assert.match(ui.element("affiliate-navigation").textContent, /отправлена/);
  assert.equal(ui.timers.length, 1);
  ui.timers[0]();
  assert.equal(ui.closed(), 1);
});

test("failure and unconfirmed success keep Mini App open and offer instructions", async () => {
  const ui = fixture();
  const button = ui.element("affiliate-open");
  for (const [ok, body] of [[false, {error: "Разблокируй бота."}], [true, {}]]) {
    const request = button.click();
    ui.respond(ok, body);
    await request;
    assert.equal(button.disabled, false);
    assert.equal(ui.element("affiliate-steps").open, true);
    assert.equal(ui.timers.length, 0);
    assert.equal(ui.closed(), 0);
  }
  assert.match(ui.element("affiliate-navigation").textContent, /не подтверждена/);
});
