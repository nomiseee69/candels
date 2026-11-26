const assert = require("node:assert/strict");
const {readFileSync} = require("node:fs");
const {join} = require("node:path");
const {test} = require("node:test");
const vm = require("node:vm");

const script = readFileSync(join(__dirname, "../miniapp/app.js"), "utf8");

function fixture() {
  const nodes = new Map();
  const requests = [];
  let active = false;
  function element(id) {
    if (!nodes.has(id)) {
      nodes.set(id, {
        hidden: true, disabled: false, textContent: "", value: "",
        replaceChildren(...children) { this.children = children; },
        addEventListener(name, handler) {
          this[name] = () => handler({currentTarget: this, preventDefault() {}});
        }
      });
    }
    return nodes.get(id);
  }
  const context = vm.createContext({
    window: {Telegram: {WebApp: {initData: "signed", ready() {}, expand() {}, onEvent() {}}}},
    document: {documentElement: {dataset: {}}, getElementById: element, querySelectorAll: () => [],
               createElement: () => ({})},
    fetch: async (path, options) => {
      requests.push([path, options]);
      if (path === "/api/partners/register") active = true;
      return {ok: true, json: async () => ({
        enabled: true, commission: 10, registered: active, link: "https://t.me/findercandles_bot?start=fc_example",
        invited: 2, invitations: [[1800000000, "purchased"]], purchases: 1, wallet: null,
        available: "0", held: "0", reserved: "0", paid: "0",
        debt: "0", minimum: "10", remaining: "10", counts: {pending: 1}, earnings: [], payouts: [], plans: []
      })};
    }
  });
  vm.runInContext(script, context);
  return {element, requests, load: () => vm.runInContext("loadPartners()", context)};
}

test("partner sees real backend data and registration submits only signed auth", async () => {
  const ui = fixture();
  await ui.load();
  assert.equal(ui.element("partner-start").hidden, false);
  await ui.element("partner-register").click();
  const [, request] = ui.requests.find(([path]) => path === "/api/partners/register");
  assert.equal(request.headers["X-Telegram-Init-Data"], "signed");
  assert.equal(request.body, "{}");
  assert.match(ui.element("partner-link").value, /start=fc_example$/);
  assert.match(ui.element("partner-counts").textContent, /Приглашено: 2/);
  assert.match(ui.element("partner-invites").children[0].textContent, /оплатил Pro$/);
  assert.equal(ui.element("partner-form").hidden, true);
  assert.equal(ui.element("partner-connect").hidden, false);
});
