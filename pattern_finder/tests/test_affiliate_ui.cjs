const assert = require("node:assert/strict");
const {readFileSync} = require("node:fs");
const {join} = require("node:path");
const {test} = require("node:test");
const vm = require("node:vm");

const script = readFileSync(join(__dirname, "../miniapp/app.js"), "utf8");

function fixture(overrides = {}, owner = null) {
  const nodes = new Map();
  const requests = [];
  let active = false;
  function element(id) {
    if (!nodes.has(id)) {
      nodes.set(id, {
        hidden: true, disabled: false, textContent: "", value: "",
        focus() {}, reset() {}, querySelector() { return this; },
        replaceChildren(...children) { this.children = children; },
        addEventListener(name, handler) {
          this[name] = () => handler({currentTarget: this, preventDefault() {}});
        }
      });
    }
    return nodes.get(id);
  }
  const context = vm.createContext({
    window: {Telegram: {WebApp: {initData: "signed", ready() {}, expand() {}, onEvent() {}}},
             TON_CONNECT_UI: {TonConnectUI: class {
      onStatusChange(callback) {this.callback = callback;}
      setConnectRequestParameters(parameters) {this.payload = parameters.value.tonProof;}
      async openModal() {await this.callback({account: {chain: "-239"}, connectItems: {tonProof: {proof: {payload: this.payload}}}});}
    }}},
    document: {documentElement: {dataset: {}}, getElementById: element, querySelectorAll: () => [],
               createElement: () => ({}), head: {append(script) {script.onload();}}},
    fetch: async (path, options) => {
      requests.push([path, options]);
      if (path === "/api/owner-wallets/state") return {ok: Boolean(owner), json: async () => owner || {error:"Недоступно"}};
      if (path === "/api/owner-wallets/challenge") return {ok: true, json: async () => ({payload:"nonce", report:owner.report})};
      if (path === "/api/owner-wallets/verify") return {ok: true, json: async () => ({report:{...owner.report, signature:"confirmed"}})};
      if (path === "/api/partners/register") active = true;
      return {ok: true, json: async () => ({
        enabled: true, commission: 10, registered: active, link: "https://t.me/findercandles_bot?start=fc_example",
        invited: 2, invitations: [[1800000000, "purchased"]], purchases: 1, wallet: null,
        available: "0", held: "0", reserved: "0", paid: "0",
        debt: "0", minimum: "10", remaining: "10", counts: {pending: 1}, earnings: [], payouts: [], plans: [],
        ...overrides
      })};
    }
  });
  vm.runInContext(script, context);
  return {element, requests, load: () => vm.runInContext("loadPartners()", context),
          loadOwner: () => vm.runInContext("loadOwnerWallets()", context)};
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

test("balance never rounds up and withdraw all uses the exact micro-USDT value", async () => {
  const ui = fixture({registered: true, available: "10.009999", remaining: "0", reserved: "5", paid: "2.15",
    estimated: "0.50", unquoted: 2, can_request: true, wallet: {address: "0:" + "a".repeat(64), unlock: 0}});
  await ui.load();
  assert.equal(ui.element("partner-available").textContent, "10,00");
  assert.match(ui.element("partner-held").textContent, /0,50 USDT \(оценка\).*Без расчётного курса 2/);
  ui.element("partner-amount").value = "10,01";
  await ui.element("partner-form").submit();
  assert.match(ui.element("partner-message").textContent, /выше доступного/);
  assert.ok(!ui.requests.some(([path]) => path === "/api/partners/request"));
  await ui.element("partner-all").click();
  assert.equal(ui.element("partner-amount").value, "10.009999");
  await ui.element("partner-form").submit();
  assert.equal(ui.requests.find(([path]) => path === "/api/partners/request")[1].body, '{"amount":"10.009999"}');
});

test("small shortfall stays visible near the minimum", async () => {
  const ui = fixture({registered: true, available: "9.999999", remaining: "0.000001"});
  await ui.load();
  assert.equal(ui.element("partner-available").textContent, "9,99");
  assert.match(ui.element("partner-threshold").textContent, /0,01 USDT/);
  assert.equal(ui.element("partner-form").hidden, true);
});

test("owner buttons use diagnostic proof routes and stay hidden without backend access", async () => {
  const ordinary = fixture();
  await ordinary.loadOwner();
  assert.equal(ordinary.element("owner-wallet-tab").hidden, true);
  const ui = fixture({}, {sdk:"sdk", manifest:"manifest", report:{account:"not_found", mainnet:"pending",
    signature:"pending", jetton:"not_found", key:"not_found", checked_at:1800000000, match:null},
    cards:{income:{checksum:true,address:"income"},partner:{checksum:true,address:"partner"}}});
  await ui.loadOwner();
  assert.equal(ui.element("owner-wallet-tab").hidden, false);
  for (const slot of ["income", "partner"]) await ui.element(`owner-${slot}-check`).click();
  const verify = ui.requests.filter(([path]) => path === "/api/owner-wallets/verify");
  assert.deepEqual(verify.map(([, request]) => JSON.parse(request.body).slot), ["income", "partner"]);
  assert.ok(!ui.requests.some(([path]) => path.startsWith("/api/partners/")));
});
