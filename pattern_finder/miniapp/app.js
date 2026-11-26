const telegram = window.Telegram?.WebApp;
const element = id => document.getElementById(id);
const message = text => { element("message").textContent = text; };
function theme() {
  document.documentElement.dataset.theme = telegram?.colorScheme === "light" ? "light" : "dark";
}
theme();
telegram?.onEvent("themeChanged", theme);
telegram?.ready();
telegram?.expand();
document.querySelectorAll("[data-panel]").forEach(button => button.addEventListener("click", () => {
  if (button.dataset.panel === "partners") loadPartners();
  document.querySelectorAll("[data-panel]").forEach(tab => {
    const active = tab === button;
    tab.setAttribute("aria-pressed", active);
    element(tab.dataset.panel).hidden = !active;
  });
}));
element("close").addEventListener("click", () => telegram?.close());
async function api(path, body = {}) {
  const response = await fetch("/api/" + path, {
    method: "POST", headers: {"Content-Type": "application/json", "X-Telegram-Init-Data": telegram.initData},
    body: JSON.stringify(body)
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error);
  return result;
}

function render(data) {
  const date = data.until ? new Date(data.until * 1000).toLocaleDateString("ru-RU") : "";
  element("status").textContent = data.pro ?
    (data.trial ? "Пробный Pro" : "Pro") + (date ? " до " + date : "") : "Free";
  element("access").hidden = data.scans_left == null && !data.purchase_blocked;
  element("access-text").textContent = data.purchase_blocked
    ? (data.cancelled ? "Автопродление отключено через Finder Candles. " :
       data.recurring ? "Последний платёж — с автопродлением. Статус можно проверить в Telegram. " : "")
      + "Новый тариф можно выбрать после окончания оплаченного доступа."
    : `Бесплатных поисков сегодня: ${data.scans_left}`;
  element("cancel-renewal").hidden = !data.can_cancel;
  const discount = element("discount");
  discount.hidden = !data.discount;
  discount.textContent = data.discount ? `Персональная скидка ${data.discount}% на одну оплату` : "";
  const list = element("plans");
  list.replaceChildren();
  for (const plan of data.plans) {
    const card = document.createElement("div");
    card.className = "plan";
    const info = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = plan.key === "month" ? "На месяц" : "На год";
    const terms = document.createElement("p");
    terms.className = "terms";
    terms.textContent = plan.recurring ? "Автопродление каждые 30 дней" :
      `${plan.days} дней · разовая оплата${data.discount ? " · обычная цена " + plan.base + " ⭐" : ""}`;
    info.append(title, terms);
    const buy = document.createElement("button");
    buy.textContent = `⭐ ${plan.amount}`;
    buy.setAttribute("aria-label", `Купить Pro ${title.textContent.toLowerCase()} за ${plan.amount} Stars`);
    buy.addEventListener("click", () => purchase(plan.key, Boolean(data.discount)));
    card.append(info, buy);
    if (data.purchase_blocked) {
      buy.replaceWith(document.createTextNode(`⭐ ${plan.amount}`));
    }
    list.append(card);
  }
}

let busy = false;
async function purchase(plan, discounted) {
  if (busy) return;
  busy = true;
  document.querySelectorAll(".plan button").forEach(button => { button.disabled = true; });
  try {
    message("Готовлю счёт…");
    const invoice = await api("invoice", {plan, discounted});
    const result = await new Promise(resolve => telegram.openInvoice(invoice.url, resolve));
    if (result === "cancelled" || result === "failed") {
      message(result === "cancelled" ? "Счёт закрыт" : "Оплата не завершена");
      return;
    }
    message("Проверяю подтверждение Telegram…");
    for (let attempt = 0; attempt < 12; attempt++) {
      const data = await api("state", {payload: invoice.payload});
      render(data);
      if (data.paid) {
        message("Оплата подтверждена");
        return;
      }
      await new Promise(resolve => setTimeout(resolve, 1500));
    }
    message("Подтверждение ещё не пришло. Открой приложение позже, чтобы проверить срок Pro.");
  } catch (error) {
    message(error.message || "Попробуй позже.");
  } finally {
    busy = false;
    document.querySelectorAll(".plan button").forEach(button => { button.disabled = false; });
  }
}

element("cancel-renewal").addEventListener("click", async event => {
  if (!window.confirm("Отключить будущие списания? Оплаченный доступ сохранится.")) return;
  const button = event.currentTarget;
  button.disabled = true;
  try {
    render(await api("cancel"));
    message("Автопродление отключено.");
  } catch (error) {
    message(error.message);
  } finally {
    button.disabled = false;
  }
});

if (telegram?.initData) {
  api("state").then(render).catch(error => {
    element("status").textContent = "—";
    message(error.message);
  });
} else {
  element("status").textContent = "—";
  message("Открой приложение из Telegram.");
}

// Partner controls are loaded on demand, independently of Pro checkout.
let partnerState;
let walletUi;
let pendingProof;
let openingWallet = false;
const partnerText = text => { element("partner-message").textContent = text; };
const labels = {pending:"ожидает сверки", matured_unfunded:"ожидает обеспечения",
  funded:"обеспечено", reversed:"отменено", requested:"на проверке", approved:"одобрено",
  submitted:"обрабатывается", paid:"выплачено", rejected:"отклонено", cancelled:"отменено"};
const inviteLabels = {invited: "присоединился", purchased: "оплатил Pro", reversed: "возврат оплаты"};
const dateText = stamp => new Date(stamp * 1000).toLocaleDateString("ru-RU");
function list(id, rows) {
  element(id).replaceChildren(...(rows.length ? rows : ["Пока нет записей"]).map(text => {
    const li = document.createElement("li");
    li.textContent = text;
    return li;
  }));
}

async function loadPartners() {
  try {
    const s = partnerState = await api("partners/state");
    element("partner-terms").textContent = s.enabled || s.registered ?
      s.attribution_open === false ? "Ссылку можно получить заранее. Приглашения начнут учитываться после открытия программы." :
        `Приглашай друзей и получай ${s.commission}% чистого дохода с их первой подходящей покупки Pro.` :
      "Регистрация пока закрыта.";
    element("partner-rules").textContent = `Комиссия ${s.commission}% с первой подходящей покупки Pro. Минимальный вывод ${cents(s.minimum)} USDT.`;
    element("partner-start").hidden = !s.enabled || s.registered;
    element("partner-panel").hidden = !s.registered;
    if (!s.registered) return;
    element("partner-link").value = s.link;
    element("partner-counts").textContent = `Приглашено: ${s.invited} · первых покупок: ${s.purchases}`;
    list("partner-invites", s.invitations.map(([at, state]) => `${dateText(at)} · ${inviteLabels[state]}`));
    element("partner-available").textContent = cents(s.available);
    element("partner-balances").textContent = `Зарезервировано ${cents(s.reserved)} · выплачено ${cents(s.paid)} USDT`;
    element("partner-held").textContent = [
      s.estimated != null ? `Ожидается ≈ ${cents(s.estimated)} USDT (оценка)` : "",
      s.unquoted ? `Без расчётного курса ${s.unquoted}` : "",
      s.held !== "0" ? `На сверке резерва ${cents(s.held)} USDT` : "",
      s.debt !== "0" ? `Корректировка после возврата ${cents(s.debt)} USDT` : ""
    ].filter(Boolean).join(" · ");
    element("partner-wallet").textContent = s.wallet ?
      s.wallet.address.slice(0, 10) + "…" + s.wallet.address.slice(-6) + " · подтверждён" : "Кошелёк пока не подключён";
    const hasOpen = s.payouts.some(row => ["requested", "approved", "submitted"].includes(row[2]));
    element("partner-connect").hidden = s.wallet_connect === false || Boolean(s.wallet) || hasOpen;
    element("partner-disconnect").hidden = !s.wallet || hasOpen;
    element("partner-threshold").textContent = hasOpen ? "⏳ Выплата обрабатывается" :
      s.wallet?.unlock * 1000 > Date.now() ? "После смены адреса вывод временно ограничен." :
      s.remaining !== "0" ? `До вывода осталось ${cents(s.remaining, true)} USDT.` :
      s.can_request ? `Минимальный вывод ${cents(s.minimum)} USDT.` : "Заявка пока недоступна.";
    element("partner-form").hidden = !s.can_request;
    list("partner-earnings", s.earnings.map(([plan, at, state, units]) =>
      `${dateText(at)} · ${plan === "month" ? "месяц" : "год"} · ${labels[state]}${units == null ? "" : " · " + microText(units) + " USDT"}`));
    list("partner-payouts", s.payouts.map(([id, units, state, due, reason]) =>
      `№${id} · ${microText(units)} USDT · ${labels[state]}` +
      (["requested", "approved", "submitted"].includes(state) ? ` · обработка до ${dateText(due)}` : "") +
      (reason ? ` · ${reason}` : "")));
  } catch (error) {
    element("partner-terms").textContent = telegram?.initData ? "Раздел временно недоступен." : "Открой приложение из Telegram.";
    partnerText(error.message);
  }
}

function microText(units) {
  const amount = BigInt(units);
  return (amount / 1000000n).toString() + "." + (amount % 1000000n).toString().padStart(6, "0");
}

function unitsFromDecimal(value) {
  const [integer, fraction = ""] = value.split(".");
  return BigInt(integer) * 1000000n + BigInt(fraction.padEnd(6, "0"));
}

function cents(value, roundUp = false) {
  const units = unitsFromDecimal(value);
  const hundredths = (units + (roundUp ? 9999n : 0n)) / 10000n;
  return `${hundredths / 100n},${String(hundredths % 100n).padStart(2, "0")}`;
}

function partnerClick(id, action, eventName = "click") {
  element(id).addEventListener(eventName, async event => {
    event.preventDefault();
    const control = event.currentTarget;
    const button = eventName === "submit" ? control.querySelector("button") : control;
    if (button.disabled) return;
    button.disabled = true;
    partnerText("");
    try { await action(control); }
    catch (error) { partnerText(error.message || "Попробуй позже."); }
    finally { button.disabled = false; }
  });
}
partnerClick("partner-register", async () => { await api("partners/register"); await loadPartners(); });
partnerClick("partner-copy", async () => {
  const field = element("partner-link");
  try { await navigator.clipboard.writeText(field.value); partnerText("Ссылка скопирована."); }
  catch { field.focus(); field.select(); partnerText("Выделил ссылку. Выбери «Копировать»."); }
});

async function connectUi(config) {
  if (walletUi) return walletUi;
  await new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = config.sdk;
    script.crossOrigin = "anonymous";
    script.integrity = "sha384-zGhKyT491TjdZycV4y1fa0Z1NUdta5MUDimkZ5n6+32EFHhkv74w3mzohlRK3X50";
    script.onload = resolve;
    script.onerror = () => { script.remove(); reject(new Error("TON Connect недоступен.")); };
    document.head.append(script);
  });
  walletUi = new window.TON_CONNECT_UI.TonConnectUI({manifestUrl: config.manifest});
  walletUi.onStatusChange(async wallet => {
    if (!wallet || !pendingProof) return;
    const attempt = pendingProof;
    pendingProof = undefined;
    const proof = wallet.connectItems?.tonProof?.proof;
    if (!proof || proof.payload !== attempt.payload) {
      attempt.error("Кошелёк не предоставил актуальное подтверждение TON. Подключи его заново.");
      return;
    }
    try {
      const result = await api(attempt.verify, {...attempt.body, account: wallet.account, proof});
      await attempt.success(result);
    } catch (error) { attempt.error(error.message); }
  });
  await walletUi.connectionRestored;
  return walletUi;
}
async function requestWalletProof(config, attempt) {
  if (openingWallet) throw new Error("Подожди открытия кошелька.");
  openingWallet = true;
  pendingProof = undefined;
  try {
    const ui = await connectUi(config);
    if (ui.connected) await ui.disconnect();
    const result = await api(attempt.challenge, attempt.body);
    if (attempt.showReport) attempt.showReport(result.report);
    pendingProof = {...attempt, payload: result.payload};
    ui.setConnectRequestParameters({state: "ready", value: {tonProof: result.payload}});
    await ui.openModal();
  } finally {
    openingWallet = false;
  }
}
partnerClick("partner-connect", async () => requestWalletProof(partnerState, {
  challenge: "partners/challenge", verify: "partners/verify", error: partnerText,
  success: async () => { await loadPartners(); partnerText("Кошелёк подтверждён."); }
}));
partnerClick("partner-disconnect", async () => {
  await api("partners/disconnect");
  if (walletUi?.connected) await walletUi.disconnect();
  await loadPartners();
});
partnerClick("partner-all", async () => {
  const field = element("partner-amount");
  field.value = partnerState.available;
  field.focus();
});
partnerClick("partner-form", async form => {
  const amount = element("partner-amount").value.trim().replace(",", ".");
  if (!/^\d{1,9}(?:\.\d{1,6})?$/.test(amount)) throw new Error("Укажи сумму в USDT.");
  if (unitsFromDecimal(amount) > unitsFromDecimal(partnerState.available)) throw new Error("Сумма выше доступного остатка.");
  const result = await api("partners/request", {amount});
  form.reset();
  await loadPartners();
  partnerText(`Заявка №${result.request} принята.`);
}, "submit");

const ownerSlots = ["income", "partner"];
let ownerWalletConfig;
const ownerStatuses = {confirmed: "Подтверждён", failed: "Требуется проверка", pending: "Требуется проверка",
  unsupported: "Не поддерживается", found: "Подтверждён", not_found: "Не найден или ещё не проиндексирован",
  unavailable: "Временно недоступен", active: "Активен", uninit: "Не развёрнут", frozen: "Заморожен",
  nonexist: "Аккаунт отсутствует"};

function showOwnerReport(slot, report) {
  const rows = ["Адрес и контрольная сумма — подтверждены", "Workchain — 0"];
  if (report) {
    rows.push(
      `Аккаунт в Mainnet — ${ownerStatuses[report.account]}`,
      `Версия — ${report.version || "Требуется проверка"}` + (report.supported === false ? " · Не поддерживается" : ""),
      `TON Connect в Mainnet — ${ownerStatuses[report.mainnet]}` + (report.network === false ? " · Выбрана другая сеть" : ""),
      `Подпись владельца — ${ownerStatuses[report.signature]}`,
      `Совпадение адреса — ${report.match === null ? "Требуется проверка" : report.match ? "Подтверждено" : "Адрес не совпадает"}`,
      `Публичный ключ в TonAPI — ${ownerStatuses[report.key]}`,
      `Официальный USDT Jetton — ${ownerStatuses[report.jetton]}`,
      `Баланс — ${report.ton ?? "—"} TON · ${report.usdt ?? "—"} USDT`,
      `Данные на ${new Date(report.checked_at * 1000).toLocaleString("ru-RU")}`
    );
    if (report.signature === "confirmed" && report.mainnet !== "confirmed") {
      rows.push("Подпись совпала с адресом. Активный контракт и ключ в Mainnet пока не подтверждены.");
    }
  }
  list(`owner-${slot}-report`, rows);
}

async function loadOwnerWallets() {
  try {
    const config = await api("owner-wallets/state");
    if (!config.cards) return;
    ownerWalletConfig = config;
    for (const slot of ownerSlots) {
      const card = config.cards[slot];
      element(`owner-${slot}-address`).textContent = card.checksum ? card.address : "Проверь адрес в настройках.";
      element(`owner-${slot}-check`).disabled = !card.checksum;
      if (card.checksum) showOwnerReport(slot, card.report);
    }
    element("owner-wallet-tab").hidden = false;
  } catch { /* The owner section stays hidden unless the backend grants access. */ }
}

for (const slot of ownerSlots) {
  element(`owner-${slot}-check`).addEventListener("click", async event => {
    const button = event.currentTarget;
    const status = text => { element(`owner-${slot}-message`).textContent = text; };
    button.disabled = true;
    status("Получаем сведения о кошельке…");
    try {
      await requestWalletProof(ownerWalletConfig, {
        challenge: "owner-wallets/challenge", verify: "owner-wallets/verify", body: {slot}, error: status,
        showReport: report => { showOwnerReport(slot, report); status("Выбери этот адрес в приложении кошелька."); },
        success: ({report}) => { showOwnerReport(slot, report); status(""); }
      });
    } catch (error) { status(error.message); }
    finally {button.disabled = false;}
  });
}
if (telegram?.initData) loadOwnerWallets();
