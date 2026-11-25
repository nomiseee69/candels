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
  document.querySelectorAll("[data-panel]").forEach(tab => {
    const active = tab === button;
    tab.setAttribute("aria-pressed", active);
    element(tab.dataset.panel).hidden = !active;
  });
}));
element("close").addEventListener("click", () => telegram?.close());
element("guide").addEventListener("click", () => telegram?.openLink("https://telegram.org/tour/affiliate-programs"));

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
  element("affiliate-guide").hidden = !data.affiliate_active;
  element("affiliate-terms").textContent = data.affiliate_active
    ? "Приглашай друзей и получай 10% в Telegram Stars от их подходящих покупок в течение одного месяца."
    : "Партнёрская программа пока не открыта.";
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
