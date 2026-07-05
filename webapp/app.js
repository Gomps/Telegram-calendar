(() => {
  "use strict";

  const tg = window.Telegram && window.Telegram.WebApp;
  if (tg) {
    tg.ready();
    tg.expand();
    applyThemeParams(tg.themeParams);
    tg.onEvent("themeChanged", () => applyThemeParams(tg.themeParams));
  }

  function applyThemeParams(tp) {
    if (!tp) return;
    const map = {
      bg_color: "--bg",
      secondary_bg_color: "--card-bg",
      text_color: "--text",
      hint_color: "--hint",
      link_color: "--link",
      button_color: "--button",
      button_text_color: "--button-text",
      destructive_text_color: "--danger",
    };
    for (const [tgKey, cssVar] of Object.entries(map)) {
      if (tp[tgKey]) document.documentElement.style.setProperty(cssVar, tp[tgKey]);
    }
  }

  // --- аутентификация -------------------------------------------------------
  const params = new URLSearchParams(location.search);
  const initData = tg && tg.initData ? tg.initData : "";
  const devUserId = params.get("dev_user_id");

  if (!initData) {
    document.getElementById("dev-banner").hidden = false;
  }

  function authHeaders() {
    if (initData) return { Authorization: "tma " + initData };
    if (devUserId) return { "X-Dev-User-Id": devUserId };
    return {};
  }

  async function api(path, options = {}) {
    const resp = await fetch(path, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...authHeaders(),
        ...(options.headers || {}),
      },
    });
    let data = null;
    try {
      data = await resp.json();
    } catch (e) {
      // тело может отсутствовать (например, 204) — не считаем это ошибкой
    }
    if (!resp.ok) {
      const message = (data && data.error) || `HTTP ${resp.status}`;
      throw new Error(message);
    }
    return data;
  }

  // --- рендер -----------------------------------------------------------------

  function el(html) {
    const t = document.createElement("template");
    t.innerHTML = html.trim();
    return t.content.firstChild;
  }

  function renderList(container, items, renderItem, emptyText) {
    container.innerHTML = "";
    if (!items.length) {
      container.appendChild(el(`<div class="empty">${emptyText}</div>`));
      return;
    }
    for (const item of items) container.appendChild(renderItem(item));
  }

  function itemRow(title, sub, onDelete) {
    const row = el(`
      <div class="item">
        <div class="item-main">
          <div class="item-title"></div>
          <div class="item-sub"></div>
        </div>
        <button class="item-del" title="Удалить">🗑</button>
      </div>
    `);
    row.querySelector(".item-title").textContent = title;
    row.querySelector(".item-sub").textContent = sub;
    row.querySelector(".item-del").addEventListener("click", onDelete);
    return row;
  }

  async function loadState() {
    let state;
    try {
      state = await api("/api/state");
    } catch (e) {
      showResult("error", "Не удалось загрузить данные: " + e.message);
      return;
    }

    document.getElementById("clock").textContent =
      `${state.timezone} · сейчас ${state.now_local}`;

    renderList(
      document.getElementById("reminders-list"),
      state.reminders,
      (r) => itemRow(r.text, r.fire_at_human, () => deleteItem("reminders", r.id)),
      "Активных напоминаний нет."
    );

    renderList(
      document.getElementById("series-list"),
      state.series,
      (s) => itemRow(
        s.text,
        s.description + (s.next_occurrence ? ` · ближайшее: ${s.next_occurrence}` : ""),
        () => deleteItem("series", s.id)
      ),
      "Активных серий нет."
    );

    renderList(
      document.getElementById("conditionals-list"),
      state.conditionals,
      (c) => itemRow(
        `${c.question} → ${c.reminder_text}`,
        `${c.status === "asked" ? "вопрос задан" : "проверка " + c.check_at_human} · напомнить ${c.fire_at_human}`,
        () => deleteItem("conditionals", c.id)
      ),
      "Условных напоминаний нет."
    );

    renderList(
      document.getElementById("context-list"),
      state.context,
      (c) => itemRow(c.label, c.value, () => deleteContextKey(c.key)),
      "Распорядок пока пуст."
    );
  }

  async function deleteItem(kind, id) {
    try {
      await api(`/api/${kind}/${id}`, { method: "DELETE" });
      await loadState();
    } catch (e) {
      showResult("error", "Не удалось удалить: " + e.message);
    }
  }

  async function deleteContextKey(key) {
    try {
      await api(`/api/context/${encodeURIComponent(key)}`, { method: "DELETE" });
      await loadState();
    } catch (e) {
      showResult("error", "Не удалось удалить: " + e.message);
    }
  }

  function showResult(kind, text, extraNode) {
    const box = document.getElementById("msg-result");
    box.hidden = false;
    box.className = "result" + (kind === "error" ? " error" : kind === "warn" ? " warn" : "");
    box.textContent = text;
    if (extraNode) box.appendChild(extraNode);
  }

  // --- действия форм -----------------------------------------------------------

  document.getElementById("refresh-btn").addEventListener("click", loadState);

  document.getElementById("msg-send").addEventListener("click", async () => {
    const input = document.getElementById("msg-input");
    const text = input.value.trim();
    if (!text) return;
    const btn = document.getElementById("msg-send");
    btn.disabled = true;
    try {
      const result = await api("/api/message", {
        method: "POST",
        body: JSON.stringify({ text }),
      });
      if (result.text_mismatch) {
        const undoBtn = document.createElement("button");
        undoBtn.className = "secondary-btn";
        undoBtn.textContent = "Отменить (не то поняли)";
        undoBtn.style.marginTop = "8px";
        undoBtn.addEventListener("click", async () => {
          const c = result.created;
          if (c) await deleteItem(c.type === "reminder" ? "reminders"
                                   : c.type === "series" ? "series" : "conditionals", c.id);
          document.getElementById("msg-result").hidden = true;
        });
        showResult("warn", result.message, undoBtn);
      } else if (result.kind === "error") {
        showResult("error", result.message);
      } else {
        showResult("ok", result.message);
        input.value = "";
      }
      await loadState();
    } catch (e) {
      showResult("error", "Ошибка: " + e.message);
    } finally {
      btn.disabled = false;
    }
  });

  document.getElementById("ctx-add").addEventListener("click", async () => {
    const key = document.getElementById("ctx-key").value.trim();
    const value = document.getElementById("ctx-value").value.trim();
    if (!key) return;
    try {
      if (value) {
        await api("/api/context", {
          method: "POST",
          body: JSON.stringify({ updates: { [key]: value } }),
        });
      } else {
        await api(`/api/context/${encodeURIComponent(key)}`, { method: "DELETE" });
      }
      document.getElementById("ctx-key").value = "";
      document.getElementById("ctx-value").value = "";
      await loadState();
    } catch (e) {
      showResult("error", "Не удалось сохранить: " + e.message);
    }
  });

  document.getElementById("tz-save").addEventListener("click", async () => {
    const value = document.getElementById("tz-value").value.trim();
    if (!value) return;
    try {
      await api("/api/timezone", { method: "POST", body: JSON.stringify({ value }) });
      document.getElementById("tz-value").value = "";
      await loadState();
    } catch (e) {
      showResult("error", "Не удалось сохранить: " + e.message);
    }
  });

  loadState();
  setInterval(loadState, 30000);
})();
