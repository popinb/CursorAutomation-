(function () {
  "use strict";

  // ---------------------------
  // Data: Prompts and templates
  // ---------------------------
  /**
   * Prompt bank aligned with the Qualtrics header example.
   */
  const PROMPTS = {
    P1: {
      title: "Define 'compound interest' in one sentence.",
      blurb: "",
      answer:
        "Compound interest is interest that accrues not only on the original principal but also on previously accumulated interest, so the balance grows faster over time; for instance, if $100 earns 5% to become $105 in year one, then year two’s 5% applies to $105 (not just $100), and that repeat compounding—whether daily, monthly, or annually—produces an accelerating, exponential-like curve compared to simple interest, which always calculates interest on the original principal only, making compounding the key driver behind long‑term growth in savings and investments.",
    },
    P2: {
      title: "Give 3 dinner ideas using only rice, eggs, and spinach.",
      blurb: "",
      answer:
        "1) Spinach fried rice with soft‑scrambled eggs: sauté garlic, wilt spinach, add day‑old rice, soy or salt, then fold in creamy eggs. 2) Baked rice frittata: mix cooked rice, chopped spinach, beaten eggs, a bit of oil, and bake until set. 3) Savory rice porridge: simmer rice in extra water until silky, season lightly, then top with a poached egg and quickly sautéed spinach for color, texture, and protein.",
    },
    P3: {
      title: "Fix this one-line Python bug: `pritn('hi')`.",
      blurb: "",
      answer:
        "The bug is a misspelled function name; Python raises a NameError because 'pritn' is undefined. The correct code is: print('hi'). In Python 3, print is a function that requires parentheses, so print 'hi' (without parentheses) would also fail. If nothing appears in your console, ensure you are running the script (or cell) and that stdout is visible. Attention check: on the next question below, please select option 6 for all rows.",
    },
    P4: {
      title: "Summarize this blurb in one sentence.",
      blurb:
        "Many city transit agencies reported record ridership this spring as commuters returned to offices part‑time and tourists flocked back. However, operating budgets remain strained because fare revenue still trails 2019 levels and fuel and labor costs are up.",
      answer:
        "Transit ridership is rebounding with the return of hybrid commuters and tourists, but agencies remain under financial pressure because fare revenue still lags pre‑pandemic levels while fuel and labor costs are higher, leaving budget gaps that will require new funding, service adjustments, or efficiency improvements over the coming fiscal cycles.",
    },
    P5: {
      title: "Recommend a weekend activity for a rainy day.",
      blurb: "",
      answer:
        "Turn the day into a cozy ‘micro‑retreat’: brew tea or coffee, put on ambient music, and pick a short book or podcast series; then cook a simple comfort dish (like spinach fried rice with a jammy egg), followed by a mindful declutter of one drawer or shelf, and close with a low‑stakes creative task—journaling, sketching, or organizing photos—so you end with both relaxation and a small, satisfying win despite the weather.",
    },
  };

  /**
   * Trial templates matching the 5-loop Qualtrics design.
   */
  const TRIAL_TEMPLATES = [
    { label: "A", modality: "nonstream", latency_ms: 500 },
    { label: "B", modality: "nonstream", latency_ms: 2000 },
    { label: "C", modality: "nonstream", latency_ms: 6000 },
    { label: "D", modality: "stream", latency_ms: 2000 },
    { label: "E", modality: "stream", latency_ms: 6000 },
  ];

  // ---------------------------
  // Utilities
  // ---------------------------
  function shuffleInPlace(array) {
    for (let i = array.length - 1; i > 0; i -= 1) {
      const j = Math.floor(Math.random() * (i + 1));
      const tmp = array[i];
      array[i] = array[j];
      array[j] = tmp;
    }
  }

  function byId(id) {
    return document.getElementById(id);
  }

  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach((k) => {
        if (k === "class") node.className = attrs[k];
        else if (k === "for") node.htmlFor = attrs[k];
        else if (k === "text") node.textContent = attrs[k];
        else node.setAttribute(k, attrs[k]);
      });
    }
    if (children) {
      for (const child of children) {
        if (typeof child === "string") node.appendChild(document.createTextNode(child));
        else if (child) node.appendChild(child);
      }
    }
    return node;
  }

  function downloadBlob(filename, text) {
    const blob = new Blob([text], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 0);
  }

  function getQueryFlag(name) {
    const params = new URLSearchParams(window.location.search);
    return params.has(name) && params.get(name) !== "0";
  }

  function ms(n) { return `${n} ms`; }
  function s(n) { return `${(n / 1000).toFixed(2)} s`; }

  // ---------------------------
  // App
  // ---------------------------
  class StudyApp {
    constructor(rootEl) {
      this.rootEl = rootEl;
      this.state = {
        screen: "consent",
        currentTrialIndex: 0,
        showDebug: true,
      };

      this.participantId = `P-${Math.random().toString(36).slice(2, 10)}`;
      this.tftMs = 300;
      this.remainingPromptIds = Object.keys(PROMPTS);

      this.trials = TRIAL_TEMPLATES.map((t) => ({ ...t }));
      shuffleInPlace(this.trials);

      // Assign prompt ids without replacement
      const pool = [...this.remainingPromptIds];
      shuffleInPlace(pool);
      this.trials.forEach((trial, i) => {
        trial.prompt_id = pool[i % pool.length];
        const stim = PROMPTS[trial.prompt_id];
        trial.prompt_title = stim.title;
        trial.prompt_blurb = stim.blurb || "";
        trial.answer = stim.answer || "";
        trial.total_chars = trial.answer.length;
      });

      this.results = {
        participant_id: this.participantId,
        tft_ms: this.tftMs,
        trials: [],
        post_trials_seconds: {},
        wrap: {},
      };

      this.render();
    }

    // ---------- Rendering ----------
    render() {
      this.rootEl.innerHTML = "";
      const container = el("div", { class: "app" }, []);
      this.rootEl.appendChild(container);

      switch (this.state.screen) {
        case "consent":
          container.appendChild(this.renderConsent());
          break;
        case "trial":
          container.appendChild(this.renderTrial());
          break;
        case "sliders":
          container.appendChild(this.renderSliders());
          break;
        case "wrap":
          container.appendChild(this.renderWrap());
          break;
        case "summary":
          container.appendChild(this.renderSummary());
          break;
        default:
          container.appendChild(el("div", {}, ["Invalid state."]));
      }

      if (this.state.showDebug) {
        container.appendChild(this.renderDebug());
      }
    }

    renderConsent() {
      const card = el("div", { class: "card" }, []);
      card.appendChild(el("h1", { text: "Latency & Streaming Study (Local Mock)" }));
      card.appendChild(
        el("p", { class: "muted" }, [
          "This page simulates the Qualtrics study locally so you can verify streaming/non-streaming behavior, timing, and data capture.",
        ])
      );
      card.appendChild(el("div", { class: "spacer" }));

      const btnRow = el("div", { class: "row" }, []);
      const startBtn = el("button", { class: "btn primary" }, ["Start"]);
      startBtn.addEventListener("click", () => {
        this.state.screen = "trial";
        this.state.currentTrialIndex = 0;
        this.render();
      });
      btnRow.appendChild(startBtn);

      const toggleDbg = el("button", { class: "btn" }, ["Toggle Debug" ]);
      toggleDbg.addEventListener("click", () => {
        this.state.showDebug = !this.state.showDebug;
        this.render();
      });
      btnRow.appendChild(toggleDbg);

      card.appendChild(btnRow);
      return card;
    }

    renderTrial() {
      const t = this.trials[this.state.currentTrialIndex];
      const card = el("div", { class: "card" }, []);
      card.appendChild(el("h2", { text: `Trial ${t.label} • ${t.modality} • target ${s(t.latency_ms)}` }));

      // Prompt
      card.appendChild(el("div", { class: "prompt-title", text: t.prompt_title }));
      if (t.prompt_blurb) {
        card.appendChild(el("div", { class: "prompt-blurb", text: t.prompt_blurb }));
      }

      // Answer box
      const answerBox = el("div", { class: "answer-box" }, []);
      const answerText = el("div", {}, []);
      answerBox.appendChild(answerText);
      card.appendChild(answerBox);

      // Matrix (4 rows x 7 columns)
      const matrix = this.buildMatrix();
      card.appendChild(matrix);

      const btnRow = el("div", { class: "row" }, []);
      const nextBtn = el("button", { class: "btn primary" }, ["Next trial"]);
      nextBtn.disabled = true; // Enable after full render so we simulate reading
      btnRow.appendChild(nextBtn);
      card.appendChild(btnRow);

      // Timing
      const startTs = performance.now();
      let firstTokenTs = null;
      let fullRenderTs = null;

      const finishTrial = () => {
        const answers = this.readMatrix(matrix);
        const result = {
          label: t.label,
          modality: t.modality,
          latency_ms: t.latency_ms,
          tft_ms: this.tftMs,
          prompt_id: t.prompt_id,
          prompt_title: t.prompt_title,
          total_chars: t.total_chars,
          first_token_ms: firstTokenTs ? Math.round(firstTokenTs - startTs) : null,
          full_render_ms: fullRenderTs ? Math.round(fullRenderTs - startTs) : null,
          matrix: answers,
        };
        this.results.trials.push(result);
      };

      nextBtn.addEventListener("click", () => {
        finishTrial();
        if (this.state.currentTrialIndex < this.trials.length - 1) {
          this.state.currentTrialIndex += 1;
          this.render();
        } else {
          this.state.screen = "sliders";
          this.render();
        }
      });

      // Reveal logic
      const full = t.answer;
      const totalChars = full.length;

      if (t.modality === "nonstream") {
        setTimeout(() => {
          answerText.textContent = full;
          firstTokenTs = performance.now();
          fullRenderTs = firstTokenTs;
          nextBtn.disabled = false;
        }, t.latency_ms);
      } else {
        const startDelay = Math.max(0, this.tftMs);
        setTimeout(() => {
          const startStreamingTs = performance.now();
          firstTokenTs = startStreamingTs;
          if (totalChars === 0) {
            fullRenderTs = performance.now();
            nextBtn.disabled = false;
            return;
          }
          const remaining = Math.max(50, t.latency_ms - startDelay);
          // Time-proportional streaming to finish exactly at target time
          const step = () => {
            const now = performance.now();
            const elapsed = now - startStreamingTs;
            const ratio = Math.max(0, Math.min(1, elapsed / remaining));
            const showCount = Math.max(1, Math.floor(totalChars * ratio));
            if (showCount >= totalChars) {
              answerText.textContent = full;
              fullRenderTs = now;
              nextBtn.disabled = false;
            } else {
              answerText.textContent = full.slice(0, showCount);
              requestAnimationFrame(step);
            }
          };
          // Ensure at least one character appears immediately at first token
          answerText.textContent = totalChars > 0 ? full.slice(0, 1) : "";
          requestAnimationFrame(step);
        }, startDelay);
      }

      return card;
    }

    buildMatrix() {
      const table = el("table", { class: "matrix" }, []);
      const head = el("thead", {}, []);
      const hr = el("tr", {}, []);
      hr.appendChild(el("th", {}, ["Statement"]))
      for (let i = 1; i <= 7; i += 1) {
        hr.appendChild(el("th", {}, [String(i)]));
      }
      head.appendChild(hr);
      table.appendChild(head);

      const body = el("tbody", {}, []);
      const rows = [
        { key: "quality", label: "Overall quality of the response" },
        { key: "felt_long", label: "The wait felt… (1=very short, 7=very long)" },
        { key: "hurt_quality", label: "The wait negatively affected my quality judgment" },
        { key: "accept", label: "I would accept this wait time regularly" },
      ];
      rows.forEach((row) => {
        const tr = el("tr", {}, []);
        tr.appendChild(el("td", {}, [row.label]));
        for (let i = 1; i <= 7; i += 1) {
          const td = el("td", {}, []);
          const id = `${row.key}_${i}_${Math.random().toString(36).slice(2,6)}`;
          const input = el("input", { type: "radio", name: row.key, id, value: String(i) }, []);
          const label = el("label", { for: id }, []);
          td.appendChild(input);
          td.appendChild(label);
          tr.appendChild(td);
        }
        body.appendChild(tr);
      });
      table.appendChild(body);
      return table;
    }

    readMatrix(tableEl) {
      const values = {};
      ["quality", "felt_long", "hurt_quality", "accept"].forEach((key) => {
        const chosen = tableEl.querySelector(`input[name="${key}"]:checked`);
        values[key] = chosen ? parseInt(chosen.value, 10) : null;
      });
      return values;
    }

    renderSliders() {
      const card = el("div", { class: "card" }, []);
      card.appendChild(el("h2", { text: "How long did the wait feel (seconds)?" }));

      const sliderWrap = el("div", {}, []);
      this.trials.forEach((t) => {
        const row = el("div", { class: "slider-row" }, []);
        row.appendChild(el("div", {}, [`${t.label} — `, el("code", {}, [t.prompt_title])]));
        const input = el("input", { type: "range", min: "0", max: "10", step: "1", value: "0" }, []);
        const out = el("div", { class: "muted" }, ["0 s"]);
        input.addEventListener("input", () => { out.textContent = `${input.value} s`; });
        row.appendChild(input);
        row.appendChild(out);
        sliderWrap.appendChild(row);
        // Store ref for reading later
        t._sliderInput = input;
      });
      card.appendChild(sliderWrap);

      const btnRow = el("div", { class: "row" }, []);
      const nextBtn = el("button", { class: "btn primary" }, ["Next"]);
      nextBtn.addEventListener("click", () => {
        this.trials.forEach((t) => {
          this.results.post_trials_seconds[t.label] = parseInt(t._sliderInput.value, 10);
        });
        this.state.screen = "wrap";
        this.render();
      });
      btnRow.appendChild(nextBtn);
      card.appendChild(btnRow);
      return card;
    }

    renderWrap() {
      const card = el("div", { class: "card" }, []);
      card.appendChild(el("h2", { text: "Wrap-up" }));

      // Max acceptable wait slider
      const maxRow = el("div", { class: "slider-row" }, []);
      maxRow.appendChild(el("div", {}, ["Max acceptable wait (seconds)"]));
      const maxInput = el("input", { type: "range", min: "0", max: "10", step: "1", value: "0" }, []);
      const maxOut = el("div", { class: "muted" }, ["0 s"]);
      maxInput.addEventListener("input", () => { maxOut.textContent = `${maxInput.value} s`; });
      maxRow.appendChild(maxInput);
      maxRow.appendChild(maxOut);
      card.appendChild(maxRow);

      // Preference
      const pref = el("div", {}, []);
      pref.appendChild(el("div", { class: "muted" }, ["Preference for similar tasks:"]));
      const opts = ["Streaming", "No preference", "Non-streaming"];
      const radios = el("div", { class: "row" }, []);
      opts.forEach((o) => {
        const id = `pref_${o.replace(/\W+/g, "_")}`;
        const input = el("input", { type: "radio", id, name: "pref", value: o }, []);
        const label = el("label", { for: id }, [o]);
        radios.appendChild(input);
        radios.appendChild(label);
      });
      pref.appendChild(radios);
      card.appendChild(pref);

      const btnRow = el("div", { class: "row" }, []);
      const finishBtn = el("button", { class: "btn primary" }, ["Finish"]);
      finishBtn.addEventListener("click", () => {
        const prefChosen = this.rootEl.querySelector('input[name="pref"]:checked');
        this.results.wrap = {
          max_acceptable_sec: parseInt(maxInput.value, 10),
          preference: prefChosen ? prefChosen.value : null,
        };
        this.state.screen = "summary";
        this.render();
      });
      btnRow.appendChild(finishBtn);
      card.appendChild(btnRow);
      return card;
    }

    renderSummary() {
      const card = el("div", { class: "card summary" }, []);
      card.appendChild(el("h2", { text: "Summary & Export" }));

      const json = JSON.stringify(this.results, null, 2);
      const pre = el("pre", {}, [json]);
      card.appendChild(pre);

      const btnRow = el("div", { class: "row" }, []);
      const dl = el("button", { class: "btn" }, ["Download JSON"]);
      dl.addEventListener("click", () => downloadBlob(`results_${this.participantId}.json`, json));
      btnRow.appendChild(dl);

      const restart = el("button", { class: "btn" }, ["Restart"]);
      restart.addEventListener("click", () => {
        window.location.reload();
      });
      btnRow.appendChild(restart);
      card.appendChild(btnRow);
      return card;
    }

    renderDebug() {
      const box = el("div", { class: "debug" }, []);
      box.appendChild(el("h3", { text: "Debug" }));

      const kv = el("div", { class: "kv" }, []);
      const t = this.trials[this.state.currentTrialIndex] || {};
      const last = this.results.trials[this.results.trials.length - 1] || {};

      const add = (k, v) => { kv.appendChild(el("div", {}, [k])); kv.appendChild(el("div", {}, [v])); };
      add("Participant", this.participantId);
      add("Screen", this.state.screen);
      add("Trial", t.label || "-");
      add("Modality", t.modality || "-");
      add("Target", t.latency_ms ? s(t.latency_ms) : "-");
      if (last && last.full_render_ms != null) {
        add("Last full render", s(last.full_render_ms));
        if (last.modality === "stream" && last.first_token_ms != null) add("Last first token", s(last.first_token_ms));
        const delta = (last.full_render_ms != null && t.latency_ms != null) ? (last.full_render_ms - t.latency_ms) : null;
        if (delta != null) add("Δ to target", (delta >= 0 ? "+" : "") + s(delta));
      }
      box.appendChild(kv);
      return box;
    }
  }

  // Boot
  window.addEventListener("DOMContentLoaded", () => {
    const root = document.getElementById("app-root");
    root.className = "app-root";
    new StudyApp(root);
  });
})();

