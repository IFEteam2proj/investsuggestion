/* ==========================================================================
   IFE Team #2 — Interactive Research Portal Application Logic (app.js)
   ========================================================================== */

(function () {
  "use strict";

  // --------------------------------------------------------------------------
  // 1. Theme & Navigation Initialization
  // --------------------------------------------------------------------------
  function initThemeAndNav() {
    const savedTheme = localStorage.getItem("ife_theme") || "light";
    document.documentElement.setAttribute("data-theme", savedTheme);

    const themeBtn = document.getElementById("themeToggle");
    if (themeBtn) {
      themeBtn.textContent = savedTheme === "dark" ? "☀️" : "🌙";
      themeBtn.addEventListener("click", () => {
        const cur = document.documentElement.getAttribute("data-theme") || "light";
        const next = cur === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", next);
        localStorage.setItem("ife_theme", next);
        themeBtn.textContent = next === "dark" ? "☀️" : "🌙";
        if (window.renderAllCharts) window.renderAllCharts();
      });
    }

    const menuBtn = document.getElementById("mobileMenuBtn");
    const navLinks = document.getElementById("navLinks");
    if (menuBtn && navLinks) {
      menuBtn.addEventListener("click", () => {
        navLinks.classList.toggle("open");
      });
    }

    // Lightbox for report figures
    const lightbox = document.getElementById("lightbox");
    const lightboxImg = document.getElementById("lightboxImg");
    const lightboxClose = document.getElementById("lightboxClose");

    document.querySelectorAll(".zoomable-img").forEach((img) => {
      img.addEventListener("click", () => {
        if (lightbox && lightboxImg) {
          lightboxImg.src = img.src;
          lightboxImg.alt = img.alt || "Report Figure";
          lightbox.classList.add("open");
        }
      });
    });

    if (lightbox) {
      lightbox.addEventListener("click", (e) => {
        if (e.target === lightbox || e.target === lightboxClose) {
          lightbox.classList.remove("open");
        }
      });
    }
  }

  // --------------------------------------------------------------------------
  // 2. Helper Utilities
  // --------------------------------------------------------------------------
  function fmtNum(val, digits = 2, suffix = "") {
    if (val === null || val === undefined || val === "") return "—";
    const n = parseFloat(val);
    if (isNaN(n)) return val;
    return n.toFixed(digits) + suffix;
  }

  function fmtSigned(val, digits = 2, suffix = "") {
    if (val === null || val === undefined || val === "") return "—";
    const n = parseFloat(val);
    if (isNaN(n)) return val;
    const sign = n > 0 ? "+" : "";
    return sign + n.toFixed(digits) + suffix;
  }

  function getChartColors() {
    const isDark = document.documentElement.getAttribute("data-theme") === "dark";
    return {
      text: isDark ? "#cbd5e1" : "#334155",
      grid: isDark ? "rgba(148, 163, 184, 0.14)" : "rgba(15, 23, 42, 0.08)",
      main: isDark ? "#3b82f6" : "#1d4ed8",
      mainFill: isDark ? "rgba(59, 130, 246, 0.16)" : "rgba(29, 78, 216, 0.10)",
      sp500: isDark ? "#94a3b8" : "#475569",
      ew20: isDark ? "#10b981" : "#059669",
      enet: isDark ? "#a78bfa" : "#7c3aed",
      rf: isDark ? "#f59e0b" : "#d97706",
      tr: isDark ? "#ec4899" : "#db2777",
      ann: isDark ? "#14b8a6" : "#0d9488",
      danger: isDark ? "#ef4444" : "#dc2626"
    };
  }

  const activeCharts = {};
  function createOrUpdateChart(id, config) {
    const canvas = document.getElementById(id);
    if (!canvas || typeof Chart === "undefined") return null;
    if (activeCharts[id]) {
      activeCharts[id].destroy();
    }
    activeCharts[id] = new Chart(canvas.getContext("2d"), config);
    return activeCharts[id];
  }

  // --------------------------------------------------------------------------
  // 3. Chart Renderers (Home, Results, QA Pages)
  // --------------------------------------------------------------------------
  function renderWealthAndDrawdownCharts() {
    const data = window.PROJECT_CSV_DATA;
    if (!data || !data.monthlySeries) return;
    const series = data.monthlySeries;
    const colors = getChartColors();

    const labels = ["2010-12", ...series.map((d) => String(d.date).slice(0, 7))];
    let wMain = 100, wGross = 100, wSP = 100, wEW = 100, wEnet = 100, wRF = 100;
    const cMain = [100], cGross = [100], cSP = [100], cEW = [100], cEnet = [100], cRF = [100];
    let pMain = 100, pSP = 100, pEW = 100;
    const ddMain = [0], ddSP = [0], ddEW = [0];

    series.forEach((m) => {
      wMain *= 1 + (parseFloat(m.main_net) || 0) / 100;
      wGross *= 1 + (parseFloat(m.main_gross) || 0) / 100;
      wSP *= 1 + (parseFloat(m.sp500) || 0) / 100;
      wEW *= 1 + (parseFloat(m.ew20) || 0) / 100;
      wEnet *= 1 + (parseFloat(m.enet_net) || 0) / 100;
      wRF *= 1 + (parseFloat(m.rf_net) || 0) / 100;

      cMain.push(+wMain.toFixed(2));
      cGross.push(+wGross.toFixed(2));
      cSP.push(+wSP.toFixed(2));
      cEW.push(+wEW.toFixed(2));
      cEnet.push(+wEnet.toFixed(2));
      cRF.push(+wRF.toFixed(2));

      pMain = Math.max(pMain, wMain);
      pSP = Math.max(pSP, wSP);
      pEW = Math.max(pEW, wEW);

      ddMain.push(+(((wMain / pMain) - 1) * 100).toFixed(2));
      ddSP.push(+(((wSP / pSP) - 1) * 100).toFixed(2));
      ddEW.push(+(((wEW / pEW) - 1) * 100).toFixed(2));
    });

    const scaleModeBtn = document.querySelector(".scale-toggle-btn.active");
    const isLog = scaleModeBtn && scaleModeBtn.dataset.scale === "logarithmic";

    createOrUpdateChart("wealthChart", {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Main: XGBoost × M4 (Net, $603.09)",
            data: cMain,
            borderColor: colors.main,
            backgroundColor: colors.mainFill,
            borderWidth: 2.8,
            pointRadius: 0,
            pointHoverRadius: 4,
            fill: true,
            tension: 0.15
          },
          {
            label: "Main: XGBoost × M4 (Gross, $662.54)",
            data: cGross,
            borderColor: colors.enet,
            borderDash: [4, 4],
            borderWidth: 1.6,
            pointRadius: 0,
            fill: false,
            tension: 0.15
          },
          {
            label: "S&P 500 Total Return ($367.00)",
            data: cSP,
            borderColor: colors.sp500,
            borderWidth: 2.2,
            pointRadius: 0,
            fill: false,
            tension: 0.15
          },
          {
            label: "Equal-Weight 20 (Net, $340.11)",
            data: cEW,
            borderColor: colors.ew20,
            borderDash: [6, 3],
            borderWidth: 2,
            pointRadius: 0,
            fill: false,
            tension: 0.15
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { labels: { color: colors.text, font: { weight: "600", size: 12 } } },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.dataset.label.split(" (")[0]}: $${ctx.parsed.y.toFixed(2)}`
            }
          }
        },
        scales: {
          x: {
            ticks: { color: colors.text, maxTicksLimit: 11 },
            grid: { color: colors.grid }
          },
          y: {
            type: isLog ? "logarithmic" : "linear",
            ticks: {
              color: colors.text,
              callback: (v) => "$" + v
            },
            grid: { color: colors.grid }
          }
        }
      }
    });

    createOrUpdateChart("drawdownChart", {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Main: XGBoost × M4 (MDD -19.55%)",
            data: ddMain,
            borderColor: colors.main,
            backgroundColor: colors.mainFill,
            borderWidth: 2.2,
            pointRadius: 0,
            fill: true
          },
          {
            label: "S&P 500 TR (MDD -19.60%)",
            data: ddSP,
            borderColor: colors.sp500,
            borderWidth: 1.8,
            pointRadius: 0,
            fill: false
          },
          {
            label: "Equal-Weight 20 (MDD -25.07%)",
            data: ddEW,
            borderColor: colors.ew20,
            borderDash: [5, 3],
            borderWidth: 1.8,
            pointRadius: 0,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { labels: { color: colors.text, font: { weight: "600", size: 12 } } },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.dataset.label.split(" (")[0]}: ${ctx.parsed.y.toFixed(2)}%`
            }
          }
        },
        scales: {
          x: { ticks: { color: colors.text, maxTicksLimit: 11 }, grid: { color: colors.grid } },
          y: {
            max: 0,
            min: -30,
            ticks: { color: colors.text, callback: (v) => v + "%" },
            grid: { color: colors.grid }
          }
        }
      }
    });

    // Annual Returns Bar Chart
    if (data.annualReturns && document.getElementById("annualBarChart")) {
      const years = data.annualReturns.map((r) => r.year || r.H1);
      createOrUpdateChart("annualBarChart", {
        type: "bar",
        data: {
          labels: years,
          datasets: [
            {
              label: "Main (XGBoost × M4, Net)",
              data: data.annualReturns.map((r) => parseFloat(r["XGB_M4 (메인, 비용 후)"] ?? r["Main (Net)"])),
              backgroundColor: colors.main,
              borderRadius: 4
            },
            {
              label: "S&P 500 TR",
              data: data.annualReturns.map((r) => parseFloat(r["S&P500 TR"] ?? r["S&P 500"])),
              backgroundColor: colors.sp500,
              borderRadius: 4
            },
            {
              label: "Equal-Weight 20 (Net)",
              data: data.annualReturns.map((r) => parseFloat(r["동일가중 20"] ?? r["Equal-Weight (Net)"])),
              backgroundColor: colors.ew20,
              borderRadius: 4
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: "index", intersect: false },
          plugins: {
            legend: { labels: { color: colors.text, font: { weight: "600", size: 12 } } },
            tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)}%` } }
          },
          scales: {
            x: { ticks: { color: colors.text }, grid: { color: colors.grid } },
            y: { ticks: { color: colors.text, callback: (v) => v + "%" }, grid: { color: colors.grid } }
          }
        }
      });
    }

    // Monthly Turnover Chart
    if (document.getElementById("turnoverChart")) {
      createOrUpdateChart("turnoverChart", {
        type: "bar",
        data: {
          labels: series.map((d) => String(d.date).slice(0, 7)),
          datasets: [
            {
              label: "월별 턴오버 (One-Way Turnover %, 평균 39.20%)",
              data: series.map((d) => parseFloat(d.turnover) || 0),
              backgroundColor: colors.main,
              borderRadius: 2
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { labels: { color: colors.text } }
          },
          scales: {
            x: { ticks: { color: colors.text, maxTicksLimit: 12 }, grid: { color: colors.grid } },
            y: { ticks: { color: colors.text, callback: (v) => v + "%" }, grid: { color: colors.grid } }
          }
        }
      });
    }

    // Dynamic Portfolio Weights Chart (Top 6 Holdings + Others)
    if (data.weightsMain && document.getElementById("weightsAreaChart")) {
      const wRows = data.weightsMain;
      const wDates = wRows.map((r) => String(r.decision_date || r.H1).slice(0, 7));
      const topHoldings = [
        { key: "AAPL(14593)", ticker: "AAPL", label: "AAPL (평균 17.89%)" },
        { key: "CNI(84181)", ticker: "CNI", label: "CNI (평균 8.31%)" },
        { key: "UGI(17929)", ticker: "UGI", label: "UGI (평균 8.29%)" },
        { key: "CMI(41080)", ticker: "CMI", label: "CMI (평균 7.28%)" },
        { key: "ANSS(83621)", ticker: "ANSS", label: "ANSS (평균 7.03%)" },
        { key: "SRCL(83906)", ticker: "SRCL", label: "SRCL (평균 7.00%)" }
      ];
      const palette = ["#1d4ed8", "#0284c7", "#059669", "#d97706", "#7c3aed", "#db2777", "#94a3b8"];

      const getVal = (row, item) => {
        const raw = parseFloat(row[item.key] ?? row[item.ticker] ?? 0);
        if (isNaN(raw)) return 0;
        return raw <= 1.05 ? raw * 100 : raw;
      };

      const datasets = topHoldings.map((item, idx) => ({
        label: item.label,
        data: wRows.map((r) => +getVal(r, item).toFixed(2)),
        backgroundColor: palette[idx] + "cc",
        borderColor: palette[idx],
        borderWidth: 1,
        pointRadius: 0,
        fill: true
      }));

      datasets.push({
        label: "Other 14 Stocks (기타 14개 종목 합산)",
        data: wRows.map((r) => {
          let sumTop = 0;
          topHoldings.forEach((item) => { sumTop += getVal(r, item); });
          return +Math.max(0, 100 - sumTop).toFixed(2);
        }),
        backgroundColor: palette[6] + "88",
        borderColor: palette[6],
        borderWidth: 1,
        pointRadius: 0,
        fill: true
      });

      createOrUpdateChart("weightsAreaChart", {
        type: "line",
        data: { labels: wDates, datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: "index", intersect: false },
          plugins: {
            legend: { labels: { color: colors.text, font: { size: 11, weight: "600" } } },
            tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)}%` } }
          },
          scales: {
            x: { ticks: { color: colors.text, maxTicksLimit: 12 }, grid: { color: colors.grid } },
            y: {
              stacked: true,
              min: 0,
              max: 100,
              ticks: { color: colors.text, callback: (v) => v + "%" },
              grid: { color: colors.grid }
            }
          }
        }
      });
    }

    // OOS 2021-2025 Chart on QA page
    if (data.oosYear && document.getElementById("oosYearlyChart")) {
      const yrs = data.oosYear.map((r) => r.year);
      createOrUpdateChart("oosYearlyChart", {
        type: "bar",
        data: {
          labels: yrs,
          datasets: [
            {
              label: "S&P 500 TR (2021-2025 CAGR 14.43%)",
              data: data.oosYear.map((r) => parseFloat(r["S&P 500 TR"])),
              backgroundColor: colors.sp500,
              borderRadius: 4
            },
            {
              label: "New Top 20: Main XGBoost × M4 (CAGR 5.51%)",
              data: data.oosYear.map((r) => parseFloat(r["Main: XGBoost x M4, delta 3, 20% cap (new top 20)"])),
              backgroundColor: colors.main,
              borderRadius: 4
            },
            {
              label: "New Top 20: Equal-Weight (CAGR 6.28%)",
              data: data.oosYear.map((r) => parseFloat(r["Equal weight, new top 20"])),
              backgroundColor: colors.ew20,
              borderRadius: 4
            },
            {
              label: "Original 2010 Stocks (16 Surviving): XGBoost × M4 (CAGR 10.27%)",
              data: data.oosYear.map((r) => parseFloat(r["Original 2010 stocks kept (16), XGBoost x M4"])),
              backgroundColor: colors.rf,
              borderRadius: 4
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: "index", intersect: false },
          plugins: {
            legend: { labels: { color: colors.text, font: { size: 11, weight: "600" } } },
            tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label.split(" (")[0]}: ${ctx.parsed.y.toFixed(2)}%` } }
          },
          scales: {
            x: { ticks: { color: colors.text }, grid: { color: colors.grid } },
            y: { ticks: { color: colors.text, callback: (v) => v + "%" }, grid: { color: colors.grid } }
          }
        }
      });
    }
  }

  window.renderAllCharts = renderWealthAndDrawdownCharts;

  // --------------------------------------------------------------------------
  // 4. Dynamic Table Renderers (Methodology, Results, QA)
  // --------------------------------------------------------------------------
  function initTablesAndExplorers() {
    const data = window.PROJECT_CSV_DATA;
    if (!data) return;

    // Scale toggle buttons for Wealth Chart
    document.querySelectorAll(".scale-toggle-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".scale-toggle-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        renderWealthAndDrawdownCharts();
      });
    });

    // A. FF3 Top 20 / All 131 Table (Methodology Page)
    const ff3Body = document.getElementById("ff3Top20Body");
    if (ff3Body && data.ff3Top20) {
      const holdingsByPermno = {};
      if (data.holdingsSummary) {
        data.holdingsSummary.forEach((h) => {
          if (h.permno) holdingsByPermno[String(h.permno)] = h;
          if (h.ticker) holdingsByPermno[h.ticker] = h;
          holdingsByPermno[h.H1] = h;
        });
      }

      let currentScope = "top20";
      let currentQuery = "";

      const renderFF3 = () => {
        const q = currentQuery.trim().toLowerCase();
        const sourceList = (currentScope === "all131" && data.ff3All131) ? data.ff3All131 : data.ff3Top20;
        const rows = sourceList.filter((r) =>
          !q ||
          String(r.TICKER).toLowerCase().includes(q) ||
          String(r.COMNAM).toLowerCase().includes(q) ||
          String(r.PERMNO).includes(q)
        );
        ff3Body.innerHTML = rows.map((r, idx) => {
          const h = holdingsByPermno[String(r.PERMNO)] || holdingsByPermno[r.TICKER] || {};
          const rawAvgW = r.avgWeight ?? h["Avg Weight (%)"] ?? h["평균 비중(%)"];
          const avgW = rawAvgW !== undefined && rawAvgW !== null ? fmtNum(rawAvgW, 2, "%") : "— (비선정)";
          const bMkt = r.b_mkt ?? r.beta_mkt;
          const sSmb = r.s_smb ?? r.beta_smb;
          const hHml = r.h_hml ?? r.beta_hml;
          const r2Val = r.r2 ?? r.R2;
          const shVal = r.sharpe ?? r.Sharpe;
          const rankNum = r.rank || (idx + 1);
          return `
            <tr class="${r.TICKER === "AAPL" ? "highlight-row" : ""}">
              <td class="num">${rankNum}</td>
              <td><strong>${r.TICKER}</strong></td>
              <td>${r.COMNAM}</td>
              <td class="num">${r.PERMNO}</td>
              <td class="num pos">${fmtNum(r.t_alpha, 2)}</td>
              <td class="num pos">${fmtNum(parseFloat(r.alpha_ann) * 100, 2, "%")}</td>
              <td class="num">${fmtNum(bMkt, 2)}</td>
              <td class="num">${fmtNum(sSmb, 2)}</td>
              <td class="num">${fmtNum(hHml, 2)}</td>
              <td class="num">${fmtNum(parseFloat(r2Val) * 100, 1, "%")}</td>
              <td class="num">${fmtNum(shVal, 2)}</td>
              <td class="num"><strong>${avgW}</strong></td>
            </tr>
          `;
        }).join("");
      };

      renderFF3();
      const searchInput = document.getElementById("ff3SearchInput");
      if (searchInput) {
        searchInput.addEventListener("input", (e) => {
          currentQuery = e.target.value;
          renderFF3();
        });
      }
      document.querySelectorAll(".ff3-scope-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
          document.querySelectorAll(".ff3-scope-btn").forEach((b) => b.classList.remove("active"));
          btn.classList.add("active");
          currentScope = btn.dataset.scope || "top20";
          renderFF3();
        });
      });
    }

    // B. 40 Combinations Heatmap Explorer (Results Page)
    const comboTableBody = document.getElementById("combo40Body");
    if (comboTableBody && data.cagr40 && data.sharpe40 && data.turnover40) {
      const methods = ["M1_SMB", "M1_SMB_fixed", "M2_HML", "M2_HML_fixed", "M3_SMB_HML", "M3_SMB_HML_fixed", "M4_Identity", "M5_Proportional"];
      const renderComboMatrix = (metric = "cagr") => {
        const source = metric === "cagr" ? data.cagr40 : metric === "sharpe" ? data.sharpe40 : data.turnover40;
        const unit = metric === "sharpe" ? "" : "%";

        comboTableBody.innerHTML = source.map((row) => {
          const modelName = row.model || row.H1;
          const cells = methods.map((m) => {
            const val = parseFloat(row[m]);
            let badgeClass = "";
            if (metric === "cagr") {
              if (val >= 18) badgeClass = "pos";
              else if (val < 13.5) badgeClass = "neg";
            } else if (metric === "sharpe") {
              if (val >= 1.05) badgeClass = "pos";
              else if (val < 0.85) badgeClass = "neg";
            }
            const isMain = modelName === "XGBoost" && m === "M4_Identity";
            return `<td class="num ${badgeClass}" style="${isMain ? "background:rgba(29,78,216,0.16);font-weight:800;border:2px solid var(--accent-primary);" : ""}">${val.toFixed(2)}${unit}</td>`;
          }).join("");

          return `<tr><td><strong>${modelName}</strong></td>${cells}</tr>`;
        }).join("");
      };

      renderComboMatrix("cagr");
      document.querySelectorAll(".combo-metric-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
          document.querySelectorAll(".combo-metric-btn").forEach((b) => b.classList.remove("active"));
          btn.classList.add("active");
          renderComboMatrix(btn.dataset.metric);
        });
      });
    }

    // C. Holdings Summary Table (Results Page)
    const holdingsBody = document.getElementById("holdingsTableBody");
    if (holdingsBody && data.holdingsSummary) {
      holdingsBody.innerHTML = data.holdingsSummary.map((r, i) => {
        const tickerLabel = r.ticker ? `${r.ticker} <span style="font-size:0.78rem;color:var(--text-muted);">(${r.permno})</span>` : r.H1;
        const avgW = r["Avg Weight (%)"] ?? r["평균 비중(%)"];
        const stdW = r["Std Weight (%)"];
        const minW = r["Min Weight (%)"];
        const maxW = r["Max Weight (%)"];
        const mHeld = r["Months Held (>0.1%)"];
        const pctHeld = r["보유 개월 비율(%)"];
        const mCap = r["Months at Cap (20%)"];
        const pctCap = r["20% 상한 도달 비율(%)"];
        const isAAPL = String(r.H1).startsWith("AAPL");

        return `
          <tr class="${isAAPL ? "highlight-row" : ""}">
            <td class="num">${i + 1}</td>
            <td><strong>${tickerLabel}</strong></td>
            <td>${r.Company || "—"}</td>
            <td class="num"><strong>${fmtNum(avgW, 2, "%")}</strong></td>
            <td class="num">${fmtNum(stdW, 2, "%")}</td>
            <td class="num">${fmtNum(minW, 2, "%")}</td>
            <td class="num">${fmtNum(maxW, 2, "%")}</td>
            <td class="num">${mHeld !== undefined ? `${mHeld} / 120 (${fmtNum(pctHeld, 1, "%")})` : fmtNum(pctHeld, 1, "%")}</td>
            <td class="num">${mCap !== undefined ? `${mCap} / 120 (${fmtNum(pctCap, 1, "%")})` : fmtNum(pctCap, 1, "%")}</td>
          </tr>
        `;
      }).join("");
    }

    // D. Multiple Testing 40 Combos Table (QA Page)
    const mtBody = document.getElementById("mt40TableBody");
    if (mtBody && data.multipleTesting40) {
      mtBody.innerHTML = data.multipleTesting40.slice(0, 15).map((r, idx) => {
        const cleanName = r.combo.replace("_net", "").replace("_", " × ");
        const isSig = parseFloat(r.p_two_sided) < 0.05;
        return `
          <tr class="${idx === 0 ? "highlight-row" : ""}">
            <td class="num">${idx + 1}</td>
            <td><strong>${cleanName}</strong></td>
            <td class="num ${isSig ? "pos" : ""}">${fmtNum(r.t, 2)}</td>
            <td class="num ${isSig ? "pos" : ""}">${fmtNum(r.p_two_sided, 4)}</td>
            <td class="num">${fmtNum(r.p_bonferroni, 3)}</td>
            <td class="num">${fmtNum(r.p_holm, 3)}</td>
            <td class="num">${fmtNum(r.p_BH, 4)}</td>
          </tr>
        `;
      }).join("");
    }

    // E. OOS 2010-2020 Selected Top 20 Table (QA Page)
    const oosTop20Body = document.getElementById("oosTop20Body");
    if (oosTop20Body && data.oosTop20) {
      oosTop20Body.innerHTML = data.oosTop20.map((r) => `
        <tr>
          <td class="num">${r.rank}</td>
          <td><strong>${r.TICKER}</strong></td>
          <td>${r.COMNAM}</td>
          <td class="num">${r.PERMNO}</td>
          <td class="num pos">${fmtNum(r.t_alpha, 2)}</td>
          <td class="num pos">${fmtNum(parseFloat(r.alpha_ann) * 100, 2, "%")}</td>
          <td class="num">${fmtNum(r.s_smb, 2)}</td>
          <td class="num">${fmtNum(r.h_hml, 2)}</td>
        </tr>
      `).join("");
    }

    // F. Q&A Accordion & Category Filter (QA Page)
    const qaItems = document.querySelectorAll(".qa-item");
    qaItems.forEach((item) => {
      const trigger = item.querySelector(".qa-trigger");
      if (trigger) {
        trigger.addEventListener("click", () => {
          item.classList.toggle("open");
        });
      }
    });

    document.querySelectorAll(".qa-filter-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".qa-filter-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const cat = btn.dataset.cat;
        qaItems.forEach((item) => {
          if (cat === "all" || item.dataset.cat === cat) {
            item.style.display = "block";
          } else {
            item.style.display = "none";
          }
        });
      });
    });

    const expandAllBtn = document.getElementById("qaExpandAll");
    const collapseAllBtn = document.getElementById("qaCollapseAll");
    if (expandAllBtn) {
      expandAllBtn.addEventListener("click", () => qaItems.forEach((i) => i.classList.add("open")));
    }
    if (collapseAllBtn) {
      collapseAllBtn.addEventListener("click", () => qaItems.forEach((i) => i.classList.remove("open")));
    }
  }

  // --------------------------------------------------------------------------
  // 5. Bootstrapping on DOMContentLoaded
  // --------------------------------------------------------------------------
  document.addEventListener("DOMContentLoaded", () => {
    initThemeAndNav();
    initTablesAndExplorers();
    renderWealthAndDrawdownCharts();

    if (typeof renderMathInElement === "function") {
      renderMathInElement(document.body, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "\\[", right: "\\]", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\(", right: "\\)", display: false }
        ],
        ignoredClasses: ["currency"],
        throwOnError: false
      });
    }
  });
})();
