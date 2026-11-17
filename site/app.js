(() => {
  "use strict";
  const data = window.CLUTCH_DATA;
  const notice = document.getElementById("notice");
  function report(message, error) {
    notice.textContent = message;
    notice.hidden = false;
    if (error) console.error(error);
  }
  if (!data || !Array.isArray(data.players) || !data.players.length) {
    report("Interactive data is unavailable. The static rankings and factor checklist remain readable.");
    return;
  }
  const byKey = new Map(data.players.map(player => [player.key, player]));
  if (byKey.size !== data.players.length || data.players.some(player =>
    !/^(batting|bowling)-\d+$/.test(player.key) ||
    !Number.isFinite(player.score) || player.points.length !== 3 ||
    player.points.some(value => !Number.isFinite(value)) ||
    Math.abs(player.points.reduce((a, b) => a + b, 0) - player.score) > 1e-8)) {
    report("The interactive snapshot failed its consistency check. Use the static tables.", new Error("Invalid CLUTCH snapshot."));
    return;
  }
  const state = {role: "batting", scope: "top", query: "", player: data.players.find(p => p.role === "batting" && p.rank === 1).key, season: "all", allMatches: false};
  const colors = ["#3864a8", "#c02f74", "#70436b"];
  const pointNames = ["Runs / wickets", "Skill / errors", "Big matches"];
  const roleWord = role => role === "batting" ? "batters" : "bowlers";
  const fmt = (value, places = 2) => value === null || value === undefined ? "Not available" :
    Number(value).toLocaleString("en-GB", {minimumFractionDigits: places, maximumFractionDigits: places});
  const normalize = value => String(value).normalize("NFKD").toLowerCase();
  const plotAvailable = typeof window.Plotly !== "undefined";
  const plotConfig = {responsive: true, displaylogo: false, scrollZoom: false, showLink: false,
    modeBarButtonsToRemove: ["select2d", "lasso2d"]};
  const baseLayout = {
    paper_bgcolor: "#fffafd", plot_bgcolor: "#fffafd",
    font: {family: "Segoe UI, sans-serif", color: "#352338", size: 12},
    hoverlabel: {font: {size: 13}, bgcolor: "#fffafd", bordercolor: "#d9adc3"},
  };
  let chartRole = null;
  let lastVisible = [];
  let profileGeneration = 0;
  const search = document.getElementById("player-search");
  const scope = document.getElementById("ranking-scope");
  const selector = document.getElementById("player-select");
  const season = document.getElementById("match-season");
  const matchPanel = document.getElementById("match-evidence");
  document.documentElement.classList.add("js");
  document.querySelectorAll("button:disabled, input:disabled, select:disabled").forEach(control => {
    control.disabled = false;
  });

  function element(tag, value, className) {
    const node = document.createElement(tag);
    if (value !== undefined) node.textContent = value;
    if (className) node.className = className;
    return node;
  }

  function drawComponents() {
    if (chartRole === state.role) return;
    chartRole = state.role;
    const players = data.players.filter(p => p.role === state.role && p.rank <= 10);
    const fallback = document.getElementById("component-fallback");
    fallback.src = `${state.role}-scores.svg`;
    fallback.alt = `Top ten ${roleWord(state.role)} with the three weighted score parts`;
    document.getElementById("component-title").textContent = `Top 10 ${state.role} score breakdown`;
    if (!plotAvailable) return;
    const target = document.getElementById("component-chart");
    target.hidden = false;
    const traces = pointNames.map((name, i) => ({
      type: "bar", orientation: "h", name,
      x: players.map(p => p.points[i]), y: players.map(p => p.name),
      customdata: players.map(p => [p.key, p.score]),
      marker: {color: colors[i]},
      hovertemplate: "%{y}<br>" + name + ": %{x:.2f} points<br>Total: %{customdata[1]:.2f}<extra></extra>",
    }));
    window.Plotly.react(target, traces, {
      ...baseLayout, barmode: "stack", showlegend: false,
      margin: {l: window.innerWidth < 720 ? 133 : 151, r: 12, t: 12, b: 45},
      xaxis: {range: [0, 100], title: {text: "Final score / 100", font: {size: 12}}, gridcolor: "#efdee7", tickvals: [0, 25, 50, 75, 100], fixedrange: false},
      yaxis: {autorange: "reversed", automargin: true, fixedrange: true, tickfont: {size: 11}},
    }, plotConfig).then(() => {
      fallback.hidden = true;
      target.removeAllListeners("plotly_click");
      target.on("plotly_click", event => {
        const key = event.points[0]?.customdata?.[0];
        if (byKey.has(key)) selectPlayer(key);
      });
    }).catch(error => {
      target.hidden = true;
      fallback.hidden = false;
      report("Interactive charts are unavailable. The static charts and tables remain available.", error);
    });
  }

  function renderRankings() {
    lastVisible = data.players.filter(player => player.role === state.role &&
      (state.scope === "all" || player.rank <= 10) &&
      normalize(player.name).includes(normalize(state.query)));
    const visible = new Set(lastVisible.map(player => player.key));
    document.querySelectorAll(".role-table").forEach(table => {
      table.hidden = table.dataset.role !== state.role;
      table.querySelectorAll("tbody tr").forEach(row => {
        row.hidden = !visible.has(row.dataset.player);
        row.classList.toggle("selected", row.dataset.player === state.player);
      });
    });
    const total = data.players.filter(player => player.role === state.role).length;
    document.getElementById("result-count").textContent =
      `${lastVisible.length} of ${total} ranked ${roleWord(state.role)} shown. Ranks 1–10 are the selections.`;
    document.getElementById("empty-ranking").hidden = lastVisible.length !== 0;
    document.querySelectorAll("[data-role-button]").forEach(button => {
      button.setAttribute("aria-pressed", String(button.dataset.roleButton === state.role));
    });
    drawComponents();
  }

  function renderChoices() {
    selector.replaceChildren(...data.players.filter(p => p.role === state.role).map(player => {
      const option = element("option", `#${player.rank} · ${player.name}`);
      option.value = player.key;
      return option;
    }));
    selector.value = state.player;
  }

  function figures(player, match) {
    if (match.runs === null || match.balls === null) return "Unavailable";
    return player.role === "batting" ? `${fmt(match.runs, 0)} (${match.balls} legal balls)` :
      `${match.wickets}/${fmt(match.runs, 0)} (${match.balls} legal balls)`;
  }

  function renderMatches(player) {
    const generation = ++profileGeneration;
    const available = player.matches.length > 0;
    matchPanel.hidden = !available;
    document.getElementById("match-unavailable").hidden = available;
    if (!available) return;
    const matches = player.matches.filter(match => state.season === "all" || match.date.startsWith(state.season));
    const ordered = [...matches].sort((a, b) => Math.abs(b.effect) - Math.abs(a.effect));
    const shown = state.allMatches ? ordered : ordered.slice(0, 8);
    document.getElementById("match-rows").replaceChildren(...shown.map(match => {
      const row = element("tr");
      [match.date, match.opponent, match.stage, figures(player, match),
        (match.effect >= 0 ? "+" : "") + fmt(match.effect)].forEach(value => row.append(element("td", value)));
      return row;
    }));
    const more = document.getElementById("more-matches");
    more.hidden = ordered.length <= 8;
    more.textContent = state.allMatches ? "Show the eight largest effects" : `Show all ${ordered.length} matches`;
    const fallback = document.getElementById("match-fallback");
    fallback.src = `${player.key}-matches.svg`;
    fallback.alt = `All scored match influences for ${player.name}`;
    fallback.hidden = state.season !== "all";
    const target = document.getElementById("match-chart");
    if (!plotAvailable) {
      target.hidden = true;
      return;
    }
    fallback.hidden = true;
    target.hidden = false;
    window.Plotly.react(target, [{
      type: "bar", x: matches.map(m => m.date), y: matches.map(m => m.effect),
      marker: {color: matches.map(m => m.effect >= 0 ? colors[0] : colors[1])},
      customdata: matches.map(m => [m.opponent, m.stage, figures(player, m)]),
      hovertemplate: "%{x|%d %b %Y}<br>v %{customdata[0]} · %{customdata[1]}<br>%{customdata[2]}<br>Score effect: %{y:+.2f}<extra></extra>",
    }], {
      ...baseLayout, showlegend: false,
      margin: {l: 53, r: 18, t: 18, b: 48},
      xaxis: {type: "date", gridcolor: "#f3e7ed", automargin: true},
      yaxis: {range: data.match_effect_extent, title: {text: "Score points", font: {size: 12}},
        zeroline: true, zerolinecolor: "#b89aaa", gridcolor: "#f0e1e9", automargin: true},
    }, plotConfig).then(() => {
      if (generation === profileGeneration) fallback.hidden = true;
    }).catch(error => {
      if (generation !== profileGeneration) return;
      target.hidden = true;
      fallback.hidden = state.season !== "all";
      report("The match chart is unavailable. The table still shows the selected evidence.", error);
    });
  }

  function selectPlayer(key) {
    const player = byKey.get(key);
    if (!player) {
      report("That player is not in the current ranked population.");
      return;
    }
    const roleChanged = state.role !== player.role;
    state.player = key;
    state.role = player.role;
    state.season = "all";
    state.allMatches = false;
    if (roleChanged) renderChoices();
    selector.value = key;
    document.getElementById("profile-rank").textContent = `${player.role.toUpperCase()} / #${player.rank}`;
    document.getElementById("profile-name").textContent = player.name;
    document.getElementById("profile-score").textContent = fmt(player.score);
    const routes = {
      both_channels: "Qualifies through pressure-ball and big-match evidence.",
      ball_channel: "Qualifies through pressure-ball evidence.",
      high_match_channel: "Qualifies through big-match evidence.",
    };
    document.getElementById("profile-route").textContent = routes[player.route] || player.route.replaceAll("_", " ");
    const rateLabel = player.role === "batting" ? "pressure SR / 100 legal balls" : "pressure runs / 6 legal balls";
    const metrics = [
      [fmt(player.pressure_balls, 0), "pressure balls"],
      [String(player.pressure_matches), "pressure matches"],
      [String(player.big_matches), "big-match games"],
      [`${fmt(player.repeat_rate, 1)}%`, "top-ten repeat rate"],
      [fmt(player.pressure_rate, 2), rateLabel],
      [fmt(player.big_balls, 0), "big-match legal balls"],
      [fmt(player.ordinary_events, 0), "ordinary-play reference events"],
      [fmt(player.role_matches, 0), "scored role appearances"],
    ];
    document.getElementById("profile-metrics").replaceChildren(...metrics.map(([value, label]) => {
      const cell = element("div");
      cell.append(element("strong", value), element("span", label));
      return cell;
    }));
    document.getElementById("profile-parts").replaceChildren(
      ...pointNames.map((name, i) => element("span", `${name}: ${fmt(player.points[i])} points`)));
    document.getElementById("profile-uncertainty").textContent =
      `95% score range: ${fmt(player.score_interval[0])} to ${fmt(player.score_interval[1])}. ` +
      `Middle 90% of joint redraw ranks: ${fmt(player.rank_interval[0], 0)}–${fmt(player.rank_interval[1], 0)}. Comparison averages stay fixed.`;
    document.getElementById("profile-warning").textContent = player.pressure_balls < 60 ?
      "The pressure-ball sample is small. This player qualifies through big-match evidence; the two sample routes are shown separately." :
      "A rank reflects this dataset and these rules, not a certain ordering of players. Different competition mixes still matter.";
    const years = [...new Set(player.matches.map(m => m.date.slice(0, 4)))].sort();
    const all = element("option", "All seasons");
    all.value = "all";
    season.replaceChildren(all, ...years.map(year => {
      const option = element("option", year);
      option.value = year;
      return option;
    }));
    renderRankings();
    renderMatches(player);
  }

  document.querySelectorAll(".player-name").forEach(cell => {
    const key = cell.parentElement.dataset.player;
    const button = element("button", cell.textContent, "player-link");
    button.type = "button";
    button.setAttribute("aria-label", `View evidence for ${cell.textContent}`);
    button.addEventListener("click", () => selectPlayer(key));
    cell.replaceChildren(button);
  });
  document.querySelectorAll("[data-role-button]").forEach(button => {
    button.addEventListener("click", () => selectPlayer(data.players.find(p => p.role === button.dataset.roleButton && p.rank === 1).key));
  });
  search.addEventListener("input", () => {
    state.query = search.value.trim();
    if (state.query) {
      state.scope = "all";
      scope.value = "all";
    }
    renderRankings();
    if (lastVisible.length === 1) selectPlayer(lastVisible[0].key);
  });
  scope.addEventListener("change", () => { state.scope = scope.value; renderRankings(); });
  selector.addEventListener("change", () => selectPlayer(selector.value));
  season.addEventListener("change", () => {
    state.season = season.value;
    state.allMatches = false;
    renderMatches(byKey.get(state.player));
  });
  document.getElementById("more-matches").addEventListener("click", () => {
    state.allMatches = !state.allMatches;
    renderMatches(byKey.get(state.player));
  });
  document.getElementById("expand-factors").addEventListener("click", event => {
    const groups = [...document.querySelectorAll(".factor-group")];
    const open = groups.some(group => !group.open);
    groups.forEach(group => { group.open = open; });
    event.currentTarget.textContent = open ? "Collapse all groups" : "Expand all groups";
  });
  document.getElementById("download").addEventListener("click", () => {
    function csv(value) {
      let text = String(value);
      if (typeof value === "string" && /^[=+@-]/.test(text.trim())) text = "'" + text;
      return `"${text.replaceAll('"', '""')}"`;
    }
    const lines = [
      ["role", "rank", "player", "score", "pressure_balls", "big_match_games", "top10_repeat_pct"],
      ...lastVisible.map(p => [p.role, p.rank, p.name, p.score, p.pressure_balls, p.big_matches, p.repeat_rate]),
    ];
    const url = URL.createObjectURL(new Blob([lines.map(row => row.map(csv).join(",")).join("\r\n")], {type: "text/csv;charset=utf-8"}));
    const link = element("a");
    link.href = url;
    link.download = `clutch-${state.role}-rankings.csv`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  if (!plotAvailable) report("Interactive charts are unavailable. Static charts, rankings and match tables remain available.");
  renderChoices();
  selectPlayer(state.player);
})();
