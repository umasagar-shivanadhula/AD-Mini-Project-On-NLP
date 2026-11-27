const $ = (id) => document.getElementById(id);
const tabs = [...document.querySelectorAll(".tab")];
let current = null;
let requestId = 0; // lets us ignore replies that arrive after the user moved on

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text; // textContent keeps user text safe
  if (className) node.className = className;
  return node;
}

function barRow(label, fraction, value) {
  const row = el("div", undefined, "row");
  const bar = el("div", undefined, "bar");
  const fill = el("span");
  fill.style.width = `${Math.round(fraction * 100)}%`;
  bar.append(fill);
  row.append(el("span", label), bar, el("span", value));
  return row;
}

// Custom views. A module without one here gets the generic view below.
const renderers = {
  summarize(data) {
    const note = `${data.original_words} words reduced to ${data.summary_words}.` +
      (data.truncated ? " Only the first part of the text was used." : "");
    return [el("h2", "Summary"), el("p", data.summary), el("p", note)];
  },
  keywords(data) {
    const max = data.keywords[0].count;
    return [el("h2", "Keywords"),
      ...data.keywords.map((k) => barRow(k.term, k.count / max, String(k.count)))];
  },
  classify(data) {
    return [el("h2", `Category: ${data.category}`),
      ...data.scores.map((s) => barRow(s.label, s.score, `${(s.score * 100).toFixed(1)}%`))];
  },
};

function genericView(data) {
  const list = el("dl");
  for (const [key, value] of Object.entries(data)) {
    list.append(el("dt", key.replaceAll("_", " ")),
      el("dd", typeof value === "object" ? JSON.stringify(value) : String(value)));
  }
  return [el("h2", "Result"), list];
}

function show(nodes, isError = false) {
  const box = $("result");
  box.replaceChildren(...nodes);
  box.classList.toggle("error", isError);
  box.hidden = false;
}

function select(tab) {
  current = tab.dataset.name;
  requestId += 1;
  tabs.forEach((t) => {
    t.classList.toggle("active", t === tab);
    t.setAttribute("aria-pressed", String(t === tab));
  });
  $("description").textContent = tab.dataset.description;
  $("result").hidden = true;
  $("run").disabled = false;
}

const updateCount = () => {
  $("count").textContent = `${$("text").value.length.toLocaleString("en-US")} / 20,000`;
};

tabs.forEach((tab) => tab.addEventListener("click", () => select(tab)));
$("text").addEventListener("input", updateCount);

$("file").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const content = await file.text();
  $("text").value = content.slice(0, 20000);
  updateCount();
  if (content.length > 20000) show([el("p", "File trimmed to 20,000 characters.")]);
});

$("form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const name = current;
  const id = ++requestId;
  $("run").disabled = true;
  show([el("p", "Analysing... the first run of a model can take a few minutes.")]);
  try {
    const response = await fetch(`/api/${name}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: $("text").value }),
    });
    const data = await response.json();
    if (id !== requestId) return;
    if (!response.ok) throw new Error(data.error || "Request failed.");
    show((renderers[name] || genericView)(data));
  } catch (error) {
    if (id === requestId) show([el("p", error.message)], true);
  } finally {
    if (id === requestId) $("run").disabled = false;
  }
});

if (tabs.length) select(tabs[0]);
