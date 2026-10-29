"use strict";

// Player Two companion window. Live events come over a WebSocket from the session;
// ?demo plays a scripted session so the window can be previewed without a game.

const $ = (id) => document.getElementById(id);
const app = $("app");
const params = new URLSearchParams(location.search);
if (window.P2_DEMO) params.set("demo", window.P2_DEMO);
if (params.has("captions")) document.body.classList.add("captions");

const STATUS = {
  watching: "Watching", listening: "Listening", thinking: "Looking",
  acting: "Playing", speaking: "Speaking", paused: "Paused",
};
const GLYPH = {
  a: ["A", "a round"], b: ["B", "b round"], x: ["X", "x round"], y: ["Y", "y round"],
  up: ["↑", "round"], down: ["↓", "round"], left: ["←", "round"],
  right: ["→", "round"], lb: ["LB", ""], rb: ["RB", ""], lt: ["LT", ""], rt: ["RT", ""],
  start: ["≡", "round"], back: ["⧉", "round"], guide: ["⌂", "round"],
  ls: ["L3", ""], rs: ["R3", ""],
};
const VERBOSITY_HINT = {
  terse: "Bare facts, for fast screen reader users.",
  quiet: "Only dialogue and choices.",
  normal: "What's new and matters, in a few words.",
  detailed: "A little more about each scene.",
};

let socket = null;
let connected = false;
let pressTimer = null;
let lastPress = 0;
let liveTimer = null;
let talkKey = "F8";
let talkToggle = false;

// What a screen reader hears from this window: the talk turns, not every flicker between
// watching and speaking (the voice already covers those).
const SPOKEN_STATES = { listening: "Listening", thinking: "Got it", acting: "Playing", paused: "Paused" };
let lastState = "";
function setState(state) {
  app.dataset.state = state;
  $("status-text").textContent = STATUS[state] || state;
  // "Looking" also means narrating a new screen; only say "Got it" after the player spoke.
  const say = state === "thinking" && lastState !== "listening" ? "" : SPOKEN_STATES[state];
  if (state !== lastState && say) $("announce").textContent = say;
  lastState = state;
}

function setCaption(text, who = "Player Two") {
  const now = document.querySelector(".now");
  now.classList.toggle("you", who === "You");
  $("now-who").textContent = who;
  const cap = $("caption");
  cap.replaceChildren();
  const span = document.createElement("span");
  span.className = "fresh";
  span.textContent = text;
  cap.append(span);
}

function setGame(name, found) {
  $("game").classList.toggle("none", !found);
  $("game-name").textContent = found ? (name || "Your game") : "No game open yet";
  $("screen-empty-text").textContent = found ? "Waiting for the game"
    : "Start a game and I'll find it";
  if (!found) {
    $("frame").hidden = true;
    $("screen-empty").hidden = false;
    $("live").hidden = true;
  }
}

function glyph(name) {
  const [label, cls] = GLYPH[name] || [name.toUpperCase(), ""];
  const el = document.createElement("span");
  el.className = `btn ${cls}`.trim();
  el.textContent = label;
  el.title = name;
  return el;
}

function addItem(kind, tag, content) {
  const li = document.createElement("li");
  li.className = `item ${kind}`;
  const t = document.createElement("span");
  t.className = "tag";
  t.textContent = tag;
  const body = document.createElement("div");
  body.className = "body";
  if (typeof content === "string") body.textContent = content;
  else body.append(...content);
  li.append(t, body);
  const feed = $("feed");
  feed.append(li);
  while (feed.children.length > 80) feed.firstElementChild.remove();
  const wrap = feed.parentElement; // the scroller
  wrap.scrollTo({ top: wrap.scrollHeight, behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
}

// Notices: the welcome on first run, and anything that needs fixing.
function notice(id, kind, title, lines, dismiss) {
  if (document.getElementById(`notice-${id}`)) return;
  const box = document.createElement("div");
  box.className = `notice ${kind}`;
  box.id = `notice-${id}`;
  box.setAttribute("role", kind === "warn" ? "alert" : "note");
  const h = document.createElement("h3");
  h.textContent = title;
  box.append(h);
  for (const line of lines) {
    const p = document.createElement(Array.isArray(line) ? "ul" : "p");
    if (Array.isArray(line)) {
      for (const item of line) {
        const li = document.createElement("li");
        li.textContent = item;
        p.append(li);
      }
    } else p.textContent = line;
    box.append(p);
  }
  if (dismiss) {
    const b = document.createElement("button");
    b.className = "notice-close";
    b.textContent = dismiss;
    b.addEventListener("click", () => {
      box.remove();
      if (id === "welcome") localStorage.setItem("p2-welcomed", "1");
    });
    box.append(b);
  }
  $("notices").append(box);
}

function showPresses(buttons, note) {
  const box = $("presses");
  box.replaceChildren(...buttons.map(glyph));
  clearTimeout(pressTimer);
  pressTimer = setTimeout(() => box.replaceChildren(), 2600);
  const parts = buttons.map(glyph);
  if (note) {
    const n = document.createElement("span");
    n.className = "note";
    n.textContent = note;
    parts.push(n);
  }
  // Presses that come in quick succession share one row.
  const last = $("feed").lastElementChild;
  if (last && last.classList.contains("act") && !note && Date.now() - lastPress < 6000) {
    last.querySelector(".body").append(...parts);
  } else {
    addItem("act", "Press", parts);
  }
  lastPress = Date.now();
}

function applySettings(ev) {
  for (const group of document.querySelectorAll(".seg[data-setting]")) {
    const value = ev[group.dataset.setting];
    if (value === undefined) continue;
    for (const b of group.querySelectorAll("[role=radio]")) {
      const on = b.dataset.value === value;
      b.setAttribute("aria-checked", String(on));
      b.tabIndex = on ? 0 : -1;
    }
  }
  if (ev.verbosity) $("verbosity-hint").textContent = VERBOSITY_HINT[ev.verbosity] || "";
  const langs = ev.choices?.language;
  if (langs && $("language").options.length !== langs.length) {
    $("language").replaceChildren(...langs.map((l) => new Option(l, l)));
  }
  if (ev.language) $("language").value = ev.language;
  if (ev.speech_rate) {
    $("speech-rate").value = ev.speech_rate;
    $("rate-out").textContent = `${Number(ev.speech_rate).toFixed(2)}×`;
  }
  if (ev.output) $("output").value = ev.output;
  if (ev.team_up !== undefined) $("team-up").checked = ev.team_up;
  if (ev.hands_free !== undefined) setHandsFree(ev.hands_free);
  if (ev.paused !== undefined) $("paused").checked = ev.paused;
  if (ev.companion) $("companion").checked = ev.companion !== "off";
}

function setHandsFree(on) {
  $("hands-free").checked = on;
  $("type-box").placeholder = on ? "Say “Player Two, open the menu”, or type here" : "Or type: open the menu";
}

function handle(ev) {
  switch (ev.kind) {
    case "ready":
      setGame(ev.game, ev.found !== false);
      if (ev.talk_key) { talkKey = ev.talk_key; $("talk-key").textContent = ev.talk_key; }
      talkToggle = Boolean(ev.talk_toggle);
      $("talk-verb").textContent = talkToggle ? "Press" : "Hold";
      setHandsFree(Boolean(ev.hands_free));
      if (ev.first_run && !localStorage.getItem("p2-welcomed")) {
        notice("welcome", "hello", "Hi, I'm Player Two", [
          ev.hands_free
            ? "I tell you what's happening in your game, and I play when you ask. I'm listening: start with my name, like “Player Two, open the menu”. Some things to try:"
            : `I tell you what's happening in your game, and I play when you ask. Hold ${talkKey} and talk, then let go. Some things to try:`,
          ["“What's on screen?”", "“Open the menu”", "“Heal whoever is hurt”",
            "“Repeat that”", "“Stop” (I stop talking)", "“Less detail”"],
        ], "Got it");
      }
      for (const [i, p] of (ev.problems || []).entries()) notice(`problem-${i}`, "warn", "Needs a fix", [p]);
      if (ev.talk_key_bound === false && !ev.hands_free) {
        notice("talk-key", "warn", "No talk key yet", [
          "Run player-two setup in a terminal to make F8 your talk key. Until then, type to me in the box below, or use the talk button.",
        ]);
        $("type-box").focus();
      }
      break;
    case "game":
      setGame(ev.name, ev.found);
      break;
    case "status":
      setState(ev.state);
      break;
    case "frame": {
      const img = $("frame");
      img.onload = () => {
        // Match the box to the game's shape so markers land where things really are.
        $("screen").style.aspectRatio = `${img.naturalWidth} / ${img.naturalHeight}`;
      };
      img.src = `data:image/jpeg;base64,${ev.jpeg}`;
      img.hidden = false;
      $("screen-empty").hidden = true;
      $("live").hidden = false;
      clearTimeout(liveTimer);
      liveTimer = setTimeout(() => { $("live").hidden = true; }, 20000);
      break;
    }
    case "hearing":
      $("hearing").hidden = false;
      $("hearing").textContent = ev.text;
      break;
    case "heard":
      $("hearing").hidden = true;
      if (ev.text) {
        setCaption(ev.text, "You");
        addItem("you", "You", ev.text);
      }
      break;
    case "say":
      setCaption(ev.text);
      addItem(ev.source === "narrator" ? "narr" : "p2", "P2", ev.text);
      break;
    case "ask":
      setCaption(ev.text);
      addItem("ask", "P2", ev.text);
      break;
    case "press":
      showPresses(ev.buttons || []);
      break;
    case "hold":
      showPresses([ev.input === "left_stick" ? "ls" : ev.input],
        `${ev.direction || ""} ${(ev.ms / 1000).toFixed(1)}s`.trim());
      break;
    case "walk":
      showPresses(["ls"], ev.direction ? `walking ${ev.direction}`
        : `walking to ${ev.target || "it"}`);
      break;
    case "where": {
      const box = $("marks");
      box.replaceChildren(...(ev.points || []).map(([x, y], i) => {
        const dot = document.createElement("span");
        dot.className = "mark";
        dot.style.left = `${x * 100}%`;
        dot.style.top = `${y * 100}%`;
        dot.style.animationDelay = `${i * 0.28}s`;
        return dot;
      }));
      clearTimeout(window.markTimer);
      window.markTimer = setTimeout(() => box.replaceChildren(), 6000);
      break;
    }
    case "settings":
      applySettings(ev);
      break;
    case "error":
      addItem("error", "Oops", ev.message || "Something went wrong");
      break;
    default:
  }
}

function send(command) {
  if (socket && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ command }));
}

function connect() {
  socket = new WebSocket(`ws://${location.host}/events`);
  socket.onopen = () => {
    connected = true;
    $("gone").hidden = true;
  };
  socket.onmessage = (m) => handle(JSON.parse(m.data));
  socket.onclose = () => {
    setState("paused");
    $("status-text").textContent = "Reconnecting";
    // Gone for a moment is a hiccup; gone for longer means Player Two stopped.
    if (connected) setTimeout(() => { if (socket.readyState !== WebSocket.OPEN) $("gone").hidden = false; }, 2500);
    connected = false;
    setTimeout(connect, 1500);
  };
}

// Talk: hold the button (mouse, touch, Space or Enter), or the talk key while this
// window has focus.
const talk = $("talk");
let talking = false;
function talkStart() { if (!talking) { talking = true; send("talk-start"); } }
function talkStop() { if (talking) { talking = false; send("talk-stop"); } }
talk.addEventListener("pointerdown", talkStart);
talk.addEventListener("pointerup", talkStop);
talk.addEventListener("pointerleave", talkStop);
let keyHeld = false;
talk.addEventListener("keydown", (e) => {
  if (e.key !== " " && e.key !== "Enter") return;
  e.preventDefault(); // every repeat too, or the browser turns it into a click
  keyHeld = true;
  if (!e.repeat) talkStart();
});
talk.addEventListener("keyup", (e) => {
  if (e.key === " " || e.key === "Enter") { keyHeld = false; talkStop(); e.preventDefault(); }
});
// Screen readers in browse mode send a click with no press and release: let a click
// start talking and the next one stop.
talk.addEventListener("click", (e) => {
  if (e.detail !== 0 || keyHeld) return;
  if (talking) talkStop(); else talkStart();
});
addEventListener("keydown", (e) => {
  if (e.key === talkKey && !e.repeat) { talkStart(); e.preventDefault(); }
});
addEventListener("keyup", (e) => { if (e.key === talkKey) { talkStop(); e.preventDefault(); } });
addEventListener("blur", talkStop);

$("type-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("type-box").value.trim();
  if (!text) return;
  send(`do ${text}`);
  $("type-box").value = "";
});

// Settings
const dialog = $("settings");
$("open-settings").addEventListener("click", () => {
  dialog.showModal();
  $("open-settings").setAttribute("aria-expanded", "true");
});
dialog.addEventListener("close", () => {
  $("open-settings").setAttribute("aria-expanded", "false");
  $("open-settings").focus();
});
dialog.addEventListener("click", (e) => { if (e.target === dialog) dialog.close(); });

function setSetting(key, value) {
  send(`set ${key} ${value}`);
  applySettings({ [key]: value });
}

for (const group of document.querySelectorAll(".seg[data-setting]")) {
  const radios = [...group.querySelectorAll("[role=radio]")];
  for (const b of radios) {
    b.addEventListener("click", () => setSetting(group.dataset.setting, b.dataset.value));
  }
  // Arrow keys move between choices, as in any radio group.
  group.addEventListener("keydown", (e) => {
    const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
    if (!step) return;
    e.preventDefault();
    const i = radios.indexOf(document.activeElement);
    const next = radios[(i + step + radios.length) % radios.length];
    next.focus();
    next.click();
  });
}
// Arrowing through a closed select fires a change per step in Chromium on Windows; wait
// for the player to settle on one before switching (a voice switch takes a moment).
const settle = {};
function setSoon(key, value) {
  clearTimeout(settle[key]);
  settle[key] = setTimeout(() => setSetting(key, value), 900);
}
$("language").addEventListener("change", (e) => setSoon("language", e.target.value));
$("output").addEventListener("change", (e) => setSoon("output", e.target.value));
$("speech-rate").addEventListener("input", (e) => {
  $("rate-out").textContent = `${Number(e.target.value).toFixed(2)}×`;
});
$("speech-rate").addEventListener("change", (e) => setSoon("speech_rate", e.target.value));
$("team-up").addEventListener("change", (e) => setSetting("team_up", e.target.checked));
$("hands-free").addEventListener("change", (e) => setSetting("hands_free", e.target.checked));
$("companion").addEventListener("change", (e) => setSetting("companion", e.target.checked ? "on" : "off"));
$("paused").addEventListener("change", (e) => send(e.target.checked ? "pause" : "resume"));

function setTextSize(scale) {
  document.documentElement.style.setProperty("--caption-scale", scale);
  $("text-size").value = scale;
  localStorage.setItem("p2-text-size", scale);
}
$("text-size").addEventListener("change", (e) => setTextSize(e.target.value));
setTextSize(localStorage.getItem("p2-text-size") || "1");

let quitArmed = null;
$("quit").addEventListener("click", () => {
  const b = $("quit");
  if (!quitArmed) {
    b.textContent = "Press again to quit";
    b.classList.add("armed");
    quitArmed = setTimeout(() => {
      quitArmed = null;
      b.textContent = "Quit Player Two";
      b.classList.remove("armed");
    }, 3500);
    return;
  }
  clearTimeout(quitArmed);
  send("quit");
  dialog.close();
});

// Scripted session for previews and the README.
const DEMO = [
  [300, { kind: "ready", game: "Persona 4 Golden", found: true, talk_key: "F8", talk_key_bound: true }],
  [0, { kind: "settings", verbosity: "normal", confirm: "important", language: "English", speech_rate: 1.1, output: "voice", team_up: false, paused: false, companion: "auto", choices: { language: ["English", "Spanish", "French"] } }],
  [100, { kind: "say", source: "hello", text: "Player Two here. I can see Persona 4 Golden. Hold F8 to talk." }],
  [1600, { kind: "frame", jpeg: window.DEMO_FRAMES?.[0] }],
  [200, { kind: "status", state: "thinking" }],
  [1100, { kind: "status", state: "speaking" }],
  [0, { kind: "say", source: "narrator", text: "Teddie: “Sensei, are you leaving?” Yes, or No." }],
  [3200, { kind: "status", state: "watching" }],
  [900, { kind: "status", state: "listening" }],
  [500, { kind: "hearing", text: "no, let's go" }],
  [600, { kind: "hearing", text: "no, let's go fight" }],
  [500, { kind: "heard", text: "No, let's go fight" }],
  [200, { kind: "status", state: "acting" }],
  [900, { kind: "press", buttons: ["down", "a"] }],
  [300, { kind: "frame", jpeg: window.DEMO_FRAMES?.[1] }],
  [1100, { kind: "status", state: "speaking" }],
  [0, { kind: "say", source: "narrator", text: "Battle. Your turn. Three enemies." }],
  [1800, { kind: "status", state: "listening" }],
  [700, { kind: "heard", text: "Hit the left one with Garu" }],
  [200, { kind: "status", state: "acting" }],
  [700, { kind: "press", buttons: ["y", "down", "a"] }],
  [300, { kind: "frame", jpeg: window.DEMO_FRAMES?.[2] }],
  [900, { kind: "press", buttons: ["a"] }],
  [700, { kind: "status", state: "speaking" }],
  [0, { kind: "say", source: "agent", text: "Garu hit the left Shadow. It's down." }],
  [3000, { kind: "status", state: "watching" }],
];

async function demo() {
  for (;;) {
    $("feed").replaceChildren();
    for (const [wait, ev] of DEMO) {
      await new Promise((r) => setTimeout(r, wait));
      if (ev.kind === "frame" && !ev.jpeg) continue;
      handle(ev);
    }
    if (params.get("demo") !== "loop") return;
    await new Promise((r) => setTimeout(r, 2500));
  }
}

if (params.has("demo")) demo();
else connect();
