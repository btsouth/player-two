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

let socket = null;
let pressTimer = null;
let lastPress = 0;

function setState(state) {
  app.dataset.state = state;
  $("status-text").textContent = STATUS[state] || state;
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
  while (feed.children.length > 60) feed.firstElementChild.remove();
  const wrap = feed.parentElement; // the scroller
  wrap.scrollTo({ top: wrap.scrollHeight, behavior: "smooth" });
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

function handle(ev) {
  switch (ev.kind) {
    case "ready":
      if (ev.game) $("game").textContent = ev.game;
      if (ev.talk_key) $("talk-key").textContent = ev.talk_key;
      break;
    case "status":
      setState(ev.state);
      break;
    case "frame": {
      const img = $("frame");
      img.src = `data:image/jpeg;base64,${ev.jpeg}`;
      img.hidden = false;
      $("screen-empty").hidden = true;
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
    case "settings":
      document.querySelectorAll(".modes button").forEach((b) =>
        b.setAttribute("aria-checked", String(b.dataset.mode === ev.verbosity)));
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
  socket.onmessage = (m) => handle(JSON.parse(m.data));
  socket.onclose = () => {
    setState("paused");
    $("status-text").textContent = "Reconnecting";
    setTimeout(connect, 1500);
  };
}

// Talk: hold the button, or the talk key while this window has focus.
const talk = $("talk");
talk.addEventListener("pointerdown", () => send("talk-start"));
talk.addEventListener("pointerup", () => send("talk-stop"));
talk.addEventListener("pointerleave", (e) => { if (e.buttons) send("talk-stop"); });
addEventListener("keydown", (e) => {
  if (e.key === "F8" && !e.repeat) { send("talk-start"); e.preventDefault(); }
});
addEventListener("keyup", (e) => { if (e.key === "F8") { send("talk-stop"); e.preventDefault(); } });
document.querySelectorAll(".modes button").forEach((b) => b.addEventListener("click", () => {
  send(b.dataset.mode);
  handle({ kind: "settings", verbosity: b.dataset.mode });
}));

// Scripted session for previews and the README.
const DEMO = [
  [400, { kind: "ready", game: "Persona 4 Golden" }],
  [300, { kind: "frame", jpeg: window.DEMO_FRAMES?.[0] }],
  [200, { kind: "status", state: "thinking" }],
  [1300, { kind: "status", state: "speaking" }],
  [0, { kind: "say", source: "narrator", text: "Mr. Morooka: “I'm Kinshiro Morooka, your homeroom teacher from today forward!”" }],
  [3600, { kind: "status", state: "watching" }],
  [900, { kind: "status", state: "listening" }],
  [500, { kind: "hearing", text: "skip to the" }],
  [600, { kind: "hearing", text: "skip to the battle and use Agi" }],
  [500, { kind: "heard", text: "Skip to the battle and use Agi on the hurt one" }],
  [200, { kind: "status", state: "acting" }],
  [900, { kind: "press", buttons: ["a", "a", "a"] }],
  [300, { kind: "frame", jpeg: window.DEMO_FRAMES?.[1] }],
  [900, { kind: "press", buttons: ["down", "a"] }],
  [1100, { kind: "status", state: "speaking" }],
  [0, { kind: "ask", text: "Agi on Shadow B, the hurt one on the right. Go?" }],
  [2400, { kind: "status", state: "listening" }],
  [700, { kind: "heard", text: "Yes" }],
  [200, { kind: "status", state: "acting" }],
  [700, { kind: "press", buttons: ["a", "down", "a"] }],
  [300, { kind: "frame", jpeg: window.DEMO_FRAMES?.[2] }],
  [900, { kind: "status", state: "speaking" }],
  [0, { kind: "say", source: "agent", text: "Agi hit Shadow B for 45. It's down to 35." }],
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
