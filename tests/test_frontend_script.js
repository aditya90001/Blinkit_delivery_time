const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const html = fs.readFileSync("templates/index.html", "utf8");
const match = html.match(/<script>([\s\S]*?)<\/script>/);

assert.ok(match, "prediction frontend must contain its client-side API integration");
new vm.Script(match[1], { filename: "templates/index.html" });
assert.match(html, /fetch\("\/predict"/);
assert.match(html, /fetch\("\/health"\)/);
console.log("Prediction frontend JavaScript syntax and API wiring are valid.");
