#!/usr/bin/env node
/**
 * Independent SemRepair execution proof bundle v0.2 verifier.
 *
 * This implementation intentionally does not import or execute the Python
 * compiler/search implementation. It uses only Node.js built-ins.
 */

import fs from "node:fs";
import crypto from "node:crypto";

const BUNDLE_SCHEMA = "semrepair-execution-proof-bundle/v0.2";
const SPEC_VERSION = "0.2";
const CANONICALIZATION_PROFILE = "semrepair-wire-c14n/v0.1";
const SEMANTIC_FIELDS = [
  "role", "entity", "frame", "representation", "mode", "convention",
  "unit", "clock", "scope", "freshness", "provenance", "ordering",
  "embodiment",
];
const NON_FORGEABLE = new Set(["provenance", "freshness"]);
const HEX64 = /^[0-9a-f]{64}$/;
const DECIMAL = /^-?(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?$/;

function assertIntegerOnlyJsonNumbers(raw) {
  let inString = false;
  let escaped = false;
  for (let i = 0; i < raw.length; i += 1) {
    const ch = raw[i];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (ch === "\\") {
        escaped = true;
      } else if (ch === '"') {
        inString = false;
      }
      continue;
    }
    if (ch === '"') {
      inString = true;
      continue;
    }
    if (ch === "-" || (ch >= "0" && ch <= "9")) {
      const m = raw.slice(i).match(/^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?/);
      if (!m) continue;
      const token = m[0];
      if (token.includes(".") || /[eE]/.test(token)) {
        throw new Error(`binary/fractional JSON number is forbidden on v0.2 wire surface: ${token}`);
      }
      i += token.length - 1;
    }
  }
}

function normalizedObjectEntries(value) {
  const seen = new Set();
  const entries = [];
  for (const [rawKey, item] of Object.entries(value)) {
    const key = rawKey.normalize("NFC");
    if (seen.has(key)) {
      throw new Error("NFC normalization produced duplicate object keys");
    }
    seen.add(key);
    entries.push([key, item]);
  }
  entries.sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
  return entries;
}

function canonicalJson(value) {
  if (value === null) return "null";
  if (value === true) return "true";
  if (value === false) return "false";
  if (typeof value === "number") {
    if (!Number.isSafeInteger(value)) {
      throw new Error("v0.2 canonical JSON permits only safe integer JSON numbers");
    }
    return String(value);
  }
  if (typeof value === "string") {
    return JSON.stringify(value.normalize("NFC"));
  }
  if (Array.isArray(value)) {
    return "[" + value.map(canonicalJson).join(",") + "]";
  }
  if (typeof value === "object") {
    return "{" + normalizedObjectEntries(value)
      .map(([key, item]) => JSON.stringify(key) + ":" + canonicalJson(item))
      .join(",") + "}";
  }
  throw new Error(`unsupported wire value type: ${typeof value}`);
}

function digest(value) {
  return crypto.createHash("sha256").update(canonicalJson(value), "utf8").digest("hex");
}

function parseDecimal(text, { nonnegative = false } = {}) {
  if (typeof text !== "string" || !DECIMAL.test(text)) {
    throw new Error(`non-canonical decimal: ${String(text)}`);
  }
  const negative = text.startsWith("-");
  const body = negative ? text.slice(1) : text;
  const [whole, frac = ""] = body.split(".");
  let coeff = BigInt(whole + frac);
  if (negative) coeff = -coeff;
  let scale = frac.length;
  while (scale > 0 && coeff % 10n === 0n) {
    coeff /= 10n;
    scale -= 1;
  }
  if (coeff === 0n && text !== "0") {
    throw new Error("zero must be encoded as 0");
  }
  if (nonnegative && coeff < 0n) {
    throw new Error("decimal must be non-negative");
  }
  return { coeff, scale };
}

function pow10(n) {
  return 10n ** BigInt(n);
}

function addDecimal(a, b) {
  const scale = Math.max(a.scale, b.scale);
  return normalizeDecimal({
    coeff: a.coeff * pow10(scale - a.scale) + b.coeff * pow10(scale - b.scale),
    scale,
  });
}

function normalizeDecimal(value) {
  let { coeff, scale } = value;
  while (scale > 0 && coeff % 10n === 0n) {
    coeff /= 10n;
    scale -= 1;
  }
  if (coeff === 0n) scale = 0;
  return { coeff, scale };
}

function compareDecimal(a, b) {
  const scale = Math.max(a.scale, b.scale);
  const left = a.coeff * pow10(scale - a.scale);
  const right = b.coeff * pow10(scale - b.scale);
  return left < right ? -1 : left > right ? 1 : 0;
}

function decimalString(value) {
  const { coeff, scale } = normalizeDecimal(value);
  if (scale === 0) return coeff.toString();
  const negative = coeff < 0n;
  let digits = (negative ? -coeff : coeff).toString().padStart(scale + 1, "0");
  const cut = digits.length - scale;
  const text = digits.slice(0, cut) + "." + digits.slice(cut);
  return negative ? "-" + text : text;
}

function assignable(actual, target) {
  for (const field of SEMANTIC_FIELDS) {
    const want = target[field];
    if (want !== null && want !== undefined && actual[field] !== want) return false;
  }
  return true;
}

function applyAdapter(current, adapter, evidenceKeys) {
  const requires = adapter.requires;
  const produces = adapter.produces;
  const requiredEvidence = adapter.required_evidence;
  if (!requires || typeof requires !== "object" || Array.isArray(requires)) return null;
  if (!produces || typeof produces !== "object" || Array.isArray(produces)) return null;
  if (!Array.isArray(requiredEvidence) || !requiredEvidence.every((x) => typeof x === "string")) {
    return null;
  }
  for (const key of requiredEvidence) if (!evidenceKeys.has(key)) return null;
  for (const [key, value] of Object.entries(requires)) {
    if (!SEMANTIC_FIELDS.includes(key) || current[key] !== value) return null;
  }
  for (const key of Object.keys(produces)) {
    if (!SEMANTIC_FIELDS.includes(key)) return null;
    if (NON_FORGEABLE.has(key) && requires[key] !== produces[key]) return null;
  }
  return { ...current, ...produces };
}

function semanticKey(value) {
  return SEMANTIC_FIELDS.map((field) => [field, value[field] ?? null]);
}

function enumerateSolutions(source, target, adapters, evidenceKeys, maxSteps, maxCost) {
  if (assignable(source, target)) return [{ cost: parseDecimal("0"), path: [] }];
  const solutions = [];
  const startKey = canonicalJson(semanticKey(source));
  const stack = [{
    current: { ...source },
    path: [],
    cost: parseDecimal("0"),
    steps: 0,
    visited: new Set([startKey]),
  }];

  while (stack.length > 0) {
    const state = stack.pop();
    if (maxCost && compareDecimal(state.cost, maxCost) > 0) continue;
    if (assignable(state.current, target)) {
      solutions.push({ cost: state.cost, path: state.path });
      continue;
    }
    if (state.steps >= maxSteps) continue;

    for (const adapter of adapters) {
      const next = applyAdapter(state.current, adapter, evidenceKeys);
      if (!next) continue;
      const key = canonicalJson(semanticKey(next));
      if (state.visited.has(key)) continue;
      let cost;
      try {
        cost = addDecimal(state.cost, parseDecimal(adapter.cost_decimal, { nonnegative: true }));
      } catch {
        continue;
      }
      if (maxCost && compareDecimal(cost, maxCost) > 0) continue;
      const visited = new Set(state.visited);
      visited.add(key);
      stack.push({
        current: next,
        path: [...state.path, String(adapter.name)],
        cost,
        steps: state.steps + 1,
        visited,
      });
    }
  }
  solutions.sort((a, b) => {
    const c = compareDecimal(a.cost, b.cost);
    if (c !== 0) return c;
    return JSON.stringify(a.path).localeCompare(JSON.stringify(b.path));
  });
  return solutions;
}

function isHex64(value) {
  return typeof value === "string" && HEX64.test(value);
}

function verify(record) {
  const checks = {};
  const errors = [];

  checks.bundle_schema = record.schema === BUNDLE_SCHEMA;
  checks.spec_version = record.spec_version === SPEC_VERSION;
  checks.canonicalization_profile = record.canonicalization_profile === CANONICALIZATION_PROFILE;

  const outer = structuredClone(record);
  const claimedBundleDigest = outer.bundle_digest;
  delete outer.bundle_digest;
  checks.bundle_digest = isHex64(claimedBundleDigest) && digest(outer) === claimedBundleDigest;

  const compilation = record.compilation_certificate;
  const effect = record.effect_certificate;
  const intent = record.intent;
  const adapters = record.adapter_registry;
  const evidence = record.evidence_identity;
  const profile = record.verification_profile;
  const bindings = record.bindings;

  const structural =
    compilation && typeof compilation === "object" && !Array.isArray(compilation) &&
    effect && typeof effect === "object" && !Array.isArray(effect) &&
    intent && typeof intent === "object" && !Array.isArray(intent) &&
    Array.isArray(adapters) && adapters.every((x) => x && typeof x === "object" && !Array.isArray(x)) &&
    evidence && typeof evidence === "object" && !Array.isArray(evidence) &&
    profile && typeof profile === "object" && !Array.isArray(profile) &&
    bindings && typeof bindings === "object" && !Array.isArray(bindings);
  checks.structure = Boolean(structural);

  if (!structural) {
    return { schema: BUNDLE_SCHEMA, valid: false, checks, errors: ["bundle structure is malformed"] };
  }

  checks.registry_wire_digest = bindings.adapter_registry_digest === digest(adapters);
  checks.evidence_wire_digest = bindings.evidence_identity_digest === digest(evidence);
  checks.compilation_wire_digest = bindings.compilation_digest === digest(compilation);
  checks.effect_wire_digest = bindings.effect_digest === digest(effect);

  const names = adapters.map((x) => x.name);
  checks.unique_adapter_names =
    names.every((x) => typeof x === "string" && x.length > 0) &&
    new Set(names).size === names.length;

  let adapterSemanticsValid = true;
  adapters.forEach((adapter, index) => {
    const requires = adapter.requires;
    const produces = adapter.produces;
    if (!requires || typeof requires !== "object" || !produces || typeof produces !== "object") {
      adapterSemanticsValid = false;
      errors.push(`adapter[${index}] has malformed semantic maps`);
      return;
    }
    const unknown = [...new Set([...Object.keys(requires), ...Object.keys(produces)])]
      .filter((field) => !SEMANTIC_FIELDS.includes(field));
    if (unknown.length > 0) {
      adapterSemanticsValid = false;
      errors.push(`adapter[${index}] uses unknown semantic fields: ${unknown.join(",")}`);
    }
    try {
      parseDecimal(adapter.cost_decimal, { nonnegative: true });
    } catch (error) {
      adapterSemanticsValid = false;
      errors.push(`adapter[${index}] cost: ${error.message}`);
    }
    for (const field of Object.keys(produces)) {
      if (NON_FORGEABLE.has(field) && requires[field] !== produces[field]) {
        adapterSemanticsValid = false;
        errors.push(`adapter[${index}] forges non-forgeable field ${field}`);
      }
    }
  });
  checks.adapter_semantics_valid = adapterSemanticsValid;

  checks.installable_compilation = compilation.repair_candidate === null;

  try {
    if (compilation.producer_selected_cost_decimal !== null) {
      parseDecimal(compilation.producer_selected_cost_decimal, { nonnegative: true });
    }
    checks.producer_selected_cost_decimal = true;
  } catch {
    checks.producer_selected_cost_decimal = false;
    errors.push("producer_selected_cost_decimal is not canonical");
  }

  const producerCompDigest = compilation.producer_decision_digest;
  const producerEffectDigest = effect.producer_decision_digest;
  checks.producer_digest_identity_shape = isHex64(producerCompDigest) && isHex64(producerEffectDigest);

  const contractId = record.semantic_contract_id;
  checks.semantic_contract_binding =
    typeof contractId === "string" && contractId.length > 0 &&
    effect.semantic_contract_id === contractId;

  checks.effect_matches_intent =
    effect.effect_id === intent.effect_id &&
    effect.action_name === intent.action_name &&
    effect.effect_class === intent.effect_class &&
    effect.dependency_version === intent.dependency_version;

  checks.compilation_effect_dependency =
    isHex64(producerCompDigest) &&
    effect.dependency_version === producerCompDigest &&
    intent.dependency_version === producerCompDigest;

  let maxSteps = profile.max_steps;
  checks.verification_profile = Number.isSafeInteger(maxSteps) && maxSteps >= 0;
  if (!checks.verification_profile) {
    errors.push("verification_profile.max_steps must be a non-negative safe integer");
    maxSteps = 0;
  }

  const selectedPath = compilation.selected_adapter_path;
  const refinedSource = compilation.refined_source;
  const target = compilation.target;
  const pathShape =
    Array.isArray(selectedPath) && selectedPath.every((x) => typeof x === "string") &&
    refinedSource && typeof refinedSource === "object" && !Array.isArray(refinedSource) &&
    target && typeof target === "object" && !Array.isArray(target);
  checks.selected_path_shape = Boolean(pathShape);

  const byName = new Map(adapters.map((adapter) => [String(adapter.name), adapter]));
  const evidenceKeys = new Set(Object.keys(evidence));
  let current = pathShape ? { ...refinedSource } : {};
  let selectedCost = parseDecimal("0");
  let replayValid = Boolean(pathShape);

  if (pathShape) {
    for (const name of selectedPath) {
      const adapter = byName.get(name);
      if (!adapter) {
        replayValid = false;
        errors.push(`selected adapter ${name} is absent from registry`);
        break;
      }
      const next = applyAdapter(current, adapter, evidenceKeys);
      if (!next) {
        replayValid = false;
        errors.push(`selected adapter ${name} is not admissible`);
        break;
      }
      try {
        selectedCost = addDecimal(selectedCost, parseDecimal(adapter.cost_decimal, { nonnegative: true }));
      } catch {
        replayValid = false;
        errors.push(`selected adapter ${name} has invalid decimal cost`);
        break;
      }
      current = next;
    }
  }

  checks.selected_path_replay = replayValid && pathShape && assignable(current, target);

  let minimumUnique = false;
  if (checks.selected_path_replay) {
    const solutions = enumerateSolutions(
      refinedSource, target, adapters, evidenceKeys, maxSteps, selectedCost
    );
    if (solutions.length > 0) {
      const minimum = solutions[0].cost;
      const minPaths = solutions
        .filter((item) => compareDecimal(item.cost, minimum) === 0)
        .map((item) => item.path);
      minimumUnique =
        compareDecimal(selectedCost, minimum) === 0 &&
        minPaths.length === 1 &&
        JSON.stringify(minPaths[0]) === JSON.stringify(selectedPath);
    }
  }
  checks.minimum_cost_unique = minimumUnique;
  if (!minimumUnique) errors.push("selected adapter path is not the unique exact-decimal minimum");

  for (const [name, passed] of Object.entries(checks)) {
    if (!passed) errors.push(`check failed: ${name}`);
  }

  return {
    schema: BUNDLE_SCHEMA,
    valid: Object.values(checks).every(Boolean),
    checks,
    errors: [...new Set(errors)],
    selected_cost_decimal: decimalString(selectedCost),
    verifier: "node-reference-v0.1",
  };
}

function main() {
  if (process.argv.length !== 3) {
    console.error("usage: verify_bundle_v2.mjs <proof-bundle-v2.json>");
    process.exit(2);
  }
  const raw = fs.readFileSync(process.argv[2], "utf8");
  try {
    assertIntegerOnlyJsonNumbers(raw);
    const record = JSON.parse(raw);
    const report = verify(record);
    process.stdout.write(JSON.stringify(report) + "\n");
    process.exit(report.valid ? 0 : 1);
  } catch (error) {
    process.stdout.write(JSON.stringify({
      schema: BUNDLE_SCHEMA,
      valid: false,
      checks: {},
      errors: [String(error.message ?? error)],
      verifier: "node-reference-v0.1",
    }) + "\n");
    process.exit(1);
  }
}

main();
