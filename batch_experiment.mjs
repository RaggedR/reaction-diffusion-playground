#!/usr/bin/env node
/**
 * Batch experiment runner for Evolving Turing Patterns.
 * CPU-based Gray-Scott simulation — no browser/WebGL needed.
 *
 * Usage:
 *   node batch_experiment.mjs [--gens 30] [--seeds 10] [--topos none,ring,star,fc] [--out results.csv]
 */

import { writeFileSync } from 'fs';

// ── CLI args ────────────────────────────────────────────────

function parseArgs() {
  const args = process.argv.slice(2);
  const opts = { gens: 30, seeds: 10, topos: ['none', 'ring', 'star', 'fc'], out: 'results.csv' };
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--gens')  opts.gens  = parseInt(args[++i]);
    if (args[i] === '--seeds') opts.seeds = parseInt(args[++i]);
    if (args[i] === '--topos') opts.topos = args[++i].split(',');
    if (args[i] === '--out')   opts.out   = args[++i];
  }
  return opts;
}

// ── Configuration ───────────────────────────────────────────

const CONFIG = {
  simSize: 128,
  simSteps: 4000,
  numIslands: 4,
  popSize: 8,
  eliteCount: 2,
  tournamentSize: 3,
  migrationInterval: 5,
  mutationSigma: { f: 0.004, k: 0.003, dA: 0.008, dB: 0.015 },
  initRange: {
    f:  [0.02, 0.08],
    k:  [0.045, 0.065],
    dA: [0.18, 0.24],
    dB: [0.04, 0.14]
  }
};

const PARAM_RANGES = {
  f:  [0.001, 0.12],
  k:  [0.03,  0.08],
  dA: [0.1,   0.3],
  dB: [0.01,  0.25]
};

// ── PRNG ────────────────────────────────────────────────────

function mulberry32(seed) {
  return function() {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

let rng = Math.random;

// ── Utilities ───────────────────────────────────────────────

function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

function gaussianRandom() {
  let u, v, s;
  do { u = rng() * 2 - 1; v = rng() * 2 - 1; s = u*u + v*v; } while (s >= 1 || s === 0);
  return u * Math.sqrt(-2 * Math.log(s) / s);
}

function randRange(lo, hi) { return lo + rng() * (hi - lo); }

function shannonEntropy(histogram, total) {
  let e = 0;
  for (let i = 0; i < histogram.length; i++) {
    if (histogram[i] > 0) { const p = histogram[i] / total; e -= p * Math.log2(p); }
  }
  return e;
}

// ── CPU Gray-Scott Simulator ────────────────────────────────

class Simulator {
  constructor(size) {
    this.size = size;
    const n = size * size;
    this.A0 = new Float32Array(n);
    this.B0 = new Float32Array(n);
    this.A1 = new Float32Array(n);
    this.B1 = new Float32Array(n);
    this._initSeed();
  }

  _initSeed() {
    const s = this.size;
    // Fill: A=1, B=0
    this.seedA = new Float32Array(s * s).fill(1.0);
    this.seedB = new Float32Array(s * s).fill(0.0);

    // Seed circles of chemical B
    const spots = [[0.5,0.5,0.12],[0.3,0.3,0.05],[0.7,0.3,0.05],[0.3,0.7,0.05],[0.7,0.7,0.05]];
    for (const [cx, cy, r] of spots) {
      const px = Math.floor(cx * s), py = Math.floor(cy * s), pr = Math.floor(r * s);
      for (let y = py - pr; y <= py + pr; y++) {
        for (let x = px - pr; x <= px + pr; x++) {
          const wy = ((y % s) + s) % s;
          const wx = ((x % s) + s) % s;
          if (Math.hypot(x - px, y - py) < pr) {
            this.seedA[wy * s + wx] = 0.5;
            this.seedB[wy * s + wx] = 0.25;
          }
        }
      }
    }
  }

  evaluate(ind) {
    const s = this.size;
    const n = s * s;
    const { f, k, dA, dB } = ind;
    const dt = 1.0;

    // Copy seed state
    this.A0.set(this.seedA);
    this.B0.set(this.seedB);

    let srcA = this.A0, srcB = this.B0, dstA = this.A1, dstB = this.B1;

    for (let step = 0; step < CONFIG.simSteps; step++) {
      for (let y = 0; y < s; y++) {
        const ym = ((y - 1 + s) % s) * s;
        const yc = y * s;
        const yp = ((y + 1) % s) * s;
        for (let x = 0; x < s; x++) {
          const xm = (x - 1 + s) % s;
          const xp = (x + 1) % s;
          const idx = yc + x;

          const a = srcA[idx];
          const b = srcB[idx];

          // 5-point Laplacian
          const lapA = srcA[yc + xp] + srcA[yc + xm] + srcA[yp + x] + srcA[ym + x] - 4 * a;
          const lapB = srcB[yc + xp] + srcB[yc + xm] + srcB[yp + x] + srcB[ym + x] - 4 * b;

          const r = a * b * b;
          dstA[idx] = clamp(a + (dA * lapA - r + f * (1 - a)) * dt, 0, 1);
          dstB[idx] = clamp(b + (dB * lapB + r - (f + k) * b) * dt, 0, 1);
        }
      }
      // Swap buffers
      [srcA, dstA] = [dstA, srcA];
      [srcB, dstB] = [dstB, srcB];
    }

    // Compute fitness from final B channel (srcB after last swap)
    return this._fitness(srcB);
  }

  _fitness(B) {
    const n = B.length;
    const bins = 32;
    const hist = new Float32Array(bins);
    let alive = 0;

    for (let i = 0; i < n; i++) {
      const b = B[i];
      hist[Math.min(Math.floor(b * bins), bins - 1)]++;
      if (b > 0.05 && b < 0.95) alive++;
    }

    const entropy = shannonEntropy(hist, n) / Math.log2(bins);
    const aliveRatio = alive / n;
    const alivePenalty = 4 * aliveRatio * (1 - aliveRatio);
    return entropy * (0.3 + 0.7 * alivePenalty);
  }
}

// ── Individual ──────────────────────────────────────────────

class Individual {
  constructor(f, k, dA, dB) {
    this.f = f; this.k = k; this.dA = dA; this.dB = dB;
    this.fitness = 0;
  }

  static random() {
    const r = CONFIG.initRange;
    return new Individual(randRange(...r.f), randRange(...r.k), randRange(...r.dA), randRange(...r.dB));
  }

  clone() { const c = new Individual(this.f, this.k, this.dA, this.dB); c.fitness = this.fitness; return c; }

  mutate() {
    const s = CONFIG.mutationSigma;
    this.f  = clamp(this.f  + gaussianRandom() * s.f,  0.001, 0.12);
    this.k  = clamp(this.k  + gaussianRandom() * s.k,  0.03,  0.08);
    this.dA = clamp(this.dA + gaussianRandom() * s.dA, 0.1,   0.3);
    this.dB = clamp(this.dB + gaussianRandom() * s.dB, 0.01,  0.25);
  }

  static crossover(a, b) {
    return new Individual(
      rng() < 0.5 ? a.f : b.f,
      rng() < 0.5 ? a.k : b.k,
      rng() < 0.5 ? a.dA : b.dA,
      rng() < 0.5 ? a.dB : b.dB
    );
  }
}

// ── Island ──────────────────────────────────────────────────

class Island {
  constructor(id) {
    this.id = id;
    this.population = Array.from({length: CONFIG.popSize}, () => Individual.random());
  }

  get best() { return this.population.reduce((a, b) => a.fitness > b.fitness ? a : b); }
  get worst() { return this.population.reduce((a, b) => a.fitness < b.fitness ? a : b); }

  tournamentSelect() {
    let best = null;
    for (let i = 0; i < CONFIG.tournamentSize; i++) {
      const c = this.population[Math.floor(rng() * this.population.length)];
      if (!best || c.fitness > best.fitness) best = c;
    }
    return best;
  }

  evolve() {
    this.population.sort((a, b) => b.fitness - a.fitness);
    const next = [];
    for (let i = 0; i < CONFIG.eliteCount; i++) next.push(this.population[i].clone());
    while (next.length < CONFIG.popSize) {
      const child = Individual.crossover(this.tournamentSelect(), this.tournamentSelect());
      child.mutate();
      next.push(child);
    }
    this.population = next;
  }
}

// ── Migration ───────────────────────────────────────────────

const TOPOLOGIES = {
  none: () => [],
  ring: (n) => Array.from({length: n}, (_, i) => [i, (i+1) % n]),
  star: (n) => Array.from({length: n-1}, (_, i) => [0, i+1]),
  fc:   (n) => { const e = []; for (let i = 0; i < n; i++) for (let j = i+1; j < n; j++) e.push([i, j]); return e; }
};

function migrate(islands, topology) {
  const edges = TOPOLOGIES[topology](islands.length);
  const migrants = islands.map(isl => isl.best.clone());
  for (const [a, b] of edges) {
    const wA = islands[a].worst;
    const wB = islands[b].worst;
    if (migrants[b].fitness > wA.fitness) {
      const idx = islands[a].population.indexOf(wA);
      islands[a].population[idx] = migrants[b].clone();
    }
    if (migrants[a].fitness > wB.fitness) {
      const idx = islands[b].population.indexOf(wB);
      islands[b].population[idx] = migrants[a].clone();
    }
  }
}

// ── Diversity Metrics ────────────────────────────────────────

function normalizeParams(ind) {
  return [
    (ind.f  - PARAM_RANGES.f[0])  / (PARAM_RANGES.f[1]  - PARAM_RANGES.f[0]),
    (ind.k  - PARAM_RANGES.k[0])  / (PARAM_RANGES.k[1]  - PARAM_RANGES.k[0]),
    (ind.dA - PARAM_RANGES.dA[0]) / (PARAM_RANGES.dA[1] - PARAM_RANGES.dA[0]),
    (ind.dB - PARAM_RANGES.dB[0]) / (PARAM_RANGES.dB[1] - PARAM_RANGES.dB[0])
  ];
}

function euclideanDist(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] - b[i]) ** 2;
  return Math.sqrt(sum);
}

function avgPairwiseDist(individuals) {
  if (individuals.length < 2) return 0;
  const normed = individuals.map(normalizeParams);
  let total = 0, count = 0;
  for (let i = 0; i < normed.length; i++) {
    for (let j = i + 1; j < normed.length; j++) {
      total += euclideanDist(normed[i], normed[j]);
      count++;
    }
  }
  return total / count;
}

function centroid(individuals) {
  const normed = individuals.map(normalizeParams);
  const dim = normed[0].length;
  const c = new Array(dim).fill(0);
  for (const p of normed) for (let i = 0; i < dim; i++) c[i] += p[i];
  for (let i = 0; i < dim; i++) c[i] /= normed.length;
  return c;
}

function computeDiversity(isls) {
  const allInds = isls.flatMap(isl => isl.population);
  const total = avgPairwiseDist(allInds);
  const withinValues = isls.map(isl => avgPairwiseDist(isl.population));
  const within = withinValues.reduce((a, b) => a + b, 0) / withinValues.length;
  const centroids = isls.map(isl => centroid(isl.population));
  let bTotal = 0, bCount = 0;
  for (let i = 0; i < centroids.length; i++) {
    for (let j = i + 1; j < centroids.length; j++) {
      bTotal += euclideanDist(centroids[i], centroids[j]);
      bCount++;
    }
  }
  return { total, within, between: bCount > 0 ? bTotal / bCount : 0 };
}

// ── Experiment Runner ────────────────────────────────────────

function run() {
  const opts = parseArgs();
  const { gens, seeds, topos, out } = opts;
  const totalRuns = topos.length * seeds;

  console.log(`Batch experiment: ${topos.join(', ')} × ${seeds} seeds × ${gens} generations`);
  console.log(`Total runs: ${totalRuns}`);

  const simulator = new Simulator(CONFIG.simSize);

  // Time a single evaluation for estimate
  rng = mulberry32(999);
  const testInd = Individual.random();
  const t0 = performance.now();
  simulator.evaluate(testInd);
  const evalMs = performance.now() - t0;
  const evalsPerRun = CONFIG.popSize * CONFIG.numIslands * gens;
  const estMinutes = (totalRuns * evalsPerRun * evalMs / 1000 / 60).toFixed(1);
  console.log(`Single evaluation: ${evalMs.toFixed(0)}ms → estimated total: ~${estMinutes} minutes\n`);

  const rows = [];
  let runIndex = 0;

  for (const topo of topos) {
    for (let seed = 1; seed <= seeds; seed++) {
      const runStart = performance.now();
      rng = mulberry32(seed);
      const islands = Array.from({length: CONFIG.numIslands}, (_, i) => new Island(i));

      for (let gen = 0; gen < gens; gen++) {
        // Evaluate
        for (const isl of islands) {
          for (const ind of isl.population) {
            ind.fitness = simulator.evaluate(ind);
          }
        }

        // Record metrics
        const diversity = computeDiversity(islands);
        const allFit = islands.flatMap(isl => isl.population.map(ind => ind.fitness));
        rows.push(
          `${topo},${seed},${gen},${Math.max(...allFit).toFixed(6)},${(allFit.reduce((a, b) => a + b, 0) / allFit.length).toFixed(6)},${diversity.total.toFixed(6)},${diversity.between.toFixed(6)},${diversity.within.toFixed(6)}`
        );

        // Evolve
        for (const isl of islands) isl.evolve();
        if (topo !== 'none' && gen > 0 && gen % CONFIG.migrationInterval === 0) {
          migrate(islands, topo);
        }
      }

      runIndex++;
      const elapsed = ((performance.now() - runStart) / 1000).toFixed(1);
      const best = Math.max(...islands.flatMap(isl => isl.population.map(ind => ind.fitness)));
      console.log(`[${runIndex}/${totalRuns}] ${topo} seed=${seed}  ${elapsed}s  best=${best.toFixed(3)}`);
    }
  }

  // Write CSV
  const header = 'topology,seed,generation,best_fitness,mean_fitness,diversity_total,diversity_between,diversity_within';
  writeFileSync(out, header + '\n' + rows.join('\n') + '\n');
  console.log(`\nWrote ${rows.length} rows to ${out}`);

  // Print summary
  console.log('\n── Summary (final generation) ──────────────────');
  for (const topo of topos) {
    const finalRows = rows
      .filter(r => r.startsWith(`${topo},`) && r.split(',')[2] === String(gens - 1))
      .map(r => r.split(',').map(Number));
    if (finalRows.length === 0) continue;

    const divTotal = finalRows.map(r => r[5]);
    const divBetween = finalRows.map(r => r[6]);
    const bestFit = finalRows.map(r => r[3]);
    const mean = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
    const std = arr => { const m = mean(arr); return Math.sqrt(arr.reduce((a, v) => a + (v - m) ** 2, 0) / arr.length); };

    console.log(`${topo.padEnd(6)} | diversity: ${mean(divTotal).toFixed(3)} ± ${std(divTotal).toFixed(3)} | between: ${mean(divBetween).toFixed(3)} ± ${std(divBetween).toFixed(3)} | fitness: ${mean(bestFit).toFixed(3)} ± ${std(bestFit).toFixed(3)}`);
  }
}

run();
