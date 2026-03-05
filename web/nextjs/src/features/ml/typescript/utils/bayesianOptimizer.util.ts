import type {
  BayesianOptimizerRuntime,
  HyperParams,
  OptimalParamsSuggestion,
  ParamKey,
  ParamSpec,
  ParsedRun,
} from "@/features/ml/__types__/typescript/utils/bayesianOptimizer.types";
import type { TrainingRunRow } from "@/features/ml/__types__/typescript/utils/trainingRuns.types";
export type { HyperParams } from "@/features/ml/__types__/typescript/utils/bayesianOptimizer.types";
type EmptyOptions = Record<string, never>;

const EPS = 1e-9;

function parseNumeric(
  { value }: { value: unknown },
  {}: EmptyOptions = {}
): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  const normalized = trimmed.replace("x10^", "e");
  const parsed = Number(normalized);
  return Number.isFinite(parsed) ? parsed : null;
}

function metricHigherIsBetter(
  { metricName }: { metricName: string },
  {}: EmptyOptions = {}
): boolean {
  const normalized = metricName.toLowerCase();
  return (
    normalized.includes("accuracy") ||
    normalized.includes("f1") ||
    normalized.includes("auc") ||
    normalized.includes("precision") ||
    normalized.includes("recall") ||
    normalized.includes("r2")
  );
}

function clamp(
  { value, min, max }: { value: number; min: number; max: number },
  {}: EmptyOptions = {}
): number {
  return Math.max(min, Math.min(max, value));
}

function mean(
  { values }: { values: number[] },
  {}: EmptyOptions = {}
): number {
  return values.reduce((acc, value) => acc + value, 0) / Math.max(1, values.length);
}

function std(
  { values }: { values: number[] },
  {}: EmptyOptions = {}
): number {
  const m = mean({ values });
  const variance =
    values.reduce((acc, value) => acc + (value - m) ** 2, 0) / Math.max(1, values.length);
  return Math.sqrt(variance);
}

function gaussianDensity(
  { x, mu, sigma }: { x: number; mu: number; sigma: number },
  {}: EmptyOptions = {}
): number {
  const s = Math.max(sigma, 1e-3);
  const z = (x - mu) / s;
  return Math.exp(-0.5 * z * z) / (s * Math.sqrt(2 * Math.PI));
}

function sampleNormal(
  { mu, sigma }: { mu: number; sigma: number },
  { random = Math.random }: Partial<Pick<BayesianOptimizerRuntime, "random">> = {}
): number {
  const u1 = random() || 1e-12;
  const u2 = random();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mu + sigma * z;
}

function parseRun(
  { row }: { row: TrainingRunRow },
  {}: EmptyOptions = {}
): ParsedRun | null {
  const metricName = String(row.metric_name ?? "").trim().toLowerCase();
  const metricScore = parseNumeric({ value: row.metric_score });
  const epochs = parseNumeric({ value: row.epochs });
  const learningRate = parseNumeric({ value: row.learning_rate });
  const testSize = parseNumeric({ value: row.test_size });
  const batchSize = parseNumeric({ value: row.batch_size });
  const hiddenDim = parseNumeric({ value: row.hidden_dim });
  const numHiddenLayers = parseNumeric({ value: row.num_hidden_layers });
  const dropout = parseNumeric({ value: row.dropout });

  if (
    !metricName ||
    metricName === "n/a" ||
    metricScore === null ||
    epochs === null ||
    learningRate === null ||
    testSize === null ||
    batchSize === null ||
    hiddenDim === null ||
    numHiddenLayers === null ||
    dropout === null
  ) {
    return null;
  }

  return {
    metric_name: metricName,
    metric_score: metricScore,
    epochs,
    learning_rate: learningRate,
    test_size: testSize,
    batch_size: batchSize,
    hidden_dim: hiddenDim,
    num_hidden_layers: numHiddenLayers,
    dropout,
  };
}

function getParamSpecs(
  { runs }: { runs: ParsedRun[] },
  {}: EmptyOptions = {}
): ParamSpec[] {
  const keys: Array<{ key: ParamKey; type: "int" | "float"; floor: number; ceil: number }> = [
    { key: "epochs", type: "int", floor: 1, ceil: 500 },
    { key: "learning_rate", type: "float", floor: 1e-5, ceil: 1 },
    { key: "test_size", type: "float", floor: 0.001, ceil: 0.999 },
    { key: "batch_size", type: "int", floor: 1, ceil: 200 },
    { key: "hidden_dim", type: "int", floor: 8, ceil: 500 },
    { key: "num_hidden_layers", type: "int", floor: 1, ceil: 15 },
    { key: "dropout", type: "float", floor: 0, ceil: 0.9 },
  ];

  return keys.map((entry) => {
    const values = runs.map((run) => run[entry.key]);
    const observedMin = Math.min(...values);
    const observedMax = Math.max(...values);
    const span = Math.max(observedMax - observedMin, EPS);
    const paddedMin = observedMin - span * 0.2;
    const paddedMax = observedMax + span * 0.2;
    const min = clamp({
      value: paddedMin,
      min: entry.floor,
      max: entry.ceil,
    });
    const max = clamp({
      value: paddedMax,
      min: entry.floor,
      max: entry.ceil,
    });
    return {
      key: entry.key,
      type: entry.type,
      min: Math.min(min, max),
      max: Math.max(min, max),
    };
  });
}

/**
 * Suggests the next hyperparameter combination from completed run history.
 *
 * @param params - Required parameter object.
 * @param params.rows - Completed and failed training runs.
 * @param _options - Optional reserved options object.
 * @returns An optimizer suggestion with predicted metric, or `null` when insufficient data exists.
 */
export function findOptimalParamsFromRuns(
  { rows }: { rows: TrainingRunRow[] },
  { random = Math.random }: Partial<BayesianOptimizerRuntime> = {}
): OptimalParamsSuggestion | null {
  const parsed = rows
    .filter((row) => String(row.result ?? "") === "completed")
    .map((row) => parseRun({ row }))
    .filter((run): run is ParsedRun => run !== null);

  if (parsed.length < 5) return null;

  const higherIsBetter = metricHigherIsBetter({ metricName: parsed[0].metric_name });
  const scored = parsed
    .map((run) => ({
      ...run,
      objective: higherIsBetter ? run.metric_score : -run.metric_score,
    }))
    .sort((a, b) => b.objective - a.objective);

  const goodCount = Math.max(2, Math.floor(scored.length * 0.35));
  const goodRuns = scored.slice(0, goodCount);
  const badRuns = scored.slice(goodCount);
  const specs = getParamSpecs({ runs: scored });

  const bestObserved = scored[0];
  let bestCandidate: HyperParams = {
    epochs: bestObserved.epochs,
    learning_rate: bestObserved.learning_rate,
    test_size: bestObserved.test_size,
    batch_size: bestObserved.batch_size,
    hidden_dim: bestObserved.hidden_dim,
    num_hidden_layers: bestObserved.num_hidden_layers,
    dropout: bestObserved.dropout,
  };
  let bestAcquisition = -Infinity;
  let bestSpecs = specs;

  for (let i = 0; i < 500; i += 1) {
    const candidate = {} as HyperParams;
    for (const spec of specs) {
      const goodValues = goodRuns.map((run) => run[spec.key]);
      const badValues = badRuns.map((run) => run[spec.key]);
      const goodMu = mean({ values: goodValues });
      const goodSigma = Math.max(std({ values: goodValues }), (spec.max - spec.min) * 0.08, EPS);
      const badMu = mean({ values: badValues });
      const badSigma = Math.max(std({ values: badValues }), (spec.max - spec.min) * 0.08, EPS);
      const sampled = clamp({
        value: sampleNormal({ mu: goodMu, sigma: goodSigma }, { random }),
        min: spec.min,
        max: spec.max,
      });

      const rounded = spec.type === "int" ? Math.round(sampled) : Number(sampled.toFixed(6));
      candidate[spec.key] = rounded as never;

      const lGood = gaussianDensity({ x: sampled, mu: goodMu, sigma: goodSigma });
      const lBad = gaussianDensity({ x: sampled, mu: badMu, sigma: badSigma });
      (candidate as Record<string, number>)[`${spec.key}__llr`] = Math.log(lGood + EPS) - Math.log(lBad + EPS);
    }

    const llr = specs.reduce(
      (acc, spec) => acc + (candidate as Record<string, number>)[`${spec.key}__llr`],
      0
    );

    const nearestDistance = scored.reduce((acc, run) => {
      const dist = Math.sqrt(
        specs.reduce((sum, spec) => {
          const span = Math.max(spec.max - spec.min, EPS);
          const delta = (candidate[spec.key] - run[spec.key]) / span;
          return sum + delta * delta;
        }, 0)
      );
      return Math.min(acc, dist);
    }, Number.POSITIVE_INFINITY);

    const acquisition = llr + nearestDistance * 0.25;
    if (acquisition > bestAcquisition) {
      bestAcquisition = acquisition;
      bestSpecs = specs;
      bestCandidate = {
        epochs: Math.max(1, Math.min(500, Math.round(candidate.epochs))),
        learning_rate: clamp({ value: candidate.learning_rate, min: 1e-5, max: 1 }),
        test_size: clamp({ value: candidate.test_size, min: 0.001, max: 0.999 }),
        batch_size: Math.max(1, Math.min(200, Math.round(candidate.batch_size))),
        hidden_dim: Math.max(8, Math.min(500, Math.round(candidate.hidden_dim))),
        num_hidden_layers: Math.max(1, Math.min(15, Math.round(candidate.num_hidden_layers))),
        dropout: clamp({ value: candidate.dropout, min: 0, max: 0.9 }),
      };
    }
  }

  const nearest = [...scored]
    .sort((a, b) => {
      const distA = Math.sqrt(
        bestSpecs.reduce((sum, spec) => {
          const span = Math.max(spec.max - spec.min, EPS);
          const delta = (bestCandidate[spec.key] - a[spec.key]) / span;
          return sum + delta * delta;
        }, 0)
      );
      const distB = Math.sqrt(
        bestSpecs.reduce((sum, spec) => {
          const span = Math.max(spec.max - spec.min, EPS);
          const delta = (bestCandidate[spec.key] - b[spec.key]) / span;
          return sum + delta * delta;
        }, 0)
      );
      return distA - distB;
    })
    .slice(0, Math.min(5, scored.length));

  const weighted = nearest.map((run) => {
    const distance = Math.sqrt(
      bestSpecs.reduce((sum, spec) => {
        const span = Math.max(spec.max - spec.min, EPS);
        const delta = (bestCandidate[spec.key] - run[spec.key]) / span;
        return sum + delta * delta;
      }, 0)
    );
    const weight = 1 / (distance + 0.05);
    return { weight, metric: run.metric_score };
  });
  const weightTotal = weighted.reduce((acc, item) => acc + item.weight, 0);
  const predictedMetricValue =
    weighted.reduce((acc, item) => acc + item.metric * item.weight, 0) / Math.max(weightTotal, EPS);

  return {
    suggestion: bestCandidate,
    basedOnRuns: parsed.length,
    predictedMetricName: parsed[0].metric_name,
    predictedMetricValue,
  };
}
