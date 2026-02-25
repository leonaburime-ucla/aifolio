import { Modal } from "@/core/views/components/General/Modal";
import {
  Background,
  Controls,
  ReactFlow,
  type Edge,
  type Node,
} from "@xyflow/react";

type ModelPreviewFramework = "pytorch" | "tensorflow";

type ModelPreviewMode =
  | "mlp_dense"
  | "linear_glm_baseline"
  | "tabresnet"
  | "wide_and_deep"
  | "imbalance_aware"
  | "quantile_regression"
  | "calibrated_classifier"
  | "entity_embeddings"
  | "autoencoder_head"
  | "multi_task_learning"
  | "time_aware_tabular"
  | "tree_teacher_distillation";

type ModelPreviewModalProps = {
  isOpen: boolean;
  onClose: () => void;
  framework: ModelPreviewFramework;
  mode: ModelPreviewMode;
};



type GraphModel = {
  title: string;
  summary: string;
  nodes: Node[];
  edges: Edge[];
};

function baseInputNode(): Node {
  return {
    id: "input",
    position: { x: 0, y: 120 },
    data: { label: "Input Features" },
    style: { background: "#f4f4f5", border: "1px solid #d4d4d8", width: 170 },
  };
}

function baseOutputNode(label: string): Node {
  return {
    id: "output",
    position: { x: 780, y: 120 },
    data: { label },
    style: { background: "#ecfeff", border: "1px solid #67e8f9", width: 180 },
  };
}

function buildGraph(framework: ModelPreviewFramework, mode: ModelPreviewMode): GraphModel {
  const outputLabel = mode === "quantile_regression" ? "Quantile Output (P80)" : "Prediction Output";

  if (mode === "mlp_dense") {
    const nodes: Node[] = [
      baseInputNode(),
      {
        id: "dense1",
        position: { x: 260, y: 120 },
        data: { label: "Dense Hidden Block 1" },
        style: { background: "#f5f3ff", border: "1px solid #c4b5fd", width: 180 },
      },
      {
        id: "dense2",
        position: { x: 500, y: 120 },
        data: { label: "Dense Hidden Block 2" },
        style: { background: "#f5f3ff", border: "1px solid #c4b5fd", width: 180 },
      },
      baseOutputNode(outputLabel),
    ];
    const edges: Edge[] = [
      { id: "e1", source: "input", target: "dense1" },
      { id: "e2", source: "dense1", target: "dense2" },
      { id: "e3", source: "dense2", target: "output" },
    ];
    return {
      title: "Multi-Layer Perceptron (Dense)",
      summary: "Standard dense neural network with fully connected hidden layers.",
      nodes,
      edges,
    };
  }

  if (mode === "linear_glm_baseline") {
    const nodes: Node[] = [
      baseInputNode(),
      {
        id: "linear",
        position: { x: 360, y: 120 },
        data: { label: "Linear / GLM Head" },
        style: { background: "#fefce8", border: "1px solid #fde047", width: 180 },
      },
      baseOutputNode(outputLabel),
    ];
    const edges: Edge[] = [
      { id: "e1", source: "input", target: "linear" },
      { id: "e2", source: "linear", target: "output" },
    ];
    return {
      title: "Linear / GLM Baseline",
      summary: "Single linear head model. Fast and interpretable benchmark.",
      nodes,
      edges,
    };
  }

  if (mode === "wide_and_deep") {
    const nodes: Node[] = [
      baseInputNode(),
      {
        id: "wide",
        position: { x: 280, y: 30 },
        data: { label: "Wide Branch (Linear)" },
        style: { background: "#fef2f2", border: "1px solid #fca5a5", width: 190 },
      },
      {
        id: "deep1",
        position: { x: 280, y: 190 },
        data: { label: "Deep Dense Block 1" },
        style: { background: "#f0f9ff", border: "1px solid #93c5fd", width: 190 },
      },
      {
        id: "deep2",
        position: { x: 520, y: 190 },
        data: { label: "Deep Dense Block 2" },
        style: { background: "#f0f9ff", border: "1px solid #93c5fd", width: 190 },
      },
      {
        id: "merge",
        position: { x: 520, y: 70 },
        data: { label: "Merge Wide + Deep" },
        style: { background: "#ecfdf5", border: "1px solid #86efac", width: 190 },
      },
      baseOutputNode(outputLabel),
    ];
    const edges: Edge[] = [
      { id: "e1", source: "input", target: "wide" },
      { id: "e2", source: "input", target: "deep1" },
      { id: "e3", source: "deep1", target: "deep2" },
      { id: "e4", source: "wide", target: "merge" },
      { id: "e5", source: "deep2", target: "merge" },
      { id: "e6", source: "merge", target: "output" },
    ];
    return {
      title: "Wide & Deep",
      summary: "Combines memorization from a linear path with generalization from a deep path.",
      nodes,
      edges,
    };
  }

  if (mode === "tabresnet") {
    const nodes: Node[] = [
      baseInputNode(),
      {
        id: "proj",
        position: { x: 220, y: 120 },
        data: { label: "Input Projection" },
        style: { background: "#eef2ff", border: "1px solid #a5b4fc", width: 180 },
      },
      {
        id: "res1",
        position: { x: 450, y: 70 },
        data: { label: "Residual Block 1" },
        style: { background: "#f5f3ff", border: "1px solid #c4b5fd", width: 180 },
      },
      {
        id: "res2",
        position: { x: 450, y: 170 },
        data: { label: "Residual Block 2" },
        style: { background: "#f5f3ff", border: "1px solid #c4b5fd", width: 180 },
      },
      baseOutputNode(outputLabel),
    ];
    const edges: Edge[] = [
      { id: "e1", source: "input", target: "proj" },
      { id: "e2", source: "proj", target: "res1" },
      { id: "e3", source: "res1", target: "res2" },
      { id: "e4", source: "res2", target: "output" },
      { id: "skip1", source: "proj", target: "res2", style: { strokeDasharray: "5 5" } },
    ];
    return {
      title: "TabResNet",
      summary: "Residual skip connections improve gradient flow and stability in deeper tabular networks.",
      nodes,
      edges,
    };
  }

  if (mode === "imbalance_aware") {
    const nodes: Node[] = [
      baseInputNode(),
      {
        id: "encoder",
        position: { x: 230, y: 120 },
        data: { label: "Dense Encoder" },
        style: { background: "#f0f9ff", border: "1px solid #7dd3fc", width: 180 },
      },
      {
        id: "head",
        position: { x: 480, y: 120 },
        data: { label: "Classifier Head" },
        style: { background: "#fef3c7", border: "1px solid #fcd34d", width: 180 },
      },
      {
        id: "loss",
        position: { x: 480, y: 20 },
        data: { label: "Class-Weighted Loss" },
        style: { background: "#fee2e2", border: "1px solid #fca5a5", width: 180 },
      },
      baseOutputNode("Class Probabilities"),
    ];
    const edges: Edge[] = [
      { id: "e1", source: "input", target: "encoder" },
      { id: "e2", source: "encoder", target: "head" },
      { id: "e3", source: "head", target: "output" },
      { id: "e4", source: "head", target: "loss", style: { strokeDasharray: "5 5" } },
    ];
    return {
      title: "Imbalance-Aware Classifier",
      summary: "Same network family with weighted objective to prioritize minority classes.",
      nodes,
      edges,
    };
  }

  if (mode === "quantile_regression") {
    const nodes: Node[] = [
      baseInputNode(),
      {
        id: "encoder",
        position: { x: 230, y: 120 },
        data: { label: "Dense Encoder" },
        style: { background: "#f0f9ff", border: "1px solid #7dd3fc", width: 180 },
      },
      {
        id: "qhead",
        position: { x: 480, y: 120 },
        data: { label: "Quantile Head (tau=0.8)" },
        style: { background: "#ede9fe", border: "1px solid #c4b5fd", width: 200 },
      },
      {
        id: "pinball",
        position: { x: 480, y: 20 },
        data: { label: "Pinball Loss" },
        style: { background: "#fef3c7", border: "1px solid #fcd34d", width: 180 },
      },
      baseOutputNode("P80 Forecast"),
    ];
    const edges: Edge[] = [
      { id: "e1", source: "input", target: "encoder" },
      { id: "e2", source: "encoder", target: "qhead" },
      { id: "e3", source: "qhead", target: "output" },
      { id: "e4", source: "qhead", target: "pinball", style: { strokeDasharray: "5 5" } },
    ];
    return {
      title: "Quantile Regression",
      summary: "Predicts distribution quantiles (P80 shown) instead of only a mean point estimate.",
      nodes,
      edges,
    };
  }

  if (mode === "calibrated_classifier") {
    const nodes: Node[] = [
      baseInputNode(),
      {
        id: "encoder",
        position: { x: 230, y: 120 },
        data: { label: "Dense Encoder" },
        style: { background: "#f0f9ff", border: "1px solid #7dd3fc", width: 180 },
      },
      {
        id: "head",
        position: { x: 480, y: 120 },
        data: { label: "Classifier Head" },
        style: { background: "#fef3c7", border: "1px solid #fcd34d", width: 180 },
      },
      {
        id: "smooth",
        position: { x: 480, y: 20 },
        data: { label: "Label Smoothing Objective" },
        style: { background: "#dcfce7", border: "1px solid #86efac", width: 210 },
      },
      baseOutputNode("Calibrated Probabilities"),
    ];
    const edges: Edge[] = [
      { id: "e1", source: "input", target: "encoder" },
      { id: "e2", source: "encoder", target: "head" },
      { id: "e3", source: "head", target: "output" },
      { id: "e4", source: "head", target: "smooth", style: { strokeDasharray: "5 5" } },
    ];
    return {
      title: "Calibrated Classifier",
      summary: "Adds label smoothing in training to reduce overconfident predictions and improve probability behavior.",
      nodes,
      edges,
    };
  }

  if (mode === "entity_embeddings") {
    const nodes: Node[] = [
      baseInputNode(),
      {
        id: "embed",
        position: { x: 220, y: 120 },
        data: { label: "Embedding Projection Layer" },
        style: { background: "#eef2ff", border: "1px solid #a5b4fc", width: 210 },
      },
      {
        id: "dense",
        position: { x: 500, y: 120 },
        data: { label: "Dense Predictor Stack" },
        style: { background: "#f0f9ff", border: "1px solid #7dd3fc", width: 200 },
      },
      baseOutputNode(outputLabel),
    ];
    const edges: Edge[] = [
      { id: "e1", source: "input", target: "embed" },
      { id: "e2", source: "embed", target: "dense" },
      { id: "e3", source: "dense", target: "output" },
    ];
    return {
      title: "Entity Embeddings",
      summary: "Projects sparse/tabular inputs into compact latent features before final prediction.",
      nodes,
      edges,
    };
  }

  if (mode === "autoencoder_head") {
    const nodes: Node[] = [
      baseInputNode(),
      {
        id: "encoder",
        position: { x: 220, y: 120 },
        data: { label: "Encoder" },
        style: { background: "#eef2ff", border: "1px solid #a5b4fc", width: 160 },
      },
      {
        id: "bottleneck",
        position: { x: 430, y: 120 },
        data: { label: "Latent Bottleneck" },
        style: { background: "#f5f3ff", border: "1px solid #c4b5fd", width: 170 },
      },
      {
        id: "decoder",
        position: { x: 640, y: 40 },
        data: { label: "Reconstruction Head" },
        style: { background: "#dcfce7", border: "1px solid #86efac", width: 190 },
      },
      {
        id: "pred",
        position: { x: 640, y: 180 },
        data: { label: "Prediction Head" },
        style: { background: "#fef3c7", border: "1px solid #fcd34d", width: 190 },
      },
    ];
    const edges: Edge[] = [
      { id: "e1", source: "input", target: "encoder" },
      { id: "e2", source: "encoder", target: "bottleneck" },
      { id: "e3", source: "bottleneck", target: "decoder" },
      { id: "e4", source: "bottleneck", target: "pred" },
    ];
    return {
      title: "Autoencoder + Head",
      summary: "Jointly learns a compact latent representation and a supervised prediction path.",
      nodes,
      edges,
    };
  }

  if (mode === "multi_task_learning") {
    const nodes: Node[] = [
      baseInputNode(),
      {
        id: "shared",
        position: { x: 250, y: 120 },
        data: { label: "Shared Trunk" },
        style: { background: "#f0f9ff", border: "1px solid #7dd3fc", width: 170 },
      },
      {
        id: "main",
        position: { x: 520, y: 70 },
        data: { label: "Main Task Head" },
        style: { background: "#fef3c7", border: "1px solid #fcd34d", width: 170 },
      },
      {
        id: "aux",
        position: { x: 520, y: 180 },
        data: { label: "Auxiliary Head" },
        style: { background: "#dcfce7", border: "1px solid #86efac", width: 170 },
      },
      baseOutputNode(outputLabel),
    ];
    const edges: Edge[] = [
      { id: "e1", source: "input", target: "shared" },
      { id: "e2", source: "shared", target: "main" },
      { id: "e3", source: "shared", target: "aux" },
      { id: "e4", source: "main", target: "output" },
    ];
    return {
      title: "Multi-Task Learning",
      summary: "Trains one shared representation with multiple supervised heads.",
      nodes,
      edges,
    };
  }

  if (mode === "time_aware_tabular") {
    const nodes: Node[] = [
      baseInputNode(),
      {
        id: "gate",
        position: { x: 230, y: 60 },
        data: { label: "Temporal Gate" },
        style: { background: "#fee2e2", border: "1px solid #fca5a5", width: 160 },
      },
      {
        id: "mult",
        position: { x: 230, y: 180 },
        data: { label: "Gated Features" },
        style: { background: "#ffe4e6", border: "1px solid #fda4af", width: 160 },
      },
      {
        id: "concat",
        position: { x: 480, y: 120 },
        data: { label: "Concat Raw + Gated" },
        style: { background: "#eef2ff", border: "1px solid #a5b4fc", width: 185 },
      },
      baseOutputNode(outputLabel),
    ];
    const edges: Edge[] = [
      { id: "e1", source: "input", target: "gate" },
      { id: "e2", source: "input", target: "mult" },
      { id: "e3", source: "gate", target: "mult", style: { strokeDasharray: "5 5" } },
      { id: "e4", source: "input", target: "concat" },
      { id: "e5", source: "mult", target: "concat" },
      { id: "e6", source: "concat", target: "output" },
    ];
    return {
      title: "Time-Aware Tabular",
      summary: "Applies temporal gating before deep prediction to emphasize time-derived patterns.",
      nodes,
      edges,
    };
  }

  if (mode === "tree_teacher_distillation") {
    const nodes: Node[] = [
      baseInputNode(),
      {
        id: "tree",
        position: { x: 220, y: 70 },
        data: { label: "Tree Teacher Ensemble" },
        style: { background: "#dcfce7", border: "1px solid #86efac", width: 190 },
      },
      {
        id: "student",
        position: { x: 500, y: 160 },
        data: { label: "Neural Student" },
        style: { background: "#f0f9ff", border: "1px solid #7dd3fc", width: 170 },
      },
      baseOutputNode(outputLabel),
    ];
    const edges: Edge[] = [
      { id: "e1", source: "input", target: "tree" },
      { id: "e2", source: "input", target: "student" },
      { id: "e3", source: "tree", target: "student", style: { strokeDasharray: "5 5" } },
      { id: "e4", source: "student", target: "output" },
    ];
    return {
      title: "Tree-Teacher Distillation",
      summary: "A tree teacher guides a compact neural student during training.",
      nodes,
      edges,
    };
  }

  return {
    title: framework === "pytorch" ? "PyTorch Neural Net" : "TensorFlow Neural Net",
    summary: "Dense neural network baseline.",
    nodes: [baseInputNode(), baseOutputNode(outputLabel)],
    edges: [{ id: "fallback", source: "input", target: "output" }],
  };
}

function buildLayerBullets(mode: ModelPreviewMode): { layers: string[]; terminology: { term: string; definition: string }[] } {
  switch (mode) {
    case "mlp_dense":
      return {
        layers: [
          "Input Features: Standardized tabular columns fed to the network as numerical values.",
          "Dense Hidden Blocks: Fully connected layers that learn nonlinear combinations.",
          "Output Head: Final layer that maps the network state to prediction logits or continuous values.",
        ],
        terminology: [
          { term: "Latent State", definition: "A hidden, internal summary of the data that the network creates to help it guess the final answer. Think of it like a detective's private notebook." },
          { term: "Logits", definition: "The raw, unpolished scores the model spits out before we mathematically convert them into nice, predictable percentages (like 85% probability)." },
          { term: "Nonlinear Combinations", definition: "Instead of just drawing straight, simplistic lines to find patterns, the model is allowed to draw curves, squiggles, and highly complex shapes to solve harder problems." },
          { term: "Standardized", definition: "Scaling all the input numbers so they are roughly the same size, which prevents large numbers (like a $1,000,000 house price) from completely overpowering small numbers (like 2 bedrooms) during training." }
        ],
      };
    case "linear_glm_baseline":
      return {
        layers: [
          "Input Features: Raw model input vector consisting of preprocessed numerical and categorical data.",
          "Linear / GLM Head: A single layer computing a weighted sum directly from inputs per output class or value.",
          "Output Head: Produces directly interpretable baseline predictions suitable for benchmarking.",
        ],
        terminology: [
          { term: "GLM (Generalized Linear Model)", definition: "A straightforward, classic statistical approach. It essentially takes all your inputs, multiplies them by a specific weight (importance score), and adds them all up to get an answer." },
          { term: "Baseline", definition: "A simple, bare-bones model we build first just to see how hard the problem is. If a massive, complex AI can't beat this simple baseline, we know the fancy AI isn't worth using." },
          { term: "Interpretable", definition: "Because the math is so simple (just adding up multiplied numbers), a human can look exactly at the formula and explain precisely why the model made its decision." }
        ],
      };
    case "wide_and_deep":
      return {
        layers: [
          "Wide Branch: A linear path directly connecting inputs to output that memorizes simple feature patterns.",
          "Deep Branch Blocks: A nonlinear dense pathway that learns generalizations.",
          "Merge Layer: Concatenates and combines wide and deep signals into one joint prediction.",
        ],
        terminology: [
          { term: "Memorization (Wide)", definition: "The network's ability to blatantly memorize facts it has seen many times. For example, 'if order includes a burger, suggest fries'." },
          { term: "Generalization (Deep)", definition: "The network's ability to be creative and guess correctly on things it has never seen before, like suggesting a new abstract pairing based on underlying user taste profiles." },
          { term: "Concatenates", definition: "A fancy programming word for gluing two different lists of numbers together into one big list side-by-side." }
        ],
      };
    case "tabresnet":
      return {
        layers: [
          "Input Projection: Maps raw features into a higher-dimensional hidden space.",
          "Residual Blocks: Applies learned transformations alongside skip connections.",
          "Output Head: Converts the final residual feature representation into the target prediction.",
        ],
        terminology: [
          { term: "Skip Connections", definition: "A clever trick where we provide the network a 'shortcut bypass' around complex math. If a layer isn't helpful, the model can literally just skip it. This prevents the model from breaking when we make it very deep." },
          { term: "Hidden Space / Higher-Dimensional", definition: "Imagine taking a 2D drawing and popping it out into a 3D sculpture. The model stretches your data into many dimensions so it has more 'room' to untangle messy patterns." },
          { term: "Residual", definition: "Instead of learning the entire answer from scratch at every step, the model just learns the 'residual' (the tiny remaining difference or correction) needed to fix the previous step's guess." }
        ],
      };
    case "imbalance_aware":
      return {
        layers: [
          "Dense Encoder: Extracts a nonlinear representational state from the raw tabular inputs.",
          "Classifier Head: Computes unnormalized scores which are transformed into class probabilities.",
          "Class-Weighted Loss Path: The optimization phase where the algorithm artificially upweights mistakes made on minority classes.",
        ],
        terminology: [
          { term: "Minority Class", definition: "A scenario where the thing you are looking for is extremely rare. For example, tracking credit card fraud where 99.9% of transactions are perfectly normal, and only 0.1% are fraudulent." },
          { term: "Class-Weighted Loss", definition: "We heavily penalize the AI if it misses the rare event. It's like telling the model: 'It's okay to accidentally flag a normal transaction as fraud, but if you let a real fraudster slip by, you fail instantly.'" },
          { term: "Encoder", definition: "The part of the AI that reads your raw data (like spreadsheets) and 'encodes' or translates it into a secret machine language that the rest of the AI can easily understand." }
        ],
      };
    case "quantile_regression":
      return {
        layers: [
          "Dense Encoder: Extracts continuous predictive latent features from the initial input space.",
          "Quantile Head (tau=0.8): Predicts the upper quartile boundary (P80) where 80% of actual values fall below the prediction.",
          "Pinball Loss Path: A specialized asymmetrical penalty function.",
        ],
        terminology: [
          { term: "Quantile Regression", definition: "Instead of just guessing the 'average' expected outcome, this model guesses specific boundary lines. E.g., 'I am 80% sure the delivery will arrive before 5:00 PM'." },
          { term: "Tau (τ)", definition: "A symbol representing the specific boundary line we want to draw. A Tau of 0.8 means we want the 80th percentile mark." },
          { term: "Pinball Loss", definition: "A unique grading system. If we want to predict the absolute worst-case scenario (like maximum possible traffic), this mathematically punishes the model way harder for underestimating the traffic than overestimating it." }
        ],
      };
    case "calibrated_classifier":
      return {
        layers: [
          "Dense Encoder: Builds a mathematical representation designed to linearly separate distinct classes.",
          "Classifier Head: Outputs raw confidence scores transformed to class probabilities.",
          "Label Smoothing Objective: Adds artificial uncertainty to the truth labels during training.",
        ],
        terminology: [
          { term: "Calibration", definition: "Ensuring the AI isn't falsely confident. If an AI says it is 90% sure it will rain, calibration testing proves it actually rains exactly 9 out of 10 times it makes that claim." },
          { term: "Label Smoothing", definition: "A trick to stop the AI from acting arrogant. Instead of letting the AI be 100% certain about training answers, we cap its maximum confidence at 90%. This forces it to remain slightly humble and open-minded when facing new, bizarre data." },
          { term: "Linearly Separate", definition: "Pushing the data around in a mathematical space until you can draw a literal, straight physical line between the 'Yes' answers and the 'No' answers." }
        ],
      };
    case "entity_embeddings":
      return {
        layers: [
          "Embedding Projection Layer: Compresses sparse categorical IDs into continuous dense latent factors.",
          "Dense Predictor Stack: Further refines those latent factors into task-relevant features.",
          "Output Head: Maps the final latent representation directly to the prediction.",
        ],
        terminology: [
          { term: "Categorical IDs", definition: "Data that exists as distinct groups or labels rather than numbers (like Zip Codes, User IDs, or Product Brands)." },
          { term: "Embeddings", definition: "An incredibly powerful trick that turns words or categories into coordinates on a map. For example, the AI might learn to naturally place 'Apple' and 'Banana' very close together on this map because they behave similarly." },
          { term: "Sparse", definition: "A situation where most of your data is entirely zeros or completely empty blank spaces." }
        ],
      };
    case "autoencoder_head":
      return {
        layers: [
          "Encoder: Computes and compresses the entire feature space down into a much smaller, compact latent representation.",
          "Latent Bottleneck: The shared, compressed inner state.",
          "Reconstruction Head: A decoder path that attempts to unpack the bottleneck vector to rebuild the original input features.",
          "Prediction Head: Operates in parallel, utilizing that exact same noise-filtered bottleneck state to make supervised target predictions.",
        ],
        terminology: [
          { term: "Autoencoder", definition: "An AI that is literally forced to play a game of 'telephone' with itself. It squashes data down, sends it through a tiny wire, and then tries to perfectly rebuild the original data on the other side." },
          { term: "Bottleneck", definition: "The tiny wire in the telephone game. Because the data has to squeeze through this restrictive bottleneck, the AI is physically forced to throw away useless noise and purely memorize the most critically important structural concepts." },
          { term: "Reconstruction / Decoder", Phase: "The part of the AI whose only job is to unpack the tightly compressed ZIP file of data back into its original size." }
        ],
      };
    case "multi_task_learning":
      return {
        layers: [
          "Shared Trunk: A set of initial dense layers that learns a common representation optimized across multiple related objectives.",
          "Main Task Head: The specialized output layer focusing solely on optimizing the primary supervised prediction objective.",
          "Auxiliary Head: An additional output layer adding secondary supervision signals.",
        ],
        terminology: [
          { term: "Multi-Task", definition: "Training the AI to solve two different problems at the exact same time using the exact same brain. Surprisingly, learning two related things together often makes the AI vastly smarter at both." },
          { term: "Shared Trunk", definition: "The base foundational layers of the AI. Like the trunk of a tree, it does all the heavy lifting before branching off into separate 'Heads' for specific tasks." },
          { term: "Auxiliary Supervision", definition: "A fake 'side-quest' we force the AI to solve during training. We don't actually care about the answer to the side-quest, it entirely exists just to force the AI to learn better foundational patterns." }
        ],
      };
    case "time_aware_tabular":
      return {
        layers: [
          "Temporal Gate: A learnable mechanism evaluating which specific time-derived signals should be emphasized or suppressed.",
          "Gated Features: Applies the learned temporal mathematical weighting back to the relevant inputs.",
          "Concat Raw + Gated: Combines both the base inputs and the time-aware features into a single array passed down for prediction.",
        ],
        terminology: [
          { term: "Temporal", definition: "A fancy word meaning 'related to time'. For example, noticing that ice cream sales fundamentally behave differently in August versus December." },
          { term: "Gating Mechanism", definition: "Like a bouncer at a club. The AI learns to automatically open the 'gate' to let highly relevant seasonal information pass through, but physically closes the gate to block out noisy, irrelevant time data." },
          { term: "Seasonality", definition: "Repeating patterns that naturally loop on a schedule (e.g., higher server traffic every morning at 9 AM, or higher retail sales every Friday)." }
        ],
      };
    case "tree_teacher_distillation":
      return {
        layers: [
          "Tree Teacher Ensemble: A pre-trained, robust forest model that naturally captures strong tabular decision boundaries.",
          "Neural Student: A more compact neural network learning to mimic the complex teacher.",
          "Teacher-to-Student Distillation Path: The loss function penalizes the neural student relative to how closely it matches the teacher's soft prediction probabilities.",
        ],
        terminology: [
          { term: "Knowledge Distillation", definition: "A process where we have a massive, slow, genius AI (the Teacher) teach a tiny, hyper-fast AI (the Student) how to emulate its exact behavior so we can run it affordably on a cell phone." },
          { term: "Tree Ensemble", definition: "An AI built not from 'neural networks' but from hundreds of 'decision trees' (essentially massive flowcharts of Yes/No questions) voting together on an answer. They are notoriously great at spreadsheet data." },
          { term: "Soft Probabilities", definition: "Instead of telling the student the final answer is simply 'Cat', the Teacher tells the student 'I am 82% sure it is a Cat, 15% sure it is a Dog, and 3% sure it is a Car.' The student learns infinitely faster by observing these nuanced doubts." }
        ],
      };
    default:
      return {
        layers: [
          "Input Features: Mapped and processed tabular columns suitable for numerical computation.",
          "Model Core: Architecture-specific internal layers responsible for representation learning.",
          "Output Head: Maps the abstract core features into final predictions.",
        ],
        terminology: [
          { term: "Features", definition: "The individual columns of data you feed the AI (like 'Age', 'Height', or 'Zip Code')." },
          { term: "Tabular", definition: "Data that lives beautifully in traditional rows and columns, exactly like a standard Excel spreadsheet or SQL database." },
          { term: "Prediction", definition: "The final guess the AI spits out after crunching all the numbers." }
        ],
      };
  }
}

export function ModelPreviewModal({
  isOpen,
  onClose,
  framework,
  mode,
}: ModelPreviewModalProps) {
  const rawGraph = buildGraph(framework, mode);
  const data = buildLayerBullets(mode);

  // Inject strict dark text color onto all nodes (prevents dark mode browsers from rendering white text on light nodes)
  const graph = {
    ...rawGraph,
    nodes: rawGraph.nodes.map((node) => ({
      ...node,
      style: { ...node.style, color: "#18181b" },
    })),
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      position="top"
      title={`${graph.title} (${framework === "pytorch" ? "PyTorch" : "TensorFlow"})`}
    >
      <div className="space-y-4 p-1 pb-8">
        <p className="text-sm text-zinc-600">{graph.summary}</p>

        {/* React Flow Visualization */}
        <div className="h-[200px] rounded-md border border-zinc-200 bg-zinc-50/50">
          <ReactFlow
            nodes={graph.nodes}
            edges={graph.edges}
            fitView
            fitViewOptions={{ padding: 0.2 }}
            zoomOnScroll={false}
          >
            <Controls />
            <Background gap={16} size={1} />
          </ReactFlow>
        </div>

        {/* Algorithm Explanations */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <div className="rounded-md border border-zinc-200 bg-white p-4">
            <p className="mb-2 text-sm font-semibold text-zinc-800">
              Layer Architecture Breakdown
            </p>
            <ul className="space-y-3 text-xs text-zinc-600">
              {data.layers.map((bullet) => {
                const [term, ...defParts] = bullet.split(": ");
                const definition = defParts.join(": ");
                return (
                  <li key={bullet} className="flex flex-col gap-0.5 relative pl-4">
                    <span className="absolute left-0 top-1.5 h-1.5 w-1.5 rounded-full bg-zinc-400" />
                    <span className="font-semibold text-zinc-800">{term}</span>
                    <span className="leading-relaxed">{definition}</span>
                  </li>
                );
              })}
            </ul>
          </div>

          {data.terminology.length > 0 ? (
            <div className="rounded-md border border-blue-100 bg-blue-50/50 p-4">
              <p className="mb-2 text-sm font-semibold text-blue-900">
                Key Terminology
              </p>
              <ul className="space-y-3 text-xs text-blue-800/80">
                {data.terminology.map(({ term, definition }) => (
                  <li key={term} className="flex flex-col gap-0.5 relative pl-4">
                    <span className="absolute left-0 top-1.5 h-1.5 w-1.5 rounded-full bg-blue-400" />
                    <span className="font-semibold text-blue-900">{term}</span>
                    <span className="leading-relaxed">{definition}</span>
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <div className="rounded-md border border-zinc-200 bg-zinc-50 p-4 flex items-center justify-center">
              <p className="text-xs text-zinc-400">No complex terminology explicitly defined for this baseline model.</p>
            </div>
          )}
        </div>
        {/* Explicit bottom spacer strictly enforcing 30px buffer */}
        <div className="h-[30px] shrink-0" />
      </div>
    </Modal>
  );
}


