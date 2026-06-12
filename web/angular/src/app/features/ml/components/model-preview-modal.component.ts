import { CommonModule } from '@angular/common';
import { Component, Input, computed, signal } from '@angular/core';
import { ModalComponent } from '../../../shared/components/modal.component';
import type { Framework } from '../model/ml-training.types';

type FlowNode = {
  label: string;
  bg: string;
  border: string;
};

type FlowType = 'sequential' | 'wide_deep' | 'autoencoder' | 'multi_task' | 'time_aware' | 'distillation';

type ModelMeta = {
  title: string;
  summary: string;
  flowType: FlowType;
  nodes: FlowNode[];
  layers: string[];
  terminology: Array<{ term: string; definition: string }>;
};

function modelMeta(mode: string, framework: Framework): ModelMeta {
  const outputLabel = mode === 'quantile_regression' ? 'Quantile Output (P80)' : 'Prediction Output';

  switch (mode) {
    case 'mlp_dense':
      return {
        title: 'Multi-Layer Perceptron (Dense)',
        summary: 'Standard dense neural network with fully connected hidden layers.',
        flowType: 'sequential',
        nodes: [
          { label: 'Input Features', bg: '#f4f4f5', border: '#d4d4d8' },
          { label: 'Dense Hidden Block 1', bg: '#f5f3ff', border: '#c4b5fd' },
          { label: 'Dense Hidden Block 2', bg: '#f5f3ff', border: '#c4b5fd' },
          { label: outputLabel, bg: '#ecfeff', border: '#67e8f9' },
        ],
        layers: [
          'Input Features: The raw data you provide (like numbers and categories), cleaned up so the AI can easily read it.',
          "Dense Hidden Blocks: The main 'brain' layers where the AI mixes and matches the data to find hidden patterns.",
          "Output Head: The final step that takes the AI's thoughts and spits out the actual guess (like predicting a house price).",
        ],
        terminology: [
          { term: 'Latent State', definition: "A hidden, internal summary of the data that the network creates to help it guess the final answer. Think of it like a detective's private notebook." },
          { term: 'Logits', definition: 'The raw, unpolished scores the model spits out before we mathematically convert them into predictable percentages.' },
          { term: 'Nonlinear Combinations', definition: 'Instead of just drawing straight lines to find patterns, the model can draw curves and complex shapes to solve harder problems.' },
          { term: 'Standardized', definition: 'Scaling all input numbers so they are roughly the same size, which prevents large values from overpowering small ones during training.' },
        ],
      };

    case 'linear_glm_baseline':
      return {
        title: 'Linear / GLM Baseline',
        summary: 'Single linear head model. Fast and interpretable benchmark.',
        flowType: 'sequential',
        nodes: [
          { label: 'Input Features', bg: '#f4f4f5', border: '#d4d4d8' },
          { label: 'Linear / GLM Head', bg: '#fefce8', border: '#fde047' },
          { label: outputLabel, bg: '#ecfeff', border: '#67e8f9' },
        ],
        layers: [
          'Input Features: The raw starting data (like numbers and categories) ready for the model to look at.',
          "Linear / GLM Head: A simple math step that just multiplies each input by a specific 'importance score' and adds them all up.",
          'Output Head: The final result, giving us a highly understandable baseline to compare our fancy AI models against.',
        ],
        terminology: [
          { term: 'GLM (Generalized Linear Model)', definition: 'A straightforward, classic statistical approach. It essentially takes all your inputs, multiplies them by a specific weight, and adds them all up to get an answer.' },
          { term: 'Baseline', definition: "A simple, bare-bones model we build first just to see how hard the problem is. If a massive, complex AI cannot beat this simple baseline, we know the fancy AI is not worth using." },
          { term: 'Interpretable', definition: 'Because the math is so simple, a human can look exactly at the formula and explain precisely why the model made its decision.' },
        ],
      };

    case 'wide_and_deep':
      return {
        title: 'Wide & Deep',
        summary: 'Combines memorization from a linear path with generalization from a deep path.',
        flowType: 'wide_deep',
        nodes: [],
        layers: [
          "Wide Branch: A fast, direct path that memorizes simple, obvious rules (like 'if burger, suggest fries').",
          'Deep Branch Blocks: A more complex thinking path that tries to learn deeper, creative generalizations.',
          'Merge Layer: The part where the AI glues together the obvious rules and the deep creative thoughts into one final answer.',
        ],
        terminology: [
          { term: 'Memorization (Wide)', definition: "The network's ability to memorize facts it has seen many times. For example, 'if order includes a burger, suggest fries'." },
          { term: 'Generalization (Deep)', definition: 'The network\'s ability to guess correctly on things it has never seen before based on underlying patterns.' },
          { term: 'Concatenates', definition: 'A programming word for gluing two different lists of numbers together into one big list side by side.' },
        ],
      };

    case 'tabresnet':
      return {
        title: 'TabResNet',
        summary: 'Residual skip connections improve gradient flow and stability in deeper tabular networks.',
        flowType: 'sequential',
        nodes: [
          { label: 'Input Features', bg: '#f4f4f5', border: '#d4d4d8' },
          { label: 'Input Projection', bg: '#eef2ff', border: '#a5b4fc' },
          { label: 'Residual Block 1', bg: '#f5f3ff', border: '#c4b5fd' },
          { label: 'Residual Block 2', bg: '#f5f3ff', border: '#c4b5fd' },
          { label: outputLabel, bg: '#ecfeff', border: '#67e8f9' },
        ],
        layers: [
          "Input Projection: Upgrades your basic data into a richer format so the AI has more mathematical 'room' to untangle messy patterns.",
          'Residual Blocks: The core learning steps. They use skip connections, meaning if a step is confusing, the data can safely bypass it.',
          'Output Head: Takes all those careful, step-by-step corrections and turns them into a final prediction.',
        ],
        terminology: [
          { term: 'Skip Connections', definition: "A shortcut bypass. If a layer's complicated math is not helping, the model can skip over it, which helps deeper systems train cleanly." },
          { term: 'Hidden Space / Higher-Dimensional', definition: 'Imagine taking a 2D drawing and popping it out into a 3D sculpture. The model stretches your data into many dimensions so it has more room to untangle messy patterns.' },
          { term: 'Residual', definition: 'Instead of learning the entire final answer from scratch at every step, each layer learns the tiny remaining correction needed to fix the previous step.' },
        ],
      };

    case 'imbalance_aware':
      return {
        title: 'Imbalance-Aware Classifier',
        summary: 'Same network family with weighted objective to prioritize minority classes.',
        flowType: 'sequential',
        nodes: [
          { label: 'Input Features', bg: '#f4f4f5', border: '#d4d4d8' },
          { label: 'Dense Encoder', bg: '#f0f9ff', border: '#7dd3fc' },
          { label: 'Classifier Head', bg: '#fef3c7', border: '#fcd34d' },
          { label: 'Class-Weighted Loss', bg: '#fee2e2', border: '#fca5a5' },
          { label: 'Class Probabilities', bg: '#ecfeff', border: '#67e8f9' },
        ],
        layers: [
          "Dense Encoder: Translates your raw spreadsheet data into a machine language the AI can easily understand.",
          'Classifier Head: Looks at that translated data and outputs confidence scores for each possible category.',
          'Class-Weighted Loss Path: A harsh grading system that severely punishes the AI if it misses rare events it was supposed to catch.',
        ],
        terminology: [
          { term: 'Minority Class', definition: 'A scenario where the thing you are looking for is extremely rare, such as credit card fraud among normal transactions.' },
          { term: 'Class-Weighted Loss', definition: 'We heavily penalize the AI if it misses the rare event, so it pays more attention to uncommon but important outcomes.' },
          { term: 'Encoder', definition: "The part of the AI that reads your raw data and 'encodes' it into an internal representation the rest of the model can use." },
        ],
      };

    case 'quantile_regression':
      return {
        title: 'Quantile Regression',
        summary: 'Predicts distribution quantiles (P80 shown) instead of only a mean point estimate.',
        flowType: 'sequential',
        nodes: [
          { label: 'Input Features', bg: '#f4f4f5', border: '#d4d4d8' },
          { label: 'Dense Encoder', bg: '#f0f9ff', border: '#7dd3fc' },
          { label: 'Quantile Head (tau=0.8)', bg: '#ede9fe', border: '#c4b5fd' },
          { label: 'Pinball Loss', bg: '#fef3c7', border: '#fcd34d' },
          { label: 'P80 Forecast', bg: '#ecfeff', border: '#67e8f9' },
        ],
        layers: [
          "Dense Encoder: Looks at your data and builds internal latent clues that help it make a guess.",
          'Quantile Head (tau=0.8): Instead of making one exact guess, it draws a boundary line where it is 80% sure the real answer will be below it.',
          'Pinball Loss Path: A special grading system that punishes the AI differently for guessing too low versus too high.',
        ],
        terminology: [
          { term: 'Quantile Regression', definition: "Instead of just guessing the average expected outcome, this model guesses specific boundary lines, such as '80% should land below this forecast'." },
          { term: 'Tau', definition: 'A symbol representing the specific boundary line we want to draw. A tau of 0.8 means the 80th percentile mark.' },
          { term: 'Pinball Loss', definition: 'A grading system that can punish underestimates and overestimates differently, depending on the quantile you care about.' },
          { term: 'Latent Clues', definition: 'Hidden, underlying features the AI figures out on its own to help solve the task.' },
        ],
      };

    case 'calibrated_classifier':
      return {
        title: 'Calibrated Classifier',
        summary: 'Adds label smoothing in training to reduce overconfident predictions and improve probability behavior.',
        flowType: 'sequential',
        nodes: [
          { label: 'Input Features', bg: '#f4f4f5', border: '#d4d4d8' },
          { label: 'Dense Encoder', bg: '#f0f9ff', border: '#7dd3fc' },
          { label: 'Classifier Head', bg: '#fef3c7', border: '#fcd34d' },
          { label: 'Label Smoothing Objective', bg: '#dcfce7', border: '#86efac' },
          { label: 'Calibrated Probabilities', bg: '#ecfeff', border: '#67e8f9' },
        ],
        layers: [
          "Dense Encoder: Organizes the data so the AI can easily draw clean boundaries between categories.",
          'Classifier Head: Outputs exactly how confident it is about its guess as a percentage.',
          'Label Smoothing Objective: The training rule that forces the AI to remain slightly unsure, preventing overconfident mistakes.',
        ],
        terminology: [
          { term: 'Calibration', definition: 'Ensuring the AI is not falsely confident. If it says 90% likely, it should be right around 9 out of 10 times.' },
          { term: 'Label Smoothing', definition: 'A trick to stop the AI from acting too strictly by preventing it from being 100% certain about training labels.' },
          { term: 'Linearly Separate', definition: "Pushing the data around in mathematical space until you can draw a straight line between the 'yes' answers and the 'no' answers." },
        ],
      };

    case 'entity_embeddings':
      return {
        title: 'Entity Embeddings',
        summary: 'Projects sparse/tabular inputs into compact latent features before final prediction.',
        flowType: 'sequential',
        nodes: [
          { label: 'Input Features', bg: '#f4f4f5', border: '#d4d4d8' },
          { label: 'Embedding Projection Layer', bg: '#eef2ff', border: '#a5b4fc' },
          { label: 'Dense Predictor Stack', bg: '#f0f9ff', border: '#7dd3fc' },
          { label: outputLabel, bg: '#ecfeff', border: '#67e8f9' },
        ],
        layers: [
          "Embedding Projection Layer: Takes hard-to-read categories and turns them into easier latent coordinates on a map.",
          'Dense Predictor Stack: Uses those map coordinates to find deeper patterns useful for predicting your specific goal.',
          'Output Head: Takes those final thought patterns and turns them into your actual answer.',
        ],
        terminology: [
          { term: 'Categorical IDs', definition: 'Data that exists as distinct groups or labels rather than numbers, like zip codes, user IDs, or product brands.' },
          { term: 'Embeddings / Latent Coordinates', definition: 'A trick that turns words or categories into hidden coordinates on a map, letting similar categories sit near each other.' },
          { term: 'Sparse', definition: 'A situation where most of your data is zeros or empty blank spaces.' },
        ],
      };

    case 'autoencoder_head':
      return {
        title: 'Autoencoder + Head',
        summary: 'Jointly learns a compact latent representation and a supervised prediction path.',
        flowType: 'autoencoder',
        nodes: [],
        layers: [
          "Encoder: Squashes all your data down into a tiny, compact summary called a latent representation.",
          'Latent Bottleneck: The narrow point the data gets squeezed through, forcing the AI to drop noise and keep what matters.',
          'Reconstruction Head: A separate path that tries to unpack that summary back into its full original form to make sure important details were not lost.',
          'Prediction Head: Uses that same clear, noise-free summary to predict what you want to know.',
        ],
        terminology: [
          { term: 'Autoencoder', definition: 'An AI trained to compress data and then rebuild it, forcing the model to learn a useful internal summary.' },
          { term: 'Bottleneck / Latent Representation', definition: 'The tight compressed code that forces the model to keep only the most useful structure.' },
          { term: 'Reconstruction / Decoder', definition: 'The part of the AI whose job is to unpack the compressed code back into the original shape.' },
        ],
      };

    case 'multi_task_learning':
      return {
        title: 'Multi-Task Learning',
        summary: 'Trains one shared representation with multiple supervised heads.',
        flowType: 'multi_task',
        nodes: [],
        layers: [
          'Shared Trunk: The foundational base of the AI. It does the heavy lifting to learn basic patterns before branching off.',
          "Main Task Head: The part of the AI's brain that focuses entirely on answering your primary question.",
          'Auxiliary Head: A side-task path. By forcing the AI to solve this extra related problem, it can get better at the main task.',
        ],
        terminology: [
          { term: 'Multi-Task', definition: 'Training the AI to solve two related problems at the same time using the same shared representation.' },
          { term: 'Shared Trunk', definition: "The base foundational layers of the AI. Like a tree trunk, it does the heavy lifting before branching into separate heads." },
          { term: 'Auxiliary Supervision', definition: 'An extra training objective that exists to force the AI to learn better foundational patterns.' },
        ],
      };

    case 'time_aware_tabular':
      return {
        title: 'Time-Aware Tabular',
        summary: 'Applies temporal gating before deep prediction to emphasize time-derived patterns.',
        flowType: 'time_aware',
        nodes: [],
        layers: [
          "Temporal Gate: Looks at the clock or calendar and decides which time-related clues are important right now.",
          'Gated Features: The cleaned-up data where important time clues are highlighted and irrelevant ones are muted.',
          'Concat Raw + Gated: Glues your regular data and your time clues together so the final AI step can use both at once.',
        ],
        terminology: [
          { term: 'Temporal', definition: 'Related to time, such as noticing that ice cream sales behave differently in August versus December.' },
          { term: 'Gating Mechanism', definition: 'A learned gate that lets relevant seasonal information pass through and blocks noisy time data.' },
          { term: 'Seasonality', definition: 'Repeating patterns that naturally loop on a schedule, like higher retail sales every Friday.' },
        ],
      };

    case 'tree_teacher_distillation':
      return {
        title: 'Tree-Teacher Distillation',
        summary: 'A tree teacher guides a compact neural student during training.',
        flowType: 'distillation',
        nodes: [],
        layers: [
          'Tree Teacher Ensemble: A large forest model, made of many decision trees, that already understands the data.',
          "Neural Student: A tiny, fast AI that is trying to copy the teacher's behavior so it can run cheaply and quickly.",
          "Teacher-to-Student Distillation Path: The grading system. The tiny student is graded on how closely it mimics the teacher's doubts and guesses.",
        ],
        terminology: [
          { term: 'Knowledge Distillation', definition: 'A process where a large teacher model trains a smaller student model to emulate its behavior.' },
          { term: 'Tree Ensemble / Forest Model', definition: 'An AI made from many decision trees voting together on an answer.' },
          { term: 'Soft Probabilities', definition: "Instead of only passing the final answer, the teacher passes nuanced confidence values that help the student learn faster." },
        ],
      };

    default:
      return {
        title: framework === 'pytorch' ? 'PyTorch Neural Net' : 'TensorFlow Neural Net',
        summary: 'Dense neural network baseline.',
        flowType: 'sequential',
        nodes: [
          { label: 'Input Features', bg: '#f4f4f5', border: '#d4d4d8' },
          { label: outputLabel, bg: '#ecfeff', border: '#67e8f9' },
        ],
        layers: [
          'Input Features: The raw columns of data, properly prepared so the AI can read them.',
          "Model Core: The internal brain where hidden patterns are discovered.",
          'Output Head: The final step that turns discovered patterns into your actual prediction.',
        ],
        terminology: [
          { term: 'Features', definition: "The individual columns of data you feed the AI, such as age, height, or zip code." },
          { term: 'Tabular', definition: 'Data that lives in traditional rows and columns, like a spreadsheet or SQL database.' },
          { term: 'Prediction', definition: 'The final guess the AI produces after processing the data.' },
        ],
      };
  }
}

@Component({
  selector: 'app-model-preview-modal',
  imports: [CommonModule, ModalComponent],
  template: `
    <app-modal [isOpen]="isOpen" [title]="title()" position="top" (close)="close()">
      <div class="model-preview">
        <p class="model-preview-summary">{{ meta().summary }}</p>

        <div class="model-flow">
          @switch (meta().flowType) {
            @case ('wide_deep') {
              <div class="model-flow-branch">
                <span class="model-flow-node">Input Features</span>
                <span class="model-flow-arrow">→</span>
                <span class="model-flow-split">
                  <span class="model-flow-node branch-red">Wide Branch (Linear)</span>
                  <span class="model-flow-node branch-blue">Deep Dense Blocks (1 &amp; 2)</span>
                </span>
                <span class="model-flow-arrow">→</span>
                <span class="model-flow-node branch-green">Merge + Prediction Output</span>
              </div>
            }
            @case ('autoencoder') {
              <div class="model-flow-branch">
                <span class="model-flow-node">Input Features</span>
                <span class="model-flow-arrow">→</span>
                <span class="model-flow-node branch-indigo">Encoder</span>
                <span class="model-flow-arrow">→</span>
                <span class="model-flow-node branch-purple">Bottleneck</span>
                <span class="model-flow-arrow">→</span>
                <span class="model-flow-split">
                  <span class="model-flow-node branch-green">Reconstruction Head</span>
                  <span class="model-flow-node branch-yellow">Prediction Head</span>
                </span>
              </div>
            }
            @case ('multi_task') {
              <div class="model-flow-branch">
                <span class="model-flow-node">Input Features</span>
                <span class="model-flow-arrow">→</span>
                <span class="model-flow-node branch-blue">Shared Trunk</span>
                <span class="model-flow-arrow">→</span>
                <span class="model-flow-split">
                  <span class="model-flow-node branch-yellow">Main Task Head</span>
                  <span class="model-flow-node branch-green">Auxiliary Head</span>
                </span>
                <span class="model-flow-arrow">→</span>
                <span class="model-flow-node branch-cyan">Prediction Output</span>
              </div>
            }
            @case ('time_aware') {
              <div class="model-flow-branch">
                <span class="model-flow-node">Input Features</span>
                <span class="model-flow-arrow">→</span>
                <span class="model-flow-split">
                  <span class="model-flow-node branch-red">Temporal Gate</span>
                  <span class="model-flow-node branch-pink">Gated Features</span>
                </span>
                <span class="model-flow-arrow">→</span>
                <span class="model-flow-node branch-indigo">Concat Raw + Gated</span>
                <span class="model-flow-arrow">→</span>
                <span class="model-flow-node branch-cyan">Prediction Output</span>
              </div>
            }
            @case ('distillation') {
              <div class="model-flow-branch">
                <span class="model-flow-node">Input Features</span>
                <span class="model-flow-arrow">→</span>
                <span class="model-flow-split">
                  <span class="model-flow-node branch-green">Tree Teacher Ensemble</span>
                  <span class="model-flow-node branch-blue">Neural Student</span>
                </span>
                <span class="model-flow-arrow">→</span>
                <span class="model-flow-node branch-cyan">Prediction Output</span>
              </div>
            }
            @default {
              <div class="model-flow-sequential">
                @for (node of meta().nodes; track node.label; let last = $last) {
                  <span class="model-flow-node" [style.background]="node.bg" [style.border-color]="node.border">{{ node.label }}</span>
                  @if (!last) {
                    <span class="model-flow-arrow">→</span>
                  }
                }
              </div>
            }
          }
        </div>

        <div class="model-details-grid">
          <section class="model-detail-card">
            <p class="model-detail-title">Layer Architecture Breakdown</p>
            <ul class="model-detail-list">
              @for (layer of meta().layers; track layer) {
                <li>
                  <span class="model-detail-dot"></span>
                  <span class="model-detail-term">{{ parseBullet(layer).term }}</span>
                  <span class="model-detail-definition">{{ parseBullet(layer).definition }}</span>
                </li>
              }
            </ul>
          </section>

          @if (meta().terminology.length > 0) {
            <section class="model-detail-card terminology-card">
              <p class="model-detail-title">Key Terminology</p>
              <ul class="model-detail-list">
                @for (item of meta().terminology; track item.term) {
                  <li>
                    <span class="model-detail-dot blue"></span>
                    <span class="model-detail-term">{{ item.term }}</span>
                    <span class="model-detail-definition">{{ item.definition }}</span>
                  </li>
                }
              </ul>
            </section>
          } @else {
            <section class="model-detail-card empty-terminology">
              <p>No complex terminology explicitly defined for this baseline model.</p>
            </section>
          }
        </div>
      </div>
    </app-modal>
  `,
})
export class ModelPreviewModalComponent {
  @Input() isOpen = false;
  @Input() close: () => void = () => {};

  private readonly frameworkValue = signal<Framework>('pytorch');
  private readonly modeValue = signal('linear_glm_baseline');

  @Input()
  set framework(value: Framework) {
    this.frameworkValue.set(value ?? 'pytorch');
  }

  get framework(): Framework {
    return this.frameworkValue();
  }

  @Input()
  set mode(value: string) {
    this.modeValue.set(value || 'linear_glm_baseline');
  }

  get mode(): string {
    return this.modeValue();
  }

  readonly meta = computed(() => modelMeta(this.modeValue(), this.frameworkValue()));
  readonly title = computed(() => `${this.meta().title} (${this.frameworkValue() === 'pytorch' ? 'PyTorch' : 'TensorFlow'})`);

  parseBullet(bullet: string): { term: string; definition: string } {
    const idx = bullet.indexOf(':');
    if (idx < 0) return { term: bullet.trim(), definition: '' };
    return {
      term: bullet.slice(0, idx).trim(),
      definition: bullet.slice(idx + 1).trim(),
    };
  }
}
