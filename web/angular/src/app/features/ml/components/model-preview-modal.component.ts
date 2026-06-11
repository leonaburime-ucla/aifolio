import { CommonModule } from '@angular/common';
import { Component, Input, computed } from '@angular/core';
import { ModalComponent } from '../../../shared/components/modal.component';
import type { Framework } from '../model/ml-training.types';

type ModelMeta = {
  title: string;
  summary: string;
  nodes: string[];
  layers: string[];
  terminology: Array<{ term: string; definition: string }>;
};

const META: Record<string, ModelMeta> = {
  linear_glm_baseline: {
    title: 'Linear / GLM Baseline',
    summary: 'Single linear head model. Fast and interpretable benchmark.',
    nodes: ['Input Features', 'Linear / GLM Head', 'Prediction Output'],
    layers: ['Input Features: cleaned tabular inputs.', 'Linear / GLM Head: weighted additive model.', 'Output Head: interpretable baseline prediction.'],
    terminology: [
      { term: 'GLM', definition: 'A generalized linear model that combines weighted inputs into an output.' },
      { term: 'Baseline', definition: 'A simple benchmark used to judge whether more complex models are worthwhile.' },
    ],
  },
  mlp_dense: {
    title: 'Multi-Layer Perceptron (Dense)',
    summary: 'Standard dense neural network with fully connected hidden layers.',
    nodes: ['Input Features', 'Dense Hidden Block 1', 'Dense Hidden Block 2', 'Prediction Output'],
    layers: ['Input Features: cleaned numeric/categorical data.', 'Dense Hidden Blocks: nonlinear feature interactions.', 'Output Head: task-specific prediction.'],
    terminology: [
      { term: 'Latent State', definition: 'A learned hidden representation of the input data.' },
      { term: 'Logits', definition: 'Raw model scores before probability conversion.' },
    ],
  },
  tabresnet: {
    title: 'TabResNet',
    summary: 'Residual skip connections improve stability in deeper tabular networks.',
    nodes: ['Input Features', 'Input Projection', 'Residual Blocks', 'Prediction Output'],
    layers: ['Input Projection: maps inputs to hidden space.', 'Residual Blocks: learn corrections with skip paths.', 'Output Head: final prediction.'],
    terminology: [{ term: 'Skip Connection', definition: 'A shortcut that helps gradients move through deeper networks.' }],
  },
  imbalance_aware: {
    title: 'Imbalance-Aware Classifier',
    summary: 'Weighted objective prioritizes minority classes.',
    nodes: ['Input Features', 'Dense Encoder', 'Classifier Head', 'Class-Weighted Loss', 'Probabilities'],
    layers: ['Dense Encoder: tabular representation.', 'Classifier Head: class scores.', 'Weighted Loss: stronger penalty for missed rare cases.'],
    terminology: [{ term: 'Minority Class', definition: 'A rare class that can be under-learned without weighting.' }],
  },
  calibrated_classifier: {
    title: 'Calibrated Classifier',
    summary: 'Label smoothing reduces overconfident predictions.',
    nodes: ['Input Features', 'Dense Encoder', 'Classifier Head', 'Label Smoothing', 'Calibrated Probabilities'],
    layers: ['Dense Encoder: feature representation.', 'Classifier Head: probability output.', 'Label Smoothing: confidence regularization.'],
    terminology: [{ term: 'Calibration', definition: 'Predicted probabilities align with observed frequencies.' }],
  },
  tree_teacher_distillation: {
    title: 'Tree-Teacher Distillation',
    summary: 'Teacher ensemble transfers behavior into a smaller neural student.',
    nodes: ['Input Features', 'Tree Teacher', 'Neural Student', 'Prediction Output'],
    layers: ['Teacher Ensemble: source behavior.', 'Neural Student: compact approximation.', 'Distillation Loss: balances truth and teacher signal.'],
    terminology: [{ term: 'Distillation', definition: 'Training a smaller model to mimic a stronger teacher.' }],
  },
  wide_and_deep: {
    title: 'Wide & Deep',
    summary: 'Combines memorization from a linear path with generalization from a deep path.',
    nodes: ['Input Features', 'Wide Branch', 'Deep Branch', 'Merge', 'Prediction Output'],
    layers: ['Wide Branch: memorized rules.', 'Deep Branch: generalization.', 'Merge: combined prediction path.'],
    terminology: [{ term: 'Memorization', definition: 'Learning repeated feature interactions directly.' }],
  },
  entity_embeddings: {
    title: 'Entity Embeddings',
    summary: 'Projects sparse categorical inputs into compact latent features.',
    nodes: ['Input Features', 'Embedding Projection', 'Dense Predictor', 'Prediction Output'],
    layers: ['Embedding Projection: category vectors.', 'Dense Stack: prediction features.', 'Output Head: task result.'],
    terminology: [{ term: 'Embedding', definition: 'A learned vector representation for a category.' }],
  },
  autoencoder_head: {
    title: 'Autoencoder + Head',
    summary: 'Representation learning with reconstruction plus prediction objective.',
    nodes: ['Input Features', 'Encoder', 'Bottleneck', 'Reconstruction Head', 'Prediction Head'],
    layers: ['Encoder: compresses inputs.', 'Bottleneck: compact representation.', 'Heads: reconstruct and predict.'],
    terminology: [{ term: 'Bottleneck', definition: 'A compressed latent layer that forces signal extraction.' }],
  },
  quantile_regression: {
    title: 'Quantile Regression',
    summary: 'Predicts distribution quantiles instead of only a mean point estimate.',
    nodes: ['Input Features', 'Dense Encoder', 'Quantile Head', 'Pinball Loss', 'P80 Forecast'],
    layers: ['Dense Encoder: latent clues.', 'Quantile Head: percentile output.', 'Pinball Loss: asymmetric training objective.'],
    terminology: [{ term: 'Tau', definition: 'The requested quantile boundary, such as 0.8 for P80.' }],
  },
  multi_task_learning: {
    title: 'Multi-Task Learning',
    summary: 'A shared trunk supports multiple related prediction heads.',
    nodes: ['Input Features', 'Shared Trunk', 'Main Head', 'Auxiliary Head', 'Prediction Output'],
    layers: ['Shared Trunk: common representation.', 'Main Head: primary objective.', 'Auxiliary Head: related training signal.'],
    terminology: [{ term: 'Auxiliary Task', definition: 'A secondary objective used to improve shared representations.' }],
  },
  time_aware_tabular: {
    title: 'Time-Aware Tabular',
    summary: 'Temporal gates highlight date-derived feature structure.',
    nodes: ['Input Features', 'Temporal Gate', 'Gated Features', 'Prediction Output'],
    layers: ['Temporal Gate: learns date-aware weights.', 'Gated Features: adjusted tabular signal.', 'Output Head: prediction.'],
    terminology: [{ term: 'Temporal Gate', definition: 'A learned multiplier over time-derived features.' }],
  },
};

@Component({
  selector: 'app-model-preview-modal',
  imports: [CommonModule, ModalComponent],
  template: `
    <app-modal [isOpen]="isOpen" [title]="title()" position="top" (close)="close()">
      <div class="stack">
        <p class="muted">{{ meta().summary }}</p>
        <div class="flow-row">
          @for (node of meta().nodes; track node; let last = $last) {
            <span class="flow-node">{{ node }}</span>
            @if (!last) {
              <span class="muted">→</span>
            }
          }
        </div>
        <div class="two-grid">
          <div class="section-card">
            <p><strong>Layer Architecture Breakdown</strong></p>
            <ul>
              @for (layer of meta().layers; track layer) {
                <li>{{ layer }}</li>
              }
            </ul>
          </div>
          <div class="section-card">
            <p><strong>Key Terminology</strong></p>
            @if (meta().terminology.length > 0) {
              @for (item of meta().terminology; track item.term) {
                <p><strong>{{ item.term }}:</strong> {{ item.definition }}</p>
              }
            } @else {
              <p class="muted">No additional terminology for this model.</p>
            }
          </div>
        </div>
      </div>
    </app-modal>
  `
})
export class ModelPreviewModalComponent {
  @Input() isOpen = false;
  @Input() framework: Framework = 'pytorch';
  @Input() mode = 'linear_glm_baseline';
  @Input() close: () => void = () => {};

  readonly meta = computed(() => META[this.mode] ?? META['linear_glm_baseline']);
  readonly title = computed(() => `${this.meta().title} (${this.framework === 'pytorch' ? 'PyTorch' : 'TensorFlow'})`);
}
