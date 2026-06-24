import { CommonModule } from '@angular/common';
import { Component, Input, OnChanges, OnDestroy, SimpleChanges, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DataTableComponent } from '../../../shared/components/data-table.component';
import { DatasetComboboxComponent } from '../../../shared/components/dataset-combobox.component';
import { MlOrchestratorRegistryService } from '../../../shared/state/ml-orchestrator-registry.service';
import type { Framework } from '../model/ml-training.types';
import { TrainingScreenOrchestrator } from '../orchestrator/training-screen-orchestrator.service';
import { DistillMetricsModalComponent } from './distill-metrics-modal.component';
import { ModelPreviewModalComponent } from './model-preview-modal.component';
import { OptimalParamsModalComponent } from './optimal-params-modal.component';
import { TrainingRunsTableComponent } from './training-runs-table.component';

@Component({
  selector: 'app-training-screen',
  imports: [
    CommonModule,
    FormsModule,
    DatasetComboboxComponent,
    DataTableComponent,
    TrainingRunsTableComponent,
    ModelPreviewModalComponent,
    OptimalParamsModalComponent,
    DistillMetricsModalComponent,
  ],
  providers: [TrainingScreenOrchestrator],
  template: `
    <div class="page-shell" style="background: #fff;">
      <main class="page-main">
        <div class="page-content">
          <p class="page-kicker">Machine Learning with {{ frameworkLabel }}</p>

          <div class="stack-tight" style="max-width: 36rem;">
            <p class="page-kicker">Dataset (CSV/XLS/XLSX)</p>
            <app-dataset-combobox
              [options]="model.datasetOptions()"
              [selectedId]="model.selectedDatasetId()"
              (changed)="model.onDatasetChange($event)"
            />
            @if (model.datasetError()) {
              <p class="error-text">{{ model.datasetError() }}</p>
            }
          </div>

          <section class="section-card">
            <p class="page-kicker">Training Algorithm</p>
            <div class="form-grid" style="margin-top: .75rem;">
              <label class="field">
                <span>Select the machine learning architecture to run for this dataset.</span>
                <select [ngModel]="model.trainingMode()" (ngModelChange)="model.trainingMode.set($event)">
                  @for (option of trainingModes; track option.value) {
                    <option [value]="option.value">{{ option.label }}</option>
                  }
                </select>
                <button type="button" class="btn primary" style="align-self: flex-start;" (click)="model.isModelPreviewOpen.set(true)">Show Model</button>
              </label>
              <div class="blue-note" style="grid-column: span 2;">
                <p><strong>What it is:</strong> {{ model.modeExplainer().what }}</p>
                <p><strong>Why it's unique:</strong> {{ model.modeExplainer().why }}</p>
                <p><strong>Distillation Note:</strong> {{ model.modeExplainer().distillationNote }}</p>
              </div>
            </div>
          </section>

          <section class="section-card">
            <p class="page-kicker">Train {{ frameworkLabel }} Model</p>
            <div class="form-grid" style="margin-top: .75rem;">
              <label class="field">
                <span>Target Column</span>
                <select [ngModel]="model.targetColumn()" (ngModelChange)="model.targetColumn.set($event)">
                  <option value="">{{ model.defaults().targetColumn || 'Select target column' }}</option>
                  @for (column of model.tableColumns(); track column) {
                    <option [value]="column">{{ column }}</option>
                  }
                </select>
              </label>
              <label class="field">
                <span>Task</span>
                <select [ngModel]="model.task()" (ngModelChange)="model.task.set($event)">
                  <option value="auto">auto</option>
                  <option value="classification">classification</option>
                  <option value="regression">regression</option>
                </select>
              </label>
              <label class="field"><span>Epoch Values</span><input [ngModel]="model.epochValues()" (ngModelChange)="model.epochValues.set($event)" placeholder="e.g. 10,20,50" /></label>
              <label class="field"><span>Batch Sizes</span><input [ngModel]="model.batchSizes()" (ngModelChange)="model.batchSizes.set($event)" placeholder="e.g. 32,64" /></label>
              <label class="field"><span>Learning Rates</span><input [ngModel]="model.learningRates()" (ngModelChange)="model.learningRates.set($event)" placeholder="e.g. 0.001" /></label>
              <label class="field"><span>Test Sizes</span><input [ngModel]="model.testSizes()" (ngModelChange)="model.testSizes.set($event)" placeholder="e.g. 0.2" /></label>
              <label class="field" [class.muted]="model.isLinearBaseline()"><span>Hidden Dims</span><input [disabled]="model.isLinearBaseline()" [ngModel]="model.hiddenDims()" (ngModelChange)="model.hiddenDims.set($event)" /></label>
              <label class="field" [class.muted]="model.isLinearBaseline()"><span>Hidden Layers</span><input [disabled]="model.isLinearBaseline()" [ngModel]="model.numHiddenLayers()" (ngModelChange)="model.numHiddenLayers.set($event)" /></label>
              <label class="field" [class.muted]="model.isLinearBaseline()"><span>Dropouts</span><input [disabled]="model.isLinearBaseline()" [ngModel]="model.dropouts()" (ngModelChange)="model.dropouts.set($event)" /></label>
              <label class="field">
                <span>Exclude Columns</span>
                <input [ngModel]="model.excludeColumns()" (ngModelChange)="model.excludeColumns.set($event)" placeholder="e.g. customerID,Order,PID" />
                <span class="field-hint">Preloaded: {{ model.defaults().excludeColumns.length ? model.defaults().excludeColumns.join(', ') : '(none)' }}</span>
              </label>
              <label class="field">
                <span>Date Columns</span>
                <input [ngModel]="model.dateColumns()" (ngModelChange)="model.dateColumns.set($event)" placeholder="e.g. Date" />
                <span class="field-hint">Preloaded: {{ model.defaults().dateColumns.length ? model.defaults().dateColumns.join(', ') : '(none)' }}</span>
              </label>
            </div>

            <div class="two-grid" style="margin-top: .75rem; font-size: .75rem;">
              <p [class.error-text]="!model.epochsValidation().ok">Epochs: {{ validationText(model.epochsValidation()) }}</p>
              <p [class.error-text]="!model.batchSizesValidation().ok">Batch sizes: {{ validationText(model.batchSizesValidation()) }}</p>
              <p [class.error-text]="!model.learningRatesValidation().ok">Learning rates: {{ validationText(model.learningRatesValidation()) }}</p>
              <p [class.error-text]="!model.testSizesValidation().ok">Test sizes: {{ validationText(model.testSizesValidation()) }}</p>
              <p [class.error-text]="!model.isLinearBaseline() && !model.hiddenDimsValidation().ok">Hidden dims: {{ model.isLinearBaseline() ? 'n/a (linear baseline)' : validationText(model.hiddenDimsValidation()) }}</p>
              <p [class.error-text]="!model.isLinearBaseline() && !model.numHiddenLayersValidation().ok">Hidden layers: {{ model.isLinearBaseline() ? 'n/a (linear baseline)' : validationText(model.numHiddenLayersValidation()) }}</p>
              <p [class.error-text]="!model.isLinearBaseline() && !model.dropoutsValidation().ok">Dropouts: {{ model.isLinearBaseline() ? 'n/a (linear baseline)' : validationText(model.dropoutsValidation()) }}</p>
            </div>

            <div class="button-row" style="margin-top: 1.25rem;">
              <button type="button" class="btn primary" [disabled]="model.isTrainDisabled()" (click)="model.onTrain()">
                {{ model.isTraining() ? 'Training ' + model.trainingProgress().current + '/' + model.trainingProgress().total + '...' : 'Train Model' }}
              </button>
              <div class="stack-tight" style="gap: .1rem;">
                <span class="muted">Dataset: <code>{{ model.selectedDatasetId() ?? 'none' }}</code></span>
                <span class="strong-red">Planned runs: {{ model.plannedRunCount() }}</span>
              </div>
            </div>
            @if (model.trainingError()) {
              <p class="error-text">{{ model.trainingError() }}</p>
            }

            <div style="margin-top: 1.5rem; border-top: 1px solid #e4e4e7; padding-top: 1.25rem;">
              <div class="optional-settings-grid">
                <div class="section-card optional-settings-card">
                  <p class="optional-settings-title"><strong>Bayesian Optimization</strong></p>
                  <p class="optional-settings-copy">What is it: A method for optimizing expensive black-box functions by using a probabilistic model to choose promising parameter settings.</p>
                  <p class="optional-settings-copy">How it works: Uses completed runs to suggest the next promising hyperparameter combination. <strong class="error-text" style="text-decoration: underline;">Requires at least 5 completed runs.</strong></p>
                  <button type="button" class="btn secondary" [disabled]="model.isTraining() || model.completedRuns().length < 5" (click)="model.onFindOptimalParamsClick()">Find Optimal Params</button>
                  @if (model.optimizerStatus()) {
                    <span class="muted" style="margin-left: .5rem;">{{ model.optimizerStatus() }}</span>
                  }
                </div>
                <div class="section-card optional-settings-card optional-settings-card-right">
                  <label class="optional-settings-label">
                    <input type="checkbox" [ngModel]="model.sweepEnabled()" (ngModelChange)="model.toggleRunSweep($event)" [disabled]="model.isTraining()" />
                    <strong>Set Sweep Values</strong>
                  </label>
                  <div class="optional-settings-row">
                    <button type="button" class="btn secondary" [disabled]="!model.sweepEnabled() || model.isTraining()" (click)="model.reloadSweepValues()">Reload</button>
                    <span class="muted" style="font-size: .75rem;">Toggle ON to apply sweep values. Use Reload for fresh sweep values.</span>
                  </div>
                  <label class="optional-settings-label optional-settings-label-bordered">
                    <input type="checkbox" [ngModel]="model.autoDistill()" (ngModelChange)="model.autoDistill.set($event)" [disabled]="model.isTraining()" />
                    <span style="display: flex; flex-direction: column;">
                      <strong>Auto-distill Training Runs</strong>
                      <span class="muted" style="font-size: .75rem;">Smaller distilled models are created automatically during training sweeps.</span>
                    </span>
                  </label>
                  @if (model.distillStatus()) {
                    <p class="muted" style="margin: 0;">{{ model.distillStatus() }}</p>
                  }
                </div>
              </div>
            </div>
          </section>

          <section style="border-top: 1px solid #e4e4e7; padding-top: 1rem;">
            <div class="button-row" style="justify-content: space-between;">
              <p class="page-kicker">Training Runs</p>
              <div class="button-row">
                @if (model.copyRunsStatus()) {
                  <span class="muted">{{ model.copyRunsStatus() }}</span>
                }
                <button type="button" class="btn secondary" [disabled]="model.trainingRuns().length === 0" (click)="model.onCopyResults()">Copy Results</button>
                <button type="button" class="btn primary" [disabled]="model.trainingRuns().length === 0" (click)="model.clearRuns()">Clear Runs</button>
                <button type="button" class="btn danger" [disabled]="!model.isTraining() || model.stopTraining()" (click)="model.stop()">
                  {{ model.stopTraining() ? 'Stop Requested...' : 'Stop Training Runs' }}
                </button>
              </div>
            </div>
            @if (model.stopTraining()) {
              <p class="warning-text">Stop requested. Current run will finish, then remaining runs are canceled.</p>
            }
            @if (model.trainingRuns().length === 0) {
              <p class="muted">No runs yet. Train once to populate the results table.</p>
            } @else {
              <app-training-runs-table
                [runs]="model.trainingRuns()"
                [framework]="framework"
                [distillingTeacherKey]="model.distillingTeacherKey()"
                [distilledByTeacher]="model.distilledByTeacher()"
                (distill)="model.onDistillFromRun($event)"
                (seeDistilled)="model.onSeeDistilledFromRun($event)"
              />
            }
          </section>

          <details class="details-card">
            <summary>Preprocessing Notes</summary>
            <div class="stack-tight" style="margin-top: .75rem;">
              <p><strong>Categorical Encoding:</strong> Text columns with &lt;= 20 unique values are automatically One-Hot Encoded.</p>
              <p><strong>High Cardinality &amp; IDs:</strong> Text columns with &gt; 20 unique values or ID-like names are dropped.</p>
              <p><strong>Date Parsing:</strong> Dates and timestamps are extracted into Year, Month, and Day numeric features.</p>
              <p><strong>Missing Values:</strong> Missing numeric values are imputed using the column median.</p>
              <p><strong>Feature Scaling:</strong> All features are standardized before analysis.</p>
            </div>
          </details>

          <details class="section-card" open>
            <summary style="cursor: pointer;" class="page-kicker">Dataset Table Preview</summary>
            <p class="muted">Showing {{ model.tableRows().length }} rows for <code>{{ model.selectedDatasetId() ?? 'no selection' }}</code>.</p>
            <app-data-table [rows]="model.tableRows()" [columns]="model.tableColumns()" />
          </details>
        </div>
      </main>

      <app-model-preview-modal
        [isOpen]="model.isModelPreviewOpen()"
        [framework]="framework"
        [mode]="model.trainingMode()"
        [close]="closeModelPreview"
      />
      <app-optimal-params-modal
        [isOpen]="model.isOptimalModalOpen()"
        [pendingOptimalParams]="model.pendingOptimalParams()"
        [pendingOptimalPrediction]="model.pendingOptimalPrediction()"
        [activeAlgorithm]="model.trainingMode()"
        [close]="closeOptimal"
        [apply]="applyOptimal"
      />
      <app-distill-metrics-modal
        [isOpen]="model.isDistillMetricsModalOpen()"
        [distillMetrics]="model.distillMetrics()"
        [distillModelId]="model.distillModelId()"
        [distillModelPath]="model.distillModelPath()"
        [distillComparison]="model.distillComparison()"
        [close]="closeDistill"
      />
    </div>
  `
})
export class TrainingScreenComponent implements OnChanges, OnDestroy {
  @Input() framework: Framework = 'pytorch';
  readonly model = inject(TrainingScreenOrchestrator);
  private readonly mlRegistry = inject(MlOrchestratorRegistryService);

  readonly closeModelPreview = () => this.model.isModelPreviewOpen.set(false);
  readonly closeOptimal = () => this.model.isOptimalModalOpen.set(false);
  readonly applyOptimal = () => this.model.onApplyOptimalParams();
  readonly closeDistill = () => this.model.isDistillMetricsModalOpen.set(false);

  get frameworkLabel(): string {
    return this.framework === 'pytorch' ? 'PyTorch' : 'TensorFlow';
  }

  get trainingModes() {
    return this.framework === 'pytorch'
      ? [
          { value: 'linear_glm_baseline', label: 'linear/glm baseline' },
          { value: 'mlp_dense', label: 'neural net (dense)' },
          { value: 'tabresnet', label: 'tabresnet (residual mlp)' },
          { value: 'imbalance_aware', label: 'imbalance-aware classifier' },
          { value: 'calibrated_classifier', label: 'calibrated classifier' },
          { value: 'tree_teacher_distillation', label: 'tree-teacher distillation' },
        ]
      : [
          { value: 'wide_and_deep', label: 'wide & deep' },
          { value: 'entity_embeddings', label: 'entity embeddings' },
          { value: 'autoencoder_head', label: 'autoencoder + head' },
          { value: 'quantile_regression', label: 'quantile regression (p80)' },
          { value: 'multi_task_learning', label: 'multi-task learning' },
          { value: 'time_aware_tabular', label: 'time-aware tabular' },
        ];
  }

  ngOnChanges(_changes: SimpleChanges): void {
    this.model.configure({
      framework: this.framework,
      defaultTrainingMode: this.framework === 'pytorch' ? 'linear_glm_baseline' : 'wide_and_deep',
      defaultExcludeColumns: 'customerID',
    });
    this.mlRegistry.register(this.framework, this.model);
  }

  ngOnDestroy(): void {
    this.mlRegistry.unregister(this.framework);
  }

  validationText(validation: { ok: boolean; values?: unknown[]; error?: string }): string {
    return validation.ok ? (validation.values ?? []).join(', ') : validation.error ?? 'Invalid';
  }
}
