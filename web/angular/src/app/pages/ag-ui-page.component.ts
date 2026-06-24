import { CommonModule } from '@angular/common';
import { Component, computed, effect, inject, OnDestroy, OnInit, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ChartsWorkspaceComponent } from '../features/recharts/components/charts-workspace.component';
import { AgenticResearchWorkspaceComponent } from '../features/agentic-research/components/agentic-research-workspace.component';
import { TrainingScreenComponent } from '../features/ml/components/training-screen.component';
import { ChartStoreService } from '../shared/state/chart-store.service';
import { MlOrchestratorRegistryService } from '../shared/state/ml-orchestrator-registry.service';
import { CopilotChat, registerFrontendTool, connectAgentContext, injectAgentStore } from '@copilotkit/angular';
import { handleSwitchAgUiTab, handleAddChartSpec, AG_UI_FALLBACK_MODELS, AG_UI_PREFERRED_MODEL_ID, resolveNextAgUiSelectedModelId, parseCopilotAssistantPayload, normalizeChartSpecInput, extractCopilotDisplayMessage, resolveMlFormPatchFromToolArgs } from '@aifolio/frontend-core/ag-ui';
import { handleAgenticSetActiveDataset, fetchAgenticDatasetManifest } from '@aifolio/frontend-core/agentic-research';
import { z } from 'zod';

type WorkspaceTab = 'charts' | 'agentic-research' | 'pytorch' | 'tensorflow';

interface ToolDef {
  name: string;
  description: string;
}

@Component({
  selector: 'app-ag-ui-page',
  imports: [
    CommonModule,
    FormsModule,
    ChartsWorkspaceComponent,
    AgenticResearchWorkspaceComponent,
    TrainingScreenComponent,
    CopilotChat,
  ],
  providers: [],
  template: `
    <div class="ag-ui-shell">
      <main class="ag-ui-main">
        <div class="ag-ui-content">

          <details class="ag-ui-details">
            <summary>What is AG-UI?</summary>
            <div class="ag-ui-details-body">
              <p>
                AG-UI is a protocol for agent-to-UI actions. Instead of only returning text, an LLM can call
                structured tools that mutate the interface: switch tabs, select datasets, clear/add charts, and
                trigger page workflows.
              </p>
              <p>
                CopilotKit is the runtime bridge that registers those frontend tools and executes them safely in
                this app. It maps model tool calls to typed handlers so chat can control real UI state.
              </p>
              <p>
                In this workspace, chat can orchestrate multi-step flows across tabs by combining navigation and
                feature-specific tools in sequence.
              </p>
              <p>
                References:
                <a href="https://github.com/ag-ui-protocol/ag-ui" target="_blank" rel="noreferrer">AG-UI</a>
                |
                <a href="https://github.com/CopilotKit/CopilotKit" target="_blank" rel="noreferrer">CopilotKit</a>
              </p>
            </div>
          </details>

          <p class="ag-ui-model-hint">For best results use Gemini 3.1 Pro Preview</p>

          <div class="ag-ui-tabs">
            @for (tab of tabs; track tab.id) {
              <button
                type="button"
                class="ag-ui-tab"
                [class.active]="activeTab() === tab.id"
                (click)="activeTab.set(tab.id)"
              >{{ tab.label }}</button>
            }
          </div>

          <div class="ag-ui-toolbar">
            <button type="button" class="btn primary" (click)="showTools.set(true)">Show Tools</button>
          </div>

          @if (showTools()) {
            <div class="ag-ui-modal-backdrop" (click)="showTools.set(false)">
              <div class="ag-ui-modal" (click)="$event.stopPropagation()">
                <div class="ag-ui-modal-header">
                  <h3>Available Tools — {{ activeTabLabel() }}</h3>
                  <button type="button" class="btn secondary" (click)="showTools.set(false)">✕</button>
                </div>
                <div class="ag-ui-modal-body">
                  <p class="muted">Tools are callable actions the model can invoke for this page to perform structured UI operations.</p>
                  <ul class="ag-ui-tool-list">
                    @for (tool of toolsForTab(); track tool.name) {
                      <li>
                        <p class="tool-name">{{ tool.name }}</p>
                        <p class="tool-desc">{{ tool.description }}</p>
                      </li>
                    }
                  </ul>
                </div>
              </div>
            </div>
          }

          @switch (activeTab()) {
            @case ('charts') {
              <app-charts-workspace />
            }
            @case ('agentic-research') {
              <details class="ag-ui-prompts" open>
                <summary>Show Sample Prompts</summary>
                <div class="ag-ui-prompts-body">
                  <p class="hint-red">Results take 1-2min</p>
                  <ol>
                    <li>Run PCA Transform</li>
                    <li>Run NMF Decomposition and PLSR</li>
                    <li>Change the dataset to fraud detection and run Random Forest</li>
                  </ol>
                </div>
              </details>
              <app-agentic-research-workspace [showPrompts]="false" [activeDatasetId]="activeDatasetId()" (datasetChange)="activeDatasetId.set($event)" />
            }
            @case ('pytorch') {
              <details class="ag-ui-prompts" open>
                <summary>Show Sample Prompts</summary>
                <div class="ag-ui-prompts-body">
                  <ol>
                    <li>Use the fraud detection dataset. Switch the training algorithm from neural net to TabResNet. Set batch sizes to 33 and 40, hidden dims to 64 and 96, and dropouts to 0.1 and 0.2.</li>
                    <li>Change from customer churn to fraud detection. Set task to classification, choose a different target column, set test sizes to 0.2 and 0.3, and start training runs.</li>
                    <li>Randomize PyTorch form fields with one value each, keep the current algorithm, and start training runs.</li>
                    <li>Switch the algorithm to calibrated classifier and set sweep values on.</li>
                  </ol>
                </div>
              </details>
              <app-training-screen framework="pytorch" />
            }
            @case ('tensorflow') {
              <details class="ag-ui-prompts" open>
                <summary>Show Sample Prompts</summary>
                <div class="ag-ui-prompts-body">
                  <ol>
                    <li>Use the house prices dataset. Switch the training algorithm from neural net to wide and deep. Set test sizes to 0.25 and 0.3, batch sizes to 32 and 64, and hidden dims to 128 and 256.</li>
                    <li>Change from customer churn to house prices. Set task to regression, set epochs to 20 and 40, and start training runs.</li>
                    <li>Randomize TensorFlow form fields with one value each, and keep the current algorithm.</li>
                    <li>Switch the algorithm to entity embeddings, and turn auto-distill on.</li>
                  </ol>
                </div>
              </details>
              <app-training-screen framework="tensorflow" />
            }
          }

        </div>
      </main>

      <div class="side-panel">
        <div class="side-panel-header">
          <h2>AIfolio Agent</h2>
          <select
            class="ag-ui-model-select"
            aria-label="Select AG-UI model"
            [ngModel]="selectedModelId()"
            (ngModelChange)="selectedModelId.set($event)"
          >
            @for (m of modelOptions(); track m.id) {
              <option [value]="m.id">{{ m.label }}</option>
            }
          </select>
        </div>
        <copilot-chat [agentId]="'agentic-research'" />
      </div>
    </div>
  `,
  styles: [`
    .ag-ui-shell {
      display: flex;
      flex-direction: row;
      height: calc(100dvh - 64px);
      overflow: hidden;
      background: #fafafa;
      color: #18181b;
    }
    .ag-ui-main {
      flex: 1;
      min-width: 0;
      overflow-y: auto;
      padding: .5rem 0;
    }
    .ag-ui-content {
      max-width: 64rem;
      margin: 0 auto;
      padding: 0 1.5rem;
      display: flex;
      flex-direction: column;
      gap: .75rem;
    }
    .side-panel {
      position: static;
      width: 420px;
      height: 100%;
      flex-shrink: 0;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      border-left: 1px solid #e4e4e7;
      background: #fff;
    }
    .ag-ui-details {
      border-radius: 1rem;
      border: 1px solid #e4e4e7;
      background: rgba(255,255,255,.7);
      padding: 1rem;
      backdrop-filter: blur(4px);
    }
    .ag-ui-details summary {
      cursor: pointer;
      font-size: .875rem;
      font-weight: 600;
    }
    .ag-ui-details-body {
      margin-top: .75rem;
      font-size: .875rem;
      color: #3f3f46;
      display: flex;
      flex-direction: column;
      gap: .75rem;
    }
    .ag-ui-details-body a {
      text-decoration: underline;
      text-underline-offset: 2px;
    }
    .ag-ui-model-hint {
      font-size: .875rem;
      font-weight: 600;
      color: #dc2626;
    }
    .side-panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: .75rem 1rem;
      border-bottom: 1px solid #e4e4e7;
    }
    .side-panel-header h2 {
      font-size: .875rem;
      font-weight: 600;
      color: #18181b;
      margin: 0;
    }
    .ag-ui-model-select {
      border-radius: .375rem;
      border: 1px solid #e4e4e7;
      background: #fff;
      padding: .25rem .5rem;
      font-size: .75rem;
      color: #3f3f46;
      box-shadow: 0 1px 2px rgba(0,0,0,.05);
    }
    .ag-ui-model-select:focus {
      outline: none;
      box-shadow: 0 0 0 2px #d4d4d8;
    }
    .ag-ui-tabs {
      position: sticky;
      top: 0;
      z-index: 20;
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: .5rem;
      padding: .5rem;
      border-radius: 1rem;
      border: 1px solid #e4e4e7;
      background: rgba(255,255,255,.9);
      backdrop-filter: blur(4px);
    }
    .ag-ui-tab {
      border-radius: .75rem;
      padding: .5rem .75rem;
      font-size: .875rem;
      font-weight: 500;
      background: #f4f4f5;
      color: #3f3f46;
      border: none;
      cursor: pointer;
      transition: background .15s, color .15s;
    }
    .ag-ui-tab:hover { background: #e4e4e7; }
    .ag-ui-tab.active { background: #18181b; color: #fff; }
    .ag-ui-toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .btn {
      border-radius: .375rem;
      padding: .375rem .75rem;
      font-size: .75rem;
      font-weight: 600;
      border: 1px solid;
      cursor: pointer;
    }
    .btn.primary { background: #059669; border-color: #059669; color: #fff; }
    .btn.primary:hover { background: #047857; }
    .btn.secondary { background: #fff; border-color: #d4d4d8; color: #3f3f46; }
    .btn.secondary:hover { background: #f4f4f5; }
    .ag-ui-modal-backdrop {
      position: fixed;
      inset: 0;
      z-index: 9999;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(0,0,0,.4);
      padding: 1rem;
    }
    .ag-ui-modal {
      width: 100%;
      max-width: 40rem;
      border-radius: .75rem;
      border: 1px solid #e4e4e7;
      background: #fff;
      box-shadow: 0 25px 50px -12px rgba(0,0,0,.25);
    }
    .ag-ui-modal-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: .75rem 1rem;
      border-bottom: 1px solid #e4e4e7;
    }
    .ag-ui-modal-header h3 { font-size: .875rem; font-weight: 600; }
    .ag-ui-modal-body {
      max-height: 60vh;
      overflow-y: auto;
      padding: .75rem 1rem;
    }
    .ag-ui-tool-list {
      list-style: none;
      padding: 0;
      margin: .75rem 0 0;
      display: flex;
      flex-direction: column;
      gap: .5rem;
    }
    .ag-ui-tool-list li {
      border-radius: .375rem;
      border: 1px solid #e4e4e7;
      padding: .5rem .75rem;
    }
    .tool-name { font-family: monospace; font-size: .75rem; color: #27272a; }
    .tool-desc { font-size: .75rem; color: #52525b; }
    .muted { font-size: .75rem; color: #52525b; }
    .ag-ui-prompts {
      border-radius: .5rem;
      border: 1px solid #e4e4e7;
      background: #fff;
      padding: .75rem 1rem;
    }
    .ag-ui-prompts summary {
      cursor: pointer;
      font-size: .75rem;
      font-weight: 600;
    }
    .ag-ui-prompts-body {
      margin-top: .75rem;
      font-size: .75rem;
    }
    .ag-ui-prompts-body ol {
      margin: .75rem 0 0;
      padding-left: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: .25rem;
    }
    .hint-red { font-weight: 700; color: #dc2626; }
    ::ng-deep .side-panel copilot-chat {
      display: flex;
      flex-direction: column;
      flex: 1;
      min-height: 0;
    }
    ::ng-deep .side-panel copilot-chat > * {
      display: contents;
    }
    ::ng-deep .side-panel copilot-chat-view {
      display: flex;
      flex-direction: column;
      flex: 1;
      min-height: 0;
    }
    ::ng-deep .side-panel .copilotKitChat {
      flex: 1 !important;
      min-height: 0;
      overflow: hidden;
    }
    ::ng-deep .side-panel .copilotKitChat > div {
      justify-content: flex-end !important;
    }
    ::ng-deep .side-panel copilot-chat-view-scroll-view {
      display: flex;
      flex-direction: column;
      flex: 1;
      min-height: 0;
      overflow: hidden;
    }
    ::ng-deep .side-panel copilot-chat-view-scroll-view > div {
      height: 100% !important;
      overflow-y: auto;
      padding: 0 1rem 6rem;
    }
  `]
})
export class AgUiPageComponent implements OnInit, OnDestroy {
  private readonly chartStore = inject(ChartStoreService);
  private readonly mlRegistry = inject(MlOrchestratorRegistryService);
  private readonly agentStore = injectAgentStore('agentic-research');
  private readonly processedMessageIds = new Set<string>();

  constructor() {
    registerFrontendTool({
      name: 'switch_ag_ui_tab',
      description: 'Switch the active workspace tab',
      parameters: z.object({ tab: z.string() }),
      handler: async ({ tab }) => {
        const result = handleSwitchAgUiTab(tab);
        if (result.status === 'ok') this.activeTab.set(result.tab as WorkspaceTab);
        return JSON.stringify(result);
      },
    });

    registerFrontendTool({
      name: 'add_chart_spec',
      description: 'Add a chart to the workspace',
      parameters: z.object({ chartSpec: z.any().optional(), chartSpecs: z.array(z.any()).optional() }),
      handler: async (args) => {
        const result = handleAddChartSpec(
          { chartSpec: args.chartSpec, chartSpecs: args.chartSpecs },
          (spec: any) => this.chartStore.addChartSpec(spec)
        );
        return JSON.stringify(result);
      },
    });

    registerFrontendTool({
      name: 'clear_charts',
      description: 'Clear all charts from the workspace',
      parameters: z.object({}),
      handler: async () => {
        this.chartStore.clearCharts();
        return JSON.stringify({ status: 'ok' });
      },
    });

    registerFrontendTool({
      name: 'set_active_dataset',
      description: 'Switch the active dataset',
      parameters: z.object({ dataset_id: z.string() }),
      handler: async ({ dataset_id }) => {
        if (!dataset_id) return JSON.stringify({ status: 'error', code: 'MISSING_DATASET_ID' });
        const result = handleAgenticSetActiveDataset(
          dataset_id,
          this.datasetManifest(),
          (resolvedId) => this.activeDatasetId.set(resolvedId),
        );
        return JSON.stringify(result);
      },
    });

    registerFrontendTool({
      name: 'set_pytorch_form_fields',
      description: 'Set PyTorch training form fields',
      parameters: z.object({ fields: z.record(z.string(), z.any()) }),
      handler: async ({ fields }) => this.patchMlFormFields('pytorch', fields),
    });

    registerFrontendTool({
      name: 'set_tensorflow_form_fields',
      description: 'Set TensorFlow training form fields',
      parameters: z.object({ fields: z.record(z.string(), z.any()) }),
      handler: async ({ fields }) => this.patchMlFormFields('tensorflow', fields),
    });

    registerFrontendTool({
      name: 'set_active_ml_form_fields',
      description: 'Set form fields on the currently active ML tab',
      parameters: z.object({ fields: z.record(z.string(), z.any()) }),
      handler: async ({ fields }) => {
        const fw = this.activeTab() === 'tensorflow' ? 'tensorflow' : 'pytorch';
        return this.patchMlFormFields(fw, fields);
      },
    });

    registerFrontendTool({
      name: 'start_pytorch_training_runs',
      description: 'Start PyTorch training runs',
      parameters: z.object({}),
      handler: async () => this.startMlTraining('pytorch'),
    });

    registerFrontendTool({
      name: 'start_tensorflow_training_runs',
      description: 'Start TensorFlow training runs',
      parameters: z.object({}),
      handler: async () => this.startMlTraining('tensorflow'),
    });

    registerFrontendTool({
      name: 'start_active_ml_training_runs',
      description: 'Start training runs on the currently active ML tab',
      parameters: z.object({}),
      handler: async () => {
        const fw = this.activeTab() === 'tensorflow' ? 'tensorflow' : 'pytorch';
        return this.startMlTraining(fw);
      },
    });

    connectAgentContext(computed(() => ({
      description: 'ag_ui_active_tab',
      value: this.activeTab(),
    })));

    connectAgentContext(computed(() => ({
      description: 'agentic_research_selected_dataset_id',
      value: this.activeDatasetId() ?? '',
    })));

    connectAgentContext(computed(() => ({
      description: 'ag_ui_selected_model_id',
      value: this.selectedModelId(),
    })));

    effect(() => {
      const store = this.agentStore();
      if (!store) return;
      const messages = store.messages();
      if (!messages?.length) return;
      for (const msg of messages) {
        if (msg.role !== 'assistant') continue;
        const content = (msg as any).content;
        if (!content || typeof content !== 'string') continue;
        const displayContent = extractCopilotDisplayMessage(content);
        if (displayContent !== content) {
          (msg as any).content = displayContent;
        }
        if (this.processedMessageIds.has(msg.id)) continue;
        const payload = parseCopilotAssistantPayload(content);
        if (!payload?.chartSpec) {
          this.processedMessageIds.add(msg.id);
          continue;
        }
        this.processedMessageIds.add(msg.id);
        const specs = normalizeChartSpecInput(payload.chartSpec);
        if (!specs) continue;
        const specArray = Array.isArray(specs) ? specs : [specs];
        for (const spec of specArray) {
          this.chartStore.addChartSpec(spec);
        }
      }
    });
  }
  readonly tabs: { id: WorkspaceTab; label: string }[] = [
    { id: 'charts', label: 'Charts' },
    { id: 'agentic-research', label: 'Agentic Research' },
    { id: 'pytorch', label: 'PyTorch' },
    { id: 'tensorflow', label: 'Tensorflow' },
  ];

  readonly activeTab = signal<WorkspaceTab>('charts');
  readonly showTools = signal(false);
  readonly activeDatasetId = signal<string | null>(null);
  readonly datasetManifest = signal<{ id: string; label: string }[]>([]);
  readonly modelOptions = signal<{ id: string; label: string }[]>(AG_UI_FALLBACK_MODELS);
  readonly selectedModelId = signal<string>(AG_UI_FALLBACK_MODELS[0]?.id ?? '');

  readonly activeTabLabel = computed(() =>
    this.tabs.find(t => t.id === this.activeTab())?.label ?? ''
  );

  ngOnInit(): void {
    fetch('/api/ai/llm/gemini-models')
      .then(res => res.ok ? res.json() : null)
      .then(payload => {
        if (payload?.status === 'ok' && Array.isArray(payload.models)) {
          const models = payload.models.map((m: any) => ({ id: m.id, label: m.label }));
          this.modelOptions.set(models);
          const resolved = resolveNextAgUiSelectedModelId({
            currentSelectedModelId: null,
            fetchedModels: models,
            apiCurrentModelId: payload.currentModel ?? null,
            preferredModelId: AG_UI_PREFERRED_MODEL_ID,
          });
          if (resolved) this.selectedModelId.set(resolved);
        }
      })
      .catch(() => {});

    fetchAgenticDatasetManifest({}, { runtimeDeps: { resolveBaseUrl: () => '/api/ai' } })
      .then(entries => {
        this.datasetManifest.set(entries.map(e => ({ id: e.id, label: e.label })));
      })
      .catch(() => {});
  }

  ngOnDestroy(): void {}

  readonly toolsForTab = computed((): ToolDef[] => {
    const base: ToolDef[] = [
      { name: 'switch_ag_ui_tab', description: 'Switch the active workspace tab' },
      { name: 'navigate_to_page', description: 'Navigate to another page in the app' },
    ];
    switch (this.activeTab()) {
      case 'charts':
        return [...base,
          { name: 'add_chart_spec', description: 'Add a chart to the workspace' },
          { name: 'clear_charts', description: 'Remove all charts from the workspace' },
        ];
      case 'agentic-research':
        return [...base,
          { name: 'add_chart_spec', description: 'Add a chart to the research workspace' },
          { name: 'clear_charts', description: 'Clear all research charts' },
          { name: 'set_active_dataset', description: 'Switch the active dataset' },
        ];
      case 'pytorch':
        return [...base,
          { name: 'set_pytorch_form_fields', description: 'Set PyTorch training form fields' },
          { name: 'train_pytorch_model', description: 'Start a PyTorch training run' },
        ];
      case 'tensorflow':
        return [...base,
          { name: 'set_tensorflow_form_fields', description: 'Set Tensorflow training form fields' },
          { name: 'train_tensorflow_model', description: 'Start a Tensorflow training run' },
        ];
      default:
        return base;
    }
  });

  private patchMlFormFields(framework: string, rawFields: Record<string, any>): string {
    const orch = this.mlRegistry.get(framework);
    if (!orch) return JSON.stringify({ status: 'error', code: 'ORCHESTRATOR_NOT_MOUNTED' });

    const fields = resolveMlFormPatchFromToolArgs({ fields: rawFields });

    const fieldMap: Record<string, (v: any) => void> = {
      training_mode: (v) => orch.trainingMode.set(v),
      task: (v) => orch.task.set(v),
      epoch_values: (v) => orch.epochValues.set(String(v)),
      batch_sizes: (v) => orch.batchSizes.set(String(v)),
      learning_rates: (v) => orch.learningRates.set(String(v)),
      test_sizes: (v) => orch.testSizes.set(String(v)),
      hidden_dims: (v) => orch.hiddenDims.set(String(v)),
      num_hidden_layers: (v) => orch.numHiddenLayers.set(String(v)),
      dropouts: (v) => orch.dropouts.set(String(v)),
      exclude_columns: (v) => orch.excludeColumns.set(String(v)),
      date_columns: (v) => orch.dateColumns.set(String(v)),
      set_sweep_values: (v) => orch.sweepEnabled.set(Boolean(v)),
      auto_distill: (v) => orch.autoDistill.set(Boolean(v)),
      dataset_id: (v) => void orch.onDatasetChange(String(v)),
      target_column: (v) => orch.targetColumn.set(String(v)),
    };

    const applied: string[] = [];
    for (const [key, value] of Object.entries(fields)) {
      const setter = fieldMap[key];
      if (setter) {
        setter(value);
        applied.push(key);
      }
    }
    return JSON.stringify({ status: 'ok', applied });
  }

  private startMlTraining(framework: string): string {
    const orch = this.mlRegistry.get(framework);
    if (!orch) return JSON.stringify({ status: 'error', code: 'ORCHESTRATOR_NOT_MOUNTED' });
    void orch.onTrain();
    return JSON.stringify({ status: 'ok' });
  }
}
