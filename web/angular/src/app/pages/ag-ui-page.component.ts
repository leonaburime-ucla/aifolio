// import { CommonModule } from '@angular/common';
import { Component /*, computed, effect, inject, OnDestroy, OnInit, signal */ } from '@angular/core';
// import { FormsModule } from '@angular/forms';
// import { ChartsWorkspaceComponent } from '../features/recharts/components/charts-workspace.component';
// import { AgenticResearchWorkspaceComponent } from '../features/agentic-research/components/agentic-research-workspace.component';
// import { TrainingScreenComponent } from '../features/ml/components/training-screen.component';
// import { ChartStoreService } from '../shared/state/chart-store.service';
// import { MlOrchestratorRegistryService } from '../shared/state/ml-orchestrator-registry.service';
// import { CopilotChat, registerFrontendTool, connectAgentContext, injectAgentStore } from '@copilotkit/angular';
// import { handleSwitchAgUiTab, handleAddChartSpec, AG_UI_FALLBACK_MODELS, AG_UI_PREFERRED_MODEL_ID, resolveNextAgUiSelectedModelId, parseCopilotAssistantPayload, normalizeChartSpecInput, extractCopilotDisplayMessage, resolveMlFormPatchFromToolArgs } from '@aifolio/frontend-core/ag-ui';
// import { handleAgenticSetActiveDataset, fetchAgenticDatasetManifest } from '@aifolio/frontend-core/agentic-research';
// import { z } from 'zod';

// type WorkspaceTab = 'charts' | 'agentic-research' | 'pytorch' | 'tensorflow';
//
// interface ToolDef {
//   name: string;
//   description: string;
// }

@Component({
  selector: 'app-ag-ui-page',
  imports: [],
  template: `
    <div class="ag-ui-construction">
      <div class="ag-ui-construction-card">
        <div class="icon">🚧</div>
        <h1>Under Construction</h1>
        <p>The AG-UI workspace is being rebuilt. Check back soon.</p>
        <div class="links">
          <a href="https://github.com/ag-ui-protocol/ag-ui" target="_blank" rel="noreferrer">AG-UI Protocol</a>
          <span>|</span>
          <a href="https://github.com/CopilotKit/CopilotKit" target="_blank" rel="noreferrer">CopilotKit</a>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .ag-ui-construction {
      display: flex;
      align-items: center;
      justify-content: center;
      height: calc(100dvh - 64px);
      background: #fafafa;
      padding: 2rem;
    }
    .ag-ui-construction-card {
      text-align: center;
      max-width: 28rem;
      padding: 3rem 2rem;
      border-radius: 1rem;
      border: 1px solid #e4e4e7;
      background: #fff;
      box-shadow: 0 4px 6px -1px rgba(0,0,0,.05);
    }
    .icon { font-size: 3rem; margin-bottom: 1rem; }
    h1 { font-size: 1.5rem; font-weight: 700; color: #18181b; margin: 0 0 .75rem; }
    p { font-size: .875rem; color: #52525b; margin: 0 0 1.5rem; }
    .links { font-size: .75rem; color: #71717a; }
    .links a { color: #3b82f6; text-decoration: underline; text-underline-offset: 2px; }
    .links span { margin: 0 .5rem; }
  `]
})
export class AgUiPageComponent {}

/* --- ORIGINAL CLASS BODY (commented out for Under Construction) ---
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
--- END ORIGINAL CLASS BODY */
