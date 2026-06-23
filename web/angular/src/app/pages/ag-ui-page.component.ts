import { CommonModule } from '@angular/common';
import { Component, computed, signal } from '@angular/core';
import { ChatSidebarComponent } from '../features/ai-chat/components/chat-sidebar.component';
import { ChartsWorkspaceComponent } from '../features/recharts/components/charts-workspace.component';
import { AgenticResearchWorkspaceComponent } from '../features/agentic-research/components/agentic-research-workspace.component';
import { TrainingScreenComponent } from '../features/ml/components/training-screen.component';

type WorkspaceTab = 'charts' | 'agentic-research' | 'pytorch' | 'tensorflow';

interface ToolDef {
  name: string;
  description: string;
}

@Component({
  selector: 'app-ag-ui-page',
  imports: [
    CommonModule,
    ChatSidebarComponent,
    ChartsWorkspaceComponent,
    AgenticResearchWorkspaceComponent,
    TrainingScreenComponent,
  ],
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
              <app-agentic-research-workspace (datasetChange)="activeDatasetId.set($event)" />
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
        <app-chat-sidebar [mode]="chatMode()" [datasetId]="activeDatasetId()" />
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
      width: 420px;
      flex-shrink: 0;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      height: 100%;
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
  `]
})
export class AgUiPageComponent {
  readonly tabs: { id: WorkspaceTab; label: string }[] = [
    { id: 'charts', label: 'Charts' },
    { id: 'agentic-research', label: 'Agentic Research' },
    { id: 'pytorch', label: 'PyTorch' },
    { id: 'tensorflow', label: 'Tensorflow' },
  ];

  readonly activeTab = signal<WorkspaceTab>('charts');
  readonly showTools = signal(false);
  readonly activeDatasetId = signal<string | null>(null);

  readonly activeTabLabel = computed(() =>
    this.tabs.find(t => t.id === this.activeTab())?.label ?? ''
  );

  readonly chatMode = computed(() =>
    this.activeTab() === 'agentic-research' ? 'research' as const : 'direct' as const
  );

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
}
