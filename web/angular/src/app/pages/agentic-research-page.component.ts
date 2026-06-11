import { CommonModule } from '@angular/common';
import { Component, signal } from '@angular/core';
import { ChatSidebarComponent } from '../features/ai-chat/components/chat-sidebar.component';
import { AgenticResearchWorkspaceComponent } from '../features/agentic-research/components/agentic-research-workspace.component';

@Component({
  selector: 'app-agentic-research-page',
  imports: [CommonModule, AgenticResearchWorkspaceComponent, ChatSidebarComponent],
  template: `
    <div class="page-shell">
      <main class="page-main">
        <div class="page-content">
          <p class="page-kicker">Agentic Research</p>
          <app-agentic-research-workspace (datasetChange)="activeDatasetId.set($event)" />
        </div>
      </main>
      <div class="side-panel">
        <app-chat-sidebar mode="research" [datasetId]="activeDatasetId()" />
      </div>
    </div>
  `
})
export class AgenticResearchPageComponent {
  readonly activeDatasetId = signal<string | null>(null);
}
