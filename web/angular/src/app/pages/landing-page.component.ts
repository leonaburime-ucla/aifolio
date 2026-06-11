import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { ChatSidebarComponent } from '../features/ai-chat/components/chat-sidebar.component';
import { ChartsWorkspaceComponent } from '../features/recharts/components/charts-workspace.component';

@Component({
  selector: 'app-landing-page',
  imports: [CommonModule, ChartsWorkspaceComponent, ChatSidebarComponent],
  template: `
    <div class="page-shell">
      <main class="page-main">
        <div class="page-content">
          <p class="page-kicker">AI-driven Chart Dashboard</p>
          <app-charts-workspace />
        </div>
      </main>
      <div class="side-panel">
        <app-chat-sidebar />
      </div>
    </div>
  `
})
export class LandingPageComponent {}
