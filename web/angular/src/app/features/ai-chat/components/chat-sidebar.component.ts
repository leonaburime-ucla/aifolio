import { CommonModule } from '@angular/common';
import { AfterViewChecked, Component, ElementRef, Input, OnChanges, OnInit, SimpleChanges, ViewChild, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ChatSidebarOrchestrator } from '../orchestrator/chat-sidebar-orchestrator.service';

@Component({
  selector: 'app-chat-sidebar',
  imports: [CommonModule, FormsModule],
  providers: [ChatSidebarOrchestrator],
  template: `
    <aside class="chat-sidebar">
      <div class="chat-header">
        <p>AI Chat</p>
        @if (chat.modelOptions().length > 0) {
          <select
            aria-label="Select AI model"
            [ngModel]="chat.selectedModelId()"
            (ngModelChange)="chat.selectedModelId.set($event)"
          >
            @for (model of chat.modelOptions(); track model.id) {
              <option [value]="model.id">{{ model.label }}</option>
            }
          </select>
        }
      </div>

      <div #messagesEl class="chat-messages">
        @if (chat.messages().length === 0) {
          <div class="muted" style="font-size: .75rem;">Ask a question to get started.</div>
        }
        @for (message of chat.messages(); track message.id) {
          <p class="chat-message" [class.user]="message.role === 'user'" [class.assistant]="message.role === 'assistant'">
            {{ message.content }}
          </p>
        }
        @if (chat.isSending()) {
          <p class="chat-message assistant"><span class="thinking-spinner"></span> Thinking...</p>
        }
      </div>

      @if (chat.screenFeedback(); as feedback) {
        <div class="chat-feedback">{{ feedback.message }}</div>
      }

      <form class="chat-form" (submit)="$event.preventDefault(); chat.submit()">
        <div class="chat-form-row">
          <button type="button" class="btn secondary" disabled title="Disabled for now">+</button>
          <input
            class="chat-input"
            type="text"
            placeholder="Ask anything"
            aria-label="Chat input"
            [ngModel]="chat.inputValue()"
            name="chatInput"
            (ngModelChange)="chat.inputValue.set($event)"
            (keydown.arrowUp)="$event.preventDefault(); chat.handleHistory('up')"
            (keydown.arrowDown)="$event.preventDefault(); chat.handleHistory('down')"
          />
          <button type="submit" class="btn primary" [disabled]="!chat.hasInput() || chat.isSending()">Send</button>
        </div>
      </form>
    </aside>
  `
})
export class ChatSidebarComponent implements OnInit, OnChanges, AfterViewChecked {
  @Input() mode: 'direct' | 'research' = 'direct';
  @Input() datasetId: string | null = null;
  @ViewChild('messagesEl') messagesEl?: ElementRef<HTMLElement>;

  readonly chat = inject(ChatSidebarOrchestrator);

  ngOnInit(): void {
    this.syncConfig();
    void this.chat.loadModels();
  }

  ngOnChanges(_changes: SimpleChanges): void {
    this.syncConfig();
  }

  ngAfterViewChecked(): void {
    const el = this.messagesEl?.nativeElement;
    if (el) el.scrollTop = el.scrollHeight;
  }

  private syncConfig(): void {
    this.chat.configure({ mode: this.mode, datasetId: this.datasetId, baseUrl: '/api/ai' });
  }
}
