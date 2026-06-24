import { Injectable, signal } from '@angular/core';
import type { TrainingScreenOrchestrator } from '../../features/ml/orchestrator/training-screen-orchestrator.service';

@Injectable({ providedIn: 'root' })
export class MlOrchestratorRegistryService {
  private readonly instances = signal<Map<string, TrainingScreenOrchestrator>>(new Map());

  register(framework: string, orchestrator: TrainingScreenOrchestrator): void {
    this.instances.update(map => {
      const next = new Map(map);
      next.set(framework, orchestrator);
      return next;
    });
  }

  unregister(framework: string): void {
    this.instances.update(map => {
      const next = new Map(map);
      next.delete(framework);
      return next;
    });
  }

  get(framework: string): TrainingScreenOrchestrator | undefined {
    return this.instances().get(framework);
  }

  getActive(activeTab: string): TrainingScreenOrchestrator | undefined {
    if (activeTab === 'pytorch') return this.get('pytorch');
    if (activeTab === 'tensorflow') return this.get('tensorflow');
    return undefined;
  }
}
