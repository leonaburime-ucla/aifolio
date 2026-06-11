import { Injectable } from '@angular/core';
import type { DistillPayload, DistillResponse, TrainPayload, TrainResponse } from '../model/ml-training.types';

@Injectable({ providedIn: 'root' })
export class MlTrainingApiService {
  async fetchManifest(baseUrl: string): Promise<{ datasets: { id: string; label?: string }[] }> {
    const res = await fetch(`${baseUrl}/ml-data`);
    if (!res.ok) throw new Error('Failed to load datasets.');
    const payload = (await res.json()) as { datasets?: { id: string; label?: string }[] };
    return { datasets: payload.datasets ?? [] };
  }

  async fetchDataset(baseUrl: string, id: string): Promise<{ rows: Record<string, unknown>[]; columns?: string[] }> {
    const res = await fetch(`${baseUrl}/ml-data/${encodeURIComponent(id)}`);
    if (!res.ok) throw new Error('Failed to load dataset.');
    return (await res.json()) as { rows: Record<string, unknown>[]; columns?: string[] };
  }

  trainPytorch(baseUrl: string, payload: TrainPayload): Promise<TrainResponse> {
    return this.post(`${baseUrl}/ml/pytorch/train`, payload);
  }

  trainTensorflow(baseUrl: string, payload: TrainPayload): Promise<TrainResponse> {
    return this.post(`${baseUrl}/ml/tensorflow/train`, payload);
  }

  distillPytorch(baseUrl: string, payload: DistillPayload): Promise<DistillResponse> {
    return this.post(`${baseUrl}/ml/pytorch/distill`, payload);
  }

  distillTensorflow(baseUrl: string, payload: DistillPayload): Promise<DistillResponse> {
    return this.post(`${baseUrl}/ml/tensorflow/distill`, payload);
  }

  private async post<T>(url: string, payload: unknown): Promise<T> {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return (await res.json()) as T;
  }
}
