export type TrainPayload = {
  dataset_id: string;
  target_column: string;
  training_mode: string;
  task: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  test_size: number;
  hidden_dim: number;
  num_hidden_layers: number;
  dropout: number;
  exclude_columns: string[];
  date_columns: string[];
};

export type TrainResponse = {
  status: string;
  run_id?: string;
  model_id?: string;
  model_path?: string;
  metrics?: {
    train_loss?: number;
    test_loss?: number;
    test_metric_name?: string;
    test_metric_value?: number;
    test_metric?: number;
    accuracy?: number;
  };
  error?: string;
};

export type DistillPayload = {
  dataset_id: string;
  target_column: string;
  training_mode: string;
  save_model: boolean;
  teacher_run_id?: string;
  teacher_model_id?: string;
  teacher_model_path?: string;
  exclude_columns: string[];
  date_columns: string[];
  task: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  test_size: number;
  temperature: number;
  alpha: number;
  student_hidden_dim: number;
  student_num_hidden_layers: number;
  student_dropout: number;
};

export type DistillResponse = {
  status: string;
  run_id?: string;
  model_id?: string;
  model_path?: string;
  metrics?: any;
  teacher_input_dim?: number | null;
  teacher_output_dim?: number | null;
  student_input_dim?: number | null;
  student_output_dim?: number | null;
  teacher_model_size_bytes?: number | null;
  student_model_size_bytes?: number | null;
  size_saved_bytes?: number | null;
  size_saved_percent?: number | null;
  teacher_param_count?: number | null;
  student_param_count?: number | null;
  param_saved_count?: number | null;
  param_saved_percent?: number | null;
  error?: string;
};

export type MlTrainingApi = {
  trainPytorch: (payload: TrainPayload) => Promise<TrainResponse>;
  trainTensorflow: (payload: TrainPayload) => Promise<TrainResponse>;
  distillPytorch: (payload: DistillPayload) => Promise<DistillResponse>;
  distillTensorflow: (payload: DistillPayload) => Promise<DistillResponse>;
};

export function createMlTrainingApi({ baseUrl }: { baseUrl: string }): MlTrainingApi {
  async function trainPytorch(payload: TrainPayload): Promise<TrainResponse> {
    const res = await fetch(`${baseUrl}/ml/pytorch/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return res.json();
  }

  async function trainTensorflow(payload: TrainPayload): Promise<TrainResponse> {
    const res = await fetch(`${baseUrl}/ml/tensorflow/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return res.json();
  }

  async function distillPytorch(payload: DistillPayload): Promise<DistillResponse> {
    const res = await fetch(`${baseUrl}/ml/pytorch/distill`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return res.json();
  }

  async function distillTensorflow(payload: DistillPayload): Promise<DistillResponse> {
    const res = await fetch(`${baseUrl}/ml/tensorflow/distill`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return res.json();
  }

  return { trainPytorch, trainTensorflow, distillPytorch, distillTensorflow };
}
