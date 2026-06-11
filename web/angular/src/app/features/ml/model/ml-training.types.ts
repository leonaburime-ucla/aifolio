import type { TrainingRunRow } from '@aifolio/contracts/entities/ml-training';

export type Framework = 'pytorch' | 'tensorflow';

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
  metrics?: Record<string, unknown>;
  error?: string;
};

export type DistillPayload = TrainPayload & {
  save_model: boolean;
  teacher_run_id?: string;
  teacher_model_id?: string;
  teacher_model_path?: string;
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
  metrics?: Record<string, unknown>;
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

export type TrainingCombo = {
  epochs: number;
  batch_size: number;
  learning_rate: number;
  test_size: number;
  hidden_dim: number;
  num_hidden_layers: number;
  dropout: number;
};

export type TrainingRow = TrainingRunRow;
