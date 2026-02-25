import { getAiApiBaseUrl } from "@/core/config/aiApi";

export type PytorchTrainRequest = {
  dataset_id: string;
  target_column: string;
  training_mode?:
    | "mlp_dense"
    | "linear_glm_baseline"
    | "tabresnet"
    | "imbalance_aware"
    | "calibrated_classifier"
    | "tree_teacher_distillation";
  save_model?: boolean;
  exclude_columns?: string[];
  date_columns?: string[];
  task?: "classification" | "regression" | "auto";
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  test_size?: number;
  hidden_dim?: number;
  num_hidden_layers?: number;
  dropout?: number;
};

export type PytorchTrainSuccess = {
  status: "ok";
  run_id?: string;
  model_id?: string;
  model_path?: string;
  metrics?: unknown;
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
};

export type PytorchTrainError = {
  status: "error";
  code: string;
  error: string;
};

export type PytorchDistillRequest = {
  dataset_id: string;
  target_column: string;
  training_mode?:
    | "mlp_dense"
    | "linear_glm_baseline"
    | "tabresnet"
    | "imbalance_aware"
    | "calibrated_classifier"
    | "tree_teacher_distillation";
  save_model?: boolean;
  teacher_run_id?: string;
  teacher_model_id?: string;
  teacher_model_path?: string;
  exclude_columns?: string[];
  date_columns?: string[];
  task?: "classification" | "regression" | "auto";
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  test_size?: number;
  temperature?: number;
  alpha?: number;
  student_hidden_dim?: number;
  student_num_hidden_layers?: number;
  student_dropout?: number;
};

const AI_API_BASE_URL = getAiApiBaseUrl();
const DISTILL_TIMEOUT_MS = 60_000;

export async function trainPytorchModel(
  payload: PytorchTrainRequest
): Promise<PytorchTrainSuccess | PytorchTrainError> {
  try {
    const response = await fetch(`${AI_API_BASE_URL}/ml/pytorch/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_id: payload.dataset_id,
        target_column: payload.target_column,
        training_mode: payload.training_mode ?? "mlp_dense",
        save_model: payload.save_model ?? false,
        exclude_columns: payload.exclude_columns,
        date_columns: payload.date_columns,
        task: payload.task ?? "auto",
        epochs: payload.epochs,
        batch_size: payload.batch_size,
        learning_rate: payload.learning_rate,
        test_size: payload.test_size,
        hidden_dim: payload.hidden_dim,
        num_hidden_layers: payload.num_hidden_layers,
        dropout: payload.dropout,
      }),
    });

    const data = (await response.json()) as {
      status?: string;
      run_id?: string;
      model_id?: string;
      model_path?: string;
      metrics?: unknown;
      teacher_input_dim?: number | null;
      teacher_output_dim?: number | null;
      student_input_dim?: number | null;
      student_output_dim?: number | null;
      teacher_model_size_bytes?: number | null;
      student_model_size_bytes?: number | null;
      size_saved_bytes?: number | null;
      size_saved_percent?: number | null;
      error?: string;
    };

    if (!response.ok || data.status !== "ok") {
      return {
        status: "error",
        code: "PYTORCH_TRAIN_FAILED",
        error: data.error ?? "Failed to train PyTorch model.",
      };
    }

    return {
      status: "ok",
      run_id: data.run_id,
      model_id: data.model_id,
      model_path: data.model_path,
      metrics: data.metrics,
      teacher_model_size_bytes: data.teacher_model_size_bytes ?? null,
      student_model_size_bytes: data.student_model_size_bytes ?? null,
      size_saved_bytes: data.size_saved_bytes ?? null,
      size_saved_percent: data.size_saved_percent ?? null,
    };
  } catch (error) {
    return {
      status: "error",
      code: "PYTORCH_TRAIN_REQUEST_FAILED",
      error:
        error instanceof Error
          ? error.message
          : "Failed to send PyTorch training request.",
    };
  }
}

export async function distillPytorchModel(
  payload: PytorchDistillRequest
): Promise<PytorchTrainSuccess | PytorchTrainError> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), DISTILL_TIMEOUT_MS);
  try {
    const response = await fetch(`${AI_API_BASE_URL}/ml/pytorch/distill`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        dataset_id: payload.dataset_id,
        target_column: payload.target_column,
        training_mode: payload.training_mode ?? "mlp_dense",
        save_model: payload.save_model ?? false,
        teacher_run_id: payload.teacher_run_id,
        teacher_model_id: payload.teacher_model_id,
        teacher_model_path: payload.teacher_model_path,
        exclude_columns: payload.exclude_columns,
        date_columns: payload.date_columns,
        task: payload.task ?? "auto",
        epochs: payload.epochs,
        batch_size: payload.batch_size,
        learning_rate: payload.learning_rate,
        test_size: payload.test_size,
        temperature: payload.temperature,
        alpha: payload.alpha,
        student_hidden_dim: payload.student_hidden_dim,
        student_num_hidden_layers: payload.student_num_hidden_layers,
        student_dropout: payload.student_dropout,
      }),
    });
    clearTimeout(timeout);

    const data = (await response.json()) as {
      status?: string;
      run_id?: string;
      model_id?: string;
      model_path?: string;
      metrics?: unknown;
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

    if (!response.ok || data.status !== "ok") {
      return {
        status: "error",
        code: "PYTORCH_DISTILL_FAILED",
        error: data.error ?? "Failed to distill PyTorch model.",
      };
    }

    return {
      status: "ok",
      run_id: data.run_id,
      model_id: data.model_id,
      model_path: data.model_path,
      metrics: data.metrics,
      teacher_input_dim: data.teacher_input_dim ?? null,
      teacher_output_dim: data.teacher_output_dim ?? null,
      student_input_dim: data.student_input_dim ?? null,
      student_output_dim: data.student_output_dim ?? null,
      teacher_model_size_bytes: data.teacher_model_size_bytes ?? null,
      student_model_size_bytes: data.student_model_size_bytes ?? null,
      size_saved_bytes: data.size_saved_bytes ?? null,
      size_saved_percent: data.size_saved_percent ?? null,
      teacher_param_count: data.teacher_param_count ?? null,
      student_param_count: data.student_param_count ?? null,
      param_saved_count: data.param_saved_count ?? null,
      param_saved_percent: data.param_saved_percent ?? null,
    };
  } catch (error) {
    clearTimeout(timeout);
    return {
      status: "error",
      code: "PYTORCH_DISTILL_REQUEST_FAILED",
      error:
        error instanceof Error
          ? error.name === "AbortError"
            ? "Distillation timed out after 60 seconds."
            : error.message
          : "Failed to send PyTorch distillation request.",
    };
  }
}
