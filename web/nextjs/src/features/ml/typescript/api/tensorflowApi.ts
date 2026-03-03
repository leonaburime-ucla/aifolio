import { getAiApiBaseUrl } from "@/core/config/aiApi";
import type {
  TensorflowApiRuntime,
  TensorflowDistillRequest,
  TensorflowTrainError,
  TensorflowTrainRequest,
  TensorflowTrainSuccess,
} from "@/features/ml/__types__/typescript/api/tensorflowApi.types";
export type {
  TensorflowDistillRequest,
  TensorflowTrainError,
  TensorflowTrainRequest,
  TensorflowTrainSuccess,
  TensorflowTrainingMode,
  TensorflowApiRuntime,
} from "@/features/ml/__types__/typescript/api/tensorflowApi.types";

const DISTILL_TIMEOUT_MS = 180_000;
const DISTILL_TIMEOUT_SECONDS = Math.round(DISTILL_TIMEOUT_MS / 1000);

export async function trainTensorflowModel(
  payload: TensorflowTrainRequest,
  {
    fetchImpl = fetch,
    resolveBaseUrl = getAiApiBaseUrl,
  }: Partial<TensorflowApiRuntime> = {}
): Promise<TensorflowTrainSuccess | TensorflowTrainError> {
  try {
    const response = await fetchImpl(`${resolveBaseUrl()}/ml/tensorflow/train`, {
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
        code: "TENSORFLOW_TRAIN_FAILED",
        error: data.error ?? "Failed to train TensorFlow model.",
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
      code: "TENSORFLOW_TRAIN_REQUEST_FAILED",
      error:
        error instanceof Error
          ? error.message
          : "Failed to send TensorFlow training request.",
    };
  }
}

export async function distillTensorflowModel(
  payload: TensorflowDistillRequest,
  {
    fetchImpl = fetch,
    resolveBaseUrl = getAiApiBaseUrl,
    scheduleTimeout = setTimeout,
    clearScheduledTimeout = clearTimeout,
  }: Partial<TensorflowApiRuntime> = {}
): Promise<TensorflowTrainSuccess | TensorflowTrainError> {
  const controller = new AbortController();
  const timeout = scheduleTimeout(() => controller.abort(), DISTILL_TIMEOUT_MS);
  try {
    const response = await fetchImpl(`${resolveBaseUrl()}/ml/tensorflow/distill`, {
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
    clearScheduledTimeout(timeout);

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
        code: "TENSORFLOW_DISTILL_FAILED",
        error: data.error ?? "Failed to distill TensorFlow model.",
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
    clearScheduledTimeout(timeout);
    return {
      status: "error",
      code: "TENSORFLOW_DISTILL_REQUEST_FAILED",
      error:
        error instanceof Error
          ? error.name === "AbortError"
            ? `Distillation timed out after ${DISTILL_TIMEOUT_SECONDS} seconds.`
            : error.message
          : "Failed to send TensorFlow distillation request.",
    };
  }
}
