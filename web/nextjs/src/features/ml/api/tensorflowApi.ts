import { getAiApiBaseUrl } from "@/core/config/aiApi";

export type TensorflowTrainRequest = {
  dataset_id: string;
  target_column: string;
  training_mode?: "mlp" | "linear_glm_baseline";
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

export type TensorflowTrainSuccess = {
  status: "ok";
  model_id?: string;
  model_path?: string;
  metrics?: unknown;
};

export type TensorflowTrainError = {
  status: "error";
  code: string;
  error: string;
};

export type TensorflowDistillRequest = {
  dataset_id: string;
  target_column: string;
  training_mode?: "mlp" | "linear_glm_baseline";
  save_model?: boolean;
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

export async function trainTensorflowModel(
  payload: TensorflowTrainRequest
): Promise<TensorflowTrainSuccess | TensorflowTrainError> {
  try {
    const response = await fetch(`${AI_API_BASE_URL}/ml/tensorflow/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_id: payload.dataset_id,
        target_column: payload.target_column,
        training_mode: payload.training_mode ?? "mlp",
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
      model_id?: string;
      model_path?: string;
      metrics?: unknown;
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
      model_id: data.model_id,
      model_path: data.model_path,
      metrics: data.metrics,
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
  payload: TensorflowDistillRequest
): Promise<TensorflowTrainSuccess | TensorflowTrainError> {
  try {
    const response = await fetch(`${AI_API_BASE_URL}/ml/tensorflow/distill`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_id: payload.dataset_id,
        target_column: payload.target_column,
        training_mode: payload.training_mode ?? "mlp",
        save_model: payload.save_model ?? false,
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

    const data = (await response.json()) as {
      status?: string;
      model_id?: string;
      model_path?: string;
      metrics?: unknown;
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
      model_id: data.model_id,
      model_path: data.model_path,
      metrics: data.metrics,
    };
  } catch (error) {
    return {
      status: "error",
      code: "TENSORFLOW_DISTILL_REQUEST_FAILED",
      error:
        error instanceof Error
          ? error.message
          : "Failed to send TensorFlow distillation request.",
    };
  }
}
