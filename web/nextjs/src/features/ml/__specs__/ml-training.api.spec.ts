/**
 * Spec: ml-training.api.spec.ts
 * Version: 1.1.0
 */
export const ML_TRAINING_API_SPEC_VERSION = "1.1.0";

export const mlTrainingApiSpec = {
  id: "ml-training.api",
  version: ML_TRAINING_API_SPEC_VERSION,
  status: "draft",
  lastUpdated: "2026-02-23",
  contracts: {
    fetchMlDatasetOptions: {
      endpoint: "GET /ml-data",
      response: "MlDatasetOption[]",
      onFailure: "throw Error('Failed to load ML datasets.')",
    },
    fetchMlDatasetRows: {
      endpoint: "GET /ml-data/:id",
      response: "{columns?:string[],rows?:Array<Record<string,string|number|null>>,rowCount?:number,totalRowCount?:number}",
      onFailure: "throw Error('Failed to load dataset rows.')",
    },
    trainPytorchModel: {
      endpoint: "POST /ml/pytorch/train",
      requestDefaults: { save_model: false, task: "auto" },
      result: "{status:'ok'| 'error', code?:string, error?:string, model_id?:string, model_path?:string, metrics?:unknown}",
    },
    distillPytorchModel: {
      endpoint: "POST /ml/pytorch/distill",
      requestDefaults: { save_model: false, task: "auto" },
      result: "{status:'ok'| 'error', code?:string, error?:string, model_id?:string, model_path?:string, metrics?:unknown}",
    },
    trainTensorflowModel: {
      endpoint: "POST /ml/tensorflow/train",
      requestDefaults: { save_model: false, training_mode: "mlp", task: "auto" },
      result: "{status:'ok'| 'error', code?:string, error?:string, model_id?:string, model_path?:string, metrics?:unknown}",
    },
    distillTensorflowModel: {
      endpoint: "POST /ml/tensorflow/distill",
      requestDefaults: { save_model: false, training_mode: "mlp", task: "auto" },
      result: "{status:'ok'| 'error', code?:string, error?:string, model_id?:string, model_path?:string, metrics?:unknown}",
    },
  },
  deterministicRules: [
    "API wrappers never throw transport/business failures for train/distill; they return status='error' with stable code values.",
    "Dataset API wrappers throw user-safe errors on non-OK responses.",
  ],
} as const;
