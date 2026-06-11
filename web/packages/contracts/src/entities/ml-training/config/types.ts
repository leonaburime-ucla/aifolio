export type MlTaskType = "classification" | "regression" | "auto";

export type DatasetTrainingDefaults = {
  targetColumn: string;
  excludeColumns: string[];
  dateColumns: string[];
  task: MlTaskType;
  epochs: number;
};
