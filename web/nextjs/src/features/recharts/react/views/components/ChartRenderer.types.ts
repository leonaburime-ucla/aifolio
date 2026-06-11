import type { ChartSpec } from "@aifolio/contracts/entities/chart";
import type { Props as RechartsLabelProps } from "recharts/types/component/Label";

export type ChartRendererProps = {
  spec: ChartSpec;
  onRemove?: (id: string) => void;
};

export type LoadingLabelProps = RechartsLabelProps;
