import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/features/ml/react/views/screens/PytorchTrainingScreen", () => ({
  default: () => <div data-testid="pytorch-training-screen" />,
}));

vi.mock("@/features/ml/react/views/screens/TensorflowTrainingScreen", () => ({
  default: () => <div data-testid="tensorflow-training-screen" />,
}));

import PytorchAgUiSurface from "@/features/ml/react/views/screens/PytorchAgUiSurface";
import TensorflowAgUiSurface from "@/features/ml/react/views/screens/TensorflowAgUiSurface";

describe("ML AG-UI surfaces", () => {
  afterEach(() => cleanup());

  it("renders PyTorch sample prompts around the training screen", () => {
    render(<PytorchAgUiSurface />);

    expect(screen.getByText("Show Sample Prompts")).toBeInTheDocument();
    expect(screen.getByText(/Use the fraud detection dataset/i)).toBeInTheDocument();
    expect(screen.getByTestId("pytorch-training-screen")).toBeInTheDocument();
  });

  it("renders TensorFlow sample prompts around the training screen", () => {
    render(<TensorflowAgUiSurface />);

    expect(screen.getByText("Show Sample Prompts")).toBeInTheDocument();
    expect(screen.getByText(/Use the house prices dataset/i)).toBeInTheDocument();
    expect(screen.getByTestId("tensorflow-training-screen")).toBeInTheDocument();
  });
});
