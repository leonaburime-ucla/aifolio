import { Modal } from "@/core/views/components/General/Modal";
import { Background, Controls, ReactFlow } from "@xyflow/react";
import type { ModelPreviewModalProps } from "@/features/ml-model-ui/__types__/typescript/react/views/modelPreviewModal.types";
import { useModelPreviewModel } from "@/features/ml-model-ui/typescript/react/hooks/useModelPreviewModel.hooks";
import { parseLayerBullet } from "@/features/ml-model-ui/typescript/utils/modelPreview.util";

export function ModelPreviewModal({
  isOpen,
  onClose,
  framework,
  mode,
}: ModelPreviewModalProps) {
  const { graph, data } = useModelPreviewModel({ framework, mode });

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      position="top"
      title={`${graph.title} (${framework === "pytorch" ? "PyTorch" : "TensorFlow"})`}
    >
      <div className="space-y-4 p-1 pb-8">
        <p className="text-sm text-zinc-600">{graph.summary}</p>

        <div className="h-[200px] rounded-md border border-zinc-200 bg-zinc-50/50">
          <ReactFlow
            nodes={graph.nodes}
            edges={graph.edges}
            fitView
            fitViewOptions={{ padding: 0.2 }}
            zoomOnScroll={false}
          >
            <Controls />
            <Background gap={16} size={1} />
          </ReactFlow>
        </div>

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <div className="rounded-md border border-zinc-200 bg-white p-4">
            <p className="mb-2 text-sm font-semibold text-zinc-800">
              Layer Architecture Breakdown
            </p>
            <ul className="space-y-3 text-xs text-zinc-600">
              {data.layers.map((bullet) => {
                const { term, definition } = parseLayerBullet(bullet);
                return (
                  <li key={bullet} className="relative flex flex-col gap-0.5 pl-4">
                    <span className="absolute left-0 top-1.5 h-1.5 w-1.5 rounded-full bg-zinc-400" />
                    <span className="font-semibold text-zinc-800">{term}</span>
                    <span className="leading-relaxed">{definition}</span>
                  </li>
                );
              })}
            </ul>
          </div>

          {data.terminology.length > 0 ? (
            <div className="rounded-md border border-blue-100 bg-blue-50/50 p-4">
              <p className="mb-2 text-sm font-semibold text-blue-900">
                Key Terminology
              </p>
              <ul className="space-y-3 text-xs text-blue-800/80">
                {data.terminology.map(({ term, definition }) => (
                  <li key={term} className="relative flex flex-col gap-0.5 pl-4">
                    <span className="absolute left-0 top-1.5 h-1.5 w-1.5 rounded-full bg-blue-400" />
                    <span className="font-semibold text-blue-900">{term}</span>
                    <span className="leading-relaxed">{definition}</span>
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <div className="flex items-center justify-center rounded-md border border-zinc-200 bg-zinc-50 p-4">
              <p className="text-xs text-zinc-400">No complex terminology explicitly defined for this baseline model.</p>
            </div>
          )}
        </div>
        <div className="h-[30px] shrink-0" />
      </div>
    </Modal>
  );
}
