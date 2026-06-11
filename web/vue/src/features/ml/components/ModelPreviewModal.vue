<template>
  <Modal
    :is-open="isOpen"
    @close="$emit('close')"
    position="top"
    :title="`${title} (${framework === 'pytorch' ? 'PyTorch' : 'TensorFlow'})`"
  >
    <div class="space-y-4 p-1 pb-8 text-sm">
      <p class="text-zinc-600 leading-relaxed">{{ summary }}</p>

      <!-- Flowchart Visualization -->
      <div class="rounded-xl border border-zinc-200 bg-zinc-50/50 p-6 flex flex-col items-center justify-center min-h-[140px]">
        <!-- Sequential Models -->
        <div v-if="flowType === 'sequential'" class="flex flex-wrap items-center justify-center gap-2">
          <div v-for="(node, idx) in nodes" :key="node.label" class="flex items-center">
            <div
              class="rounded-lg border px-3 py-2 text-xs font-medium shadow-sm transition"
              :style="{ background: node.bg, borderColor: node.border, color: '#18181b' }"
            >
              {{ node.label }}
            </div>
            <svg
              v-if="idx < nodes.length - 1"
              class="mx-2 h-4 w-4 text-zinc-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              stroke-width="2"
            >
              <path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7" />
            </svg>
          </div>
        </div>

        <!-- Wide and Deep / Split Branch Models -->
        <div v-else-if="flowType === 'wide_deep'" class="flex items-center justify-center gap-4">
          <div class="rounded-lg border border-zinc-200 bg-zinc-100 px-3 py-2 text-xs font-medium">Input Features</div>
          <svg class="h-4 w-4 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7" />
          </svg>
          <div class="flex flex-col gap-3">
            <div class="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs font-medium">Wide Branch (Linear)</div>
            <div class="rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-xs font-medium">Deep Dense Blocks (1 & 2)</div>
          </div>
          <svg class="h-4 w-4 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7" />
          </svg>
          <div class="rounded-lg border border-green-200 bg-green-50 px-3 py-2 text-xs font-medium">Merge + Prediction Output</div>
        </div>

        <!-- Autoencoder Model -->
        <div v-else-if="flowType === 'autoencoder'" class="flex items-center justify-center gap-4">
          <div class="rounded-lg border border-zinc-200 bg-zinc-100 px-3 py-2 text-xs font-medium">Input Features</div>
          <svg class="h-4 w-4 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7" />
          </svg>
          <div class="rounded-lg border border-indigo-200 bg-indigo-50 px-3 py-2 text-xs font-medium">Encoder</div>
          <svg class="h-4 w-4 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7" />
          </svg>
          <div class="rounded-lg border border-purple-200 bg-purple-50 px-3 py-2 text-xs font-medium">Bottleneck</div>
          <svg class="h-4 w-4 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7" />
          </svg>
          <div class="flex flex-col gap-3">
            <div class="rounded-lg border border-green-200 bg-green-50 px-3 py-2 text-xs font-medium">Reconstruction Head</div>
            <div class="rounded-lg border border-yellow-200 bg-yellow-50 px-3 py-2 text-xs font-medium">Prediction Head</div>
          </div>
        </div>

        <!-- Multi Task Learning Model -->
        <div v-else-if="flowType === 'multi_task'" class="flex items-center justify-center gap-4">
          <div class="rounded-lg border border-zinc-200 bg-zinc-100 px-3 py-2 text-xs font-medium">Input Features</div>
          <svg class="h-4 w-4 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7" />
          </svg>
          <div class="rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-xs font-medium">Shared Trunk</div>
          <svg class="h-4 w-4 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7" />
          </svg>
          <div class="flex flex-col gap-3">
            <div class="rounded-lg border border-yellow-200 bg-yellow-50 px-3 py-2 text-xs font-medium">Main Task Head</div>
            <div class="rounded-lg border border-green-200 bg-green-50 px-3 py-2 text-xs font-medium">Auxiliary Head</div>
          </div>
          <svg class="h-4 w-4 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7" />
          </svg>
          <div class="rounded-lg border border-cyan-200 bg-cyan-50 px-3 py-2 text-xs font-medium">Prediction Output</div>
        </div>

        <!-- Time Aware Tabular Model -->
        <div v-else-if="flowType === 'time_aware'" class="flex items-center justify-center gap-4">
          <div class="rounded-lg border border-zinc-200 bg-zinc-100 px-3 py-2 text-xs font-medium">Input Features</div>
          <svg class="h-4 w-4 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7" />
          </svg>
          <div class="flex flex-col gap-3">
            <div class="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs font-medium">Temporal Gate</div>
            <div class="rounded-lg border border-pink-200 bg-pink-50 px-3 py-2 text-xs font-medium">Gated Features</div>
          </div>
          <svg class="h-4 w-4 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7" />
          </svg>
          <div class="rounded-lg border border-indigo-200 bg-indigo-50 px-3 py-2 text-xs font-medium">Concat Raw + Gated</div>
          <svg class="h-4 w-4 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7" />
          </svg>
          <div class="rounded-lg border border-cyan-200 bg-cyan-50 px-3 py-2 text-xs font-medium">Prediction Output</div>
        </div>

        <!-- Tree Teacher Distillation Model -->
        <div v-else-if="flowType === 'distillation'" class="flex items-center justify-center gap-4">
          <div class="rounded-lg border border-zinc-200 bg-zinc-100 px-3 py-2 text-xs font-medium">Input Features</div>
          <svg class="h-4 w-4 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7" />
          </svg>
          <div class="flex flex-col gap-3">
            <div class="rounded-lg border border-green-200 bg-green-50 px-3 py-2 text-xs font-medium">Tree Teacher Ensemble</div>
            <div class="rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-xs font-medium">Neural Student</div>
          </div>
          <svg class="h-4 w-4 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7" />
          </svg>
          <div class="rounded-lg border border-cyan-200 bg-cyan-50 px-3 py-2 text-xs font-medium">Prediction Output</div>
        </div>
      </div>

      <!-- Details Section -->
      <div class="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div class="rounded-md border border-zinc-200 bg-white p-4">
          <p class="mb-2 text-sm font-semibold text-zinc-800">Layer Architecture Breakdown</p>
          <ul class="space-y-3 text-xs text-zinc-600">
            <li v-for="bullet in layers" :key="bullet" class="relative flex flex-col gap-0.5 pl-4">
              <span class="absolute left-0 top-1.5 h-1.5 w-1.5 rounded-full bg-zinc-400" />
              <span class="font-semibold text-zinc-800">{{ parseBullet(bullet).term }}</span>
              <span class="leading-relaxed">{{ parseBullet(bullet).definition }}</span>
            </li>
          </ul>
        </div>

        <div v-if="terminology.length > 0" class="rounded-md border border-blue-100 bg-blue-50/50 p-4">
          <p class="mb-2 text-sm font-semibold text-blue-900">Key Terminology</p>
          <ul class="space-y-3 text-xs text-blue-800/80">
            <li v-for="t in terminology" :key="t.term" class="relative flex flex-col gap-0.5 pl-4">
              <span class="absolute left-0 top-1.5 h-1.5 w-1.5 rounded-full bg-blue-400" />
              <span class="font-semibold text-blue-900">{{ t.term }}</span>
              <span class="leading-relaxed">{{ t.definition }}</span>
            </li>
          </ul>
        </div>
        <div v-else class="flex items-center justify-center rounded-md border border-zinc-200 bg-zinc-50 p-4 text-center">
          <p class="text-xs text-zinc-400">No complex terminology explicitly defined for this baseline model.</p>
        </div>
      </div>
    </div>
  </Modal>
</template>

<script setup lang="ts">
import { computed } from "vue";
import Modal from "~/components/General/Modal.vue";

const props = defineProps<{
  isOpen: boolean;
  framework: "pytorch" | "tensorflow";
  mode: string;
}>();

defineEmits<{ close: [] }>();

// Re-implemented from modelPreview.util.ts
const modelMeta = computed(() => {
  const outputLabel = props.mode === "quantile_regression" ? "Quantile Output (P80)" : "Prediction Output";

  switch (props.mode) {
    case "mlp_dense":
      return {
        title: "Multi-Layer Perceptron (Dense)",
        summary: "Standard dense neural network with fully connected hidden layers.",
        flowType: "sequential",
        nodes: [
          { label: "Input Features", bg: "#f4f4f5", border: "#d4d4d8" },
          { label: "Dense Hidden Block 1", bg: "#f5f3ff", border: "#c4b5fd" },
          { label: "Dense Hidden Block 2", bg: "#f5f3ff", border: "#c4b5fd" },
          { label: outputLabel, bg: "#ecfeff", border: "#67e8f9" },
        ],
        layers: [
          "Input Features: The raw data you provide (like numbers and categories), cleaned up so the AI can easily read it.",
          "Dense Hidden Blocks: The main 'brain' layers where the AI mixes and matches the data to find hidden patterns.",
          "Output Head: The final step that takes the AI's thoughts and spits out the actual guess (like predicting a house price).",
        ],
        terminology: [
          { term: "Latent State", definition: "A hidden, internal summary of the data that the network creates to help it guess the final answer. Think of it like a detective's private notebook." },
          { term: "Logits", definition: "The raw, unpolished scores the model spits out before we mathematically convert them into nice, predictable percentages (like 85% probability)." },
          { term: "Nonlinear Combinations", definition: "Instead of just drawing straight, simplistic lines to find patterns, the model is allowed to draw curves, squiggles, and highly complex shapes to solve harder problems." },
          { term: "Standardized", definition: "Scaling all the input numbers so they are roughly the same size, which prevents large numbers (like a $1,000,000 house price) from completely overpowering small numbers (like 2 bedrooms) during training." }
        ],
      };

    case "linear_glm_baseline":
      return {
        title: "Linear / GLM Baseline",
        summary: "Single linear head model. Fast and interpretable benchmark.",
        flowType: "sequential",
        nodes: [
          { label: "Input Features", bg: "#f4f4f5", border: "#d4d4d8" },
          { label: "Linear / GLM Head", bg: "#fefce8", border: "#fde047" },
          { label: outputLabel, bg: "#ecfeff", border: "#67e8f9" },
        ],
        layers: [
          "Input Features: The raw starting data (like numbers and categories) ready for the model to look at.",
          "Linear / GLM Head: A simple math step that just multiplies each input by a specific 'importance score' and adds them all up.",
          "Output Head: The final result, giving us a highly understandable baseline to compare our fancy AI models against.",
        ],
        terminology: [
          { term: "GLM (Generalized Linear Model)", definition: "A straightforward, classic statistical approach. It essentially takes all your inputs, multiplies them by a specific weight (importance score), and adds them all up to get an answer." },
          { term: "Baseline", definition: "A simple, bare-bones model we build first just to see how hard the problem is. If a massive, complex AI can't beat this simple baseline, we know the fancy AI isn't worth using." },
          { term: "Interpretable", definition: "Because the math is so simple (just adding up multiplied numbers), a human can look exactly at the formula and explain precisely why the model made its decision." }
        ],
      };

    case "wide_and_deep":
      return {
        title: "Wide & Deep",
        summary: "Combines memorization from a linear path with generalization from a deep path.",
        flowType: "wide_deep",
        nodes: [],
        layers: [
          "Wide Branch: A fast, direct path that just blatantly memorizes simple, obvious rules (like 'if burger, suggest fries').",
          "Deep Branch Blocks: A more complex thinking path that tries to learn deeper, creative generalizations.",
          "Merge Layer: The part where the AI glues together the obvious rules and the deep creative thoughts into one final answer.",
        ],
        terminology: [
          { term: "Memorization (Wide)", definition: "The network's ability to blatantly memorize facts it has seen many times. For example, 'if order includes a burger, suggest fries'." },
          { term: "Generalization (Deep)", definition: "The network's ability to be creative and guess correctly on things it has never seen before, like suggesting a new abstract pairing based on underlying user taste profiles." },
          { term: "Concatenates", definition: "A fancy programming word for gluing two different lists of numbers together into one big list side-by-side." }
        ],
      };

    case "tabresnet":
      return {
        title: "TabResNet",
        summary: "Residual skip connections improve gradient flow and stability in deeper tabular networks.",
        flowType: "sequential",
        nodes: [
          { label: "Input Features", bg: "#f4f4f5", border: "#d4d4d8" },
          { label: "Input Projection", bg: "#eef2ff", border: "#a5b4fc" },
          { label: "Residual Block 1", bg: "#f5f3ff", border: "#c4b5fd" },
          { label: "Residual Block 2", bg: "#f5f3ff", border: "#c4b5fd" },
          { label: outputLabel, bg: "#ecfeff", border: "#67e8f9" },
        ],
        layers: [
          "Input Projection: Upgrades your basic data into a richer format so the AI has more mathematical 'room' to untangle messy patterns.",
          "Residual Blocks: The core learning steps. They use 'skip connections'—meaning if a step is confusing, the data can safely bypass or skip it without breaking the AI.",
          "Output Head: Takes all those careful, step-by-step corrections (residuals) and turns them into a final prediction.",
        ],
        terminology: [
          { term: "Skip Connections", definition: "A clever trick where we provide the network a 'shortcut bypass'. If a layer's complicated math isn't helping, the model can literally just skip over it. This prevents the whole system from breaking when we make it very deep." },
          { term: "Hidden Space / Higher-Dimensional", definition: "Imagine taking a 2D drawing and popping it out into a 3D sculpture. The model stretches your data into many dimensions so it has more 'room' to untangle messy patterns." },
          { term: "Residual", definition: "Instead of trying to learn the entire final answer from scratch at every single step, each layer just learns the 'residual'—the tiny remaining correction needed to fix the previous step's guess." }
        ],
      };

    case "imbalance_aware":
      return {
        title: "Imbalance-Aware Classifier",
        summary: "Same network family with weighted objective to prioritize minority classes.",
        flowType: "sequential",
        nodes: [
          { label: "Input Features", bg: "#f4f4f5", border: "#d4d4d8" },
          { label: "Dense Encoder", bg: "#f0f9ff", border: "#7dd3fc" },
          { label: "Classifier Head", bg: "#fef3c7", border: "#fcd34d" },
          { label: "Class-Weighted Loss", bg: "#fee2e2", border: "#fca5a5" },
          { label: "Class Probabilities", bg: "#ecfeff", border: "#67e8f9" },
        ],
        layers: [
          "Dense Encoder: Translates your raw spreadsheet data into a secret machine language the AI can easily understand.",
          "Classifier Head: Looks at that translated data and spits out its confidence scores for each possible category.",
          "Class-Weighted Loss Path: A harsh grading system that severely punishes the AI if it misses the extremely rare events it was supposed to catch.",
        ],
        terminology: [
          { term: "Minority Class", definition: "A scenario where the thing you are looking for is extremely rare. For example, tracking credit card fraud where 99.9% of transactions are perfectly normal, and only 0.1% are fraudulent." },
          { term: "Class-Weighted Loss", definition: "We heavily penalize the AI if it misses the rare event. It's like telling the model: 'It's okay to accidentally flag a normal transaction as fraud, but if you let a real fraudster slip by, you fail instantly.'" },
          { term: "Encoder", definition: "The part of the AI that reads your raw data (like spreadsheets) and 'encodes' or translates it into a secret machine language that the rest of the AI can easily understand." }
        ],
      };

    case "quantile_regression":
      return {
        title: "Quantile Regression",
        summary: "Predicts distribution quantiles (P80 shown) instead of only a mean point estimate.",
        flowType: "sequential",
        nodes: [
          { label: "Input Features", bg: "#f4f4f5", border: "#d4d4d8" },
          { label: "Dense Encoder", bg: "#f0f9ff", border: "#7dd3fc" },
          { label: "Quantile Head (tau=0.8)", bg: "#ede9fe", border: "#c4b5fd" },
          { label: "Pinball Loss", bg: "#fef3c7", border: "#fcd34d" },
          { label: "P80 Forecast", bg: "#ecfeff", border: "#67e8f9" },
        ],
        layers: [
          "Dense Encoder: Looks at your data and builds internal 'latent' clues—hidden details that help it make a guess.",
          "Quantile Head (tau=0.8): Instead of making one exact guess, it draws a boundary line where it is 80% sure the real answer will be below it.",
          "Pinball Loss Path: A special grading system that harshly punishes the AI more for guessing too low than for guessing too high (or vice versa).",
        ],
        terminology: [
          { term: "Quantile Regression", definition: "Instead of just guessing the 'average' expected outcome, this model guesses specific boundary lines. E.g., 'I am 80% sure the delivery will arrive before 5:00 PM'." },
          { term: "Tau (τ)", definition: "A symbol representing the specific boundary line we want to draw. A Tau of 0.8 means we want the 80th percentile mark." },
          { term: "Pinball Loss", definition: "A unique grading system. If we want to predict the absolute worst-case scenario (like maximum possible traffic), this mathematically punishes the model way harder for underestimating the traffic than overestimating it." },
          { term: "Latent Clues", definition: "Hidden, underlying features the AI figures out on its own. It's like a detective writing down private notes that aren't in the official police report, but help solve the case." }
        ],
      };

    case "calibrated_classifier":
      return {
        title: "Calibrated Classifier",
        summary: "Adds label smoothing in training to reduce overconfident predictions and improve probability behavior.",
        flowType: "sequential",
        nodes: [
          { label: "Input Features", bg: "#f4f4f5", border: "#d4d4d8" },
          { label: "Dense Encoder", bg: "#f0f9ff", border: "#7dd3fc" },
          { label: "Classifier Head", bg: "#fef3c7", border: "#fcd34d" },
          { label: "Label Smoothing Objective", bg: "#dcfce7", border: "#86efac" },
          { label: "Calibrated Probabilities", bg: "#ecfeff", border: "#67e8f9" },
        ],
        layers: [
          "Dense Encoder: Organizes the data nicely so the AI can easily draw a clean line between different categories (like breaking out the 'Yes' vs. 'No' answers).",
          "Classifier Head: Outputs exactly how confident it is about its guess as a percentage.",
          "Label Smoothing Objective: The training rule that forces the AI to be humble and slightly unsure, preventing arrogant, overconfident mistakes.",
        ],
        terminology: [
          { term: "Calibration", definition: "Ensuring the AI isn't falsely confident. If an AI says it is 90% sure it will rain, calibration testing proves it actually rains exactly 9 out of 10 times it makes that claim." },
          { term: "Label Smoothing", definition: "A trick to stop the AI from acting too strictly. Instead of letting the AI be 100% certain about training answers, we cap its maximum confidence at 90%. This forces it to remain slightly humble and open-minded when facing new, bizarre data." },
          { term: "Linearly Separate", definition: "Pushing the data around in a mathematical space until you can draw a literal, straight physical line between the 'Yes' answers and the 'No' answers." }
        ],
      };

    case "entity_embeddings":
      return {
        title: "Entity Embeddings",
        summary: "Projects sparse/tabular inputs into compact latent features before final prediction.",
        flowType: "sequential",
        nodes: [
          { label: "Input Features", bg: "#f4f4f5", border: "#d4d4d8" },
          { label: "Embedding Projection Layer", bg: "#eef2ff", border: "#a5b4fc" },
          { label: "Dense Predictor Stack", bg: "#f0f9ff", border: "#7dd3fc" },
          { label: outputLabel, bg: "#ecfeff", border: "#67e8f9" },
        ],
        layers: [
          "Embedding Projection Layer: Takes hard-to-read categories (like Zip Codes) and turns them into easier 'latent' coordinates on a map.",
          "Dense Predictor Stack: Uses those map coordinates to find deeper patterns useful for predicting your specific goal.",
          "Output Head: Takes those final thought patterns and turns them into your actual answer.",
        ],
        terminology: [
          { term: "Categorical IDs", definition: "Data that exists as distinct groups or labels rather than numbers (like Zip Codes, User IDs, or Product Brands)." },
          { term: "Embeddings / Latent Coordinates", definition: "An incredibly powerful trick that turns words or categories into hidden 'latent' coordinates on a map. For example, the AI might learn on its own to place 'Apple' and 'Banana' very close together on this map because they behave similarly." },
          { term: "Sparse", definition: "A situation where most of your data is entirely zeros or completely empty blank spaces." }
        ],
      };

    case "autoencoder_head":
      return {
        title: "Autoencoder + Head",
        summary: "Jointly learns a compact latent representation and a supervised prediction path.",
        flowType: "autoencoder",
        nodes: [],
        layers: [
          "Encoder: Squashes all your data down into a tiny, ultra-compact summary called a 'latent' representation.",
          "Latent Bottleneck: The tiny wire the data gets squeezed through, which physically forces the AI to drop useless noise and only keep what matters.",
          "Reconstruction Head: A separate path that tries to unpack that squishy summary back into its full, original form to make sure no important details were lost.",
          "Prediction Head: Uses that exact same clear, noise-free summary to actually predict what you want to know.",
        ],
        terminology: [
          { term: "Autoencoder", definition: "An AI that is literally forced to play a game of 'telephone' with itself. It squashes data down, sends it through a tiny wire, and then tries to perfectly rebuild the original data on the other side." },
          { term: "Bottleneck / Latent Representation", definition: "The tiny wire in the telephone game. Because the data has to squeeze through this restrictive bottleneck, the AI is physically forced to format it into a tight 'latent' code, throwing away useless noise and purely memorizing the most critically important structural concepts." },
          { term: "Reconstruction / Decoder", definition: "The part of the AI whose only job is to unpack the tightly compressed ZIP file of data back into its original size." }
        ],
      };

    case "multi_task_learning":
      return {
        title: "Multi-Task Learning",
        summary: "Trains one shared representation with multiple supervised heads.",
        flowType: "multi_task",
        nodes: [],
        layers: [
          "Shared Trunk: The foundational base of the AI. It does all the heavy lifting to learn basic patterns before branching off.",
          "Main Task Head: The part of the AI's 'brain' that focuses entirely on answering your primary question.",
          "Auxiliary Head: A 'side-quest' brain. By forcing the AI to solve this extra related problem, it accidentally gets vastly smarter at the main task.",
        ],
        terminology: [
          { term: "Multi-Task", definition: "Training the AI to solve two different problems at the exact same time using the exact same brain. Surprisingly, learning two related things together often makes the AI vastly smarter at both." },
          { term: "Shared Trunk", definition: "The base foundational layers of the AI. Like the trunk of a tree, it does all the heavy lifting before branching off into separate 'Heads' for specific tasks." },
          { term: "Auxiliary Supervision", definition: "A fake 'side-quest' we force the AI to solve during training. We don't actually care about the answer to the side-quest, it entirely exists just to force the AI to learn better foundational patterns." }
        ],
      };

    case "time_aware_tabular":
      return {
        title: "Time-Aware Tabular",
        summary: "Applies temporal gating before deep prediction to emphasize time-derived patterns.",
        flowType: "time_aware",
        nodes: [],
        layers: [
          "Temporal Gate: Like a bouncer looking at a clock. It decides which time-related clues (like 'It's Friday') are important right now, and which are just noise.",
          "Gated Features: The cleaned-up data where the important time clues are mathematically highlighted and the irrelevant ones are muted.",
          "Concat Raw + Gated: Glues your regular data and your time clues together so the final AI step can use both at once.",
        ],
        terminology: [
          { term: "Temporal", definition: "A fancy word meaning 'related to time'. For example, noticing that ice cream sales fundamentally behave differently in August versus December." },
          { term: "Gating Mechanism", definition: "Like a bouncer at a club. The AI learns to automatically open the 'gate' to let highly relevant seasonal information pass through, but physically closes the gate to block out noisy, irrelevant time data." },
          { term: "Seasonality", definition: "Repeating patterns that naturally loop on a schedule (e.g., higher server traffic every morning at 9 AM, or higher retail sales every Friday)." }
        ],
      };

    case "tree_teacher_distillation":
      return {
        title: "Tree-Teacher Distillation",
        summary: "A tree teacher guides a compact neural student during training.",
        flowType: "distillation",
        nodes: [],
        layers: [
          "Tree Teacher Ensemble: A massive, slow 'forest model' (made of many decision trees) that already perfectly understands the data.",
          "Neural Student: A tiny, fast AI that is desperately trying to copy the massive teacher's behavior so it can run quickly on a cell phone.",
          "Teacher-to-Student Distillation Path: The grading system. The tiny student is graded on how perfectly it mimics all the teacher's nuanced doubts and guesses.",
        ],
        terminology: [
          { term: "Knowledge Distillation", definition: "A process where we have a massive, slow, genius AI (the Teacher) teach a tiny, hyper-fast AI (the Student) how to emulate its exact behavior so we can run it affordably on a cell phone." },
          { term: "Tree Ensemble / Forest Model", definition: "An AI built not from 'neural networks' but from hundreds of 'decision trees' (essentially massive flowcharts of Yes/No questions) voting together on an answer. It's frequently called a 'Forest Model' because it's a giant group of trees." },
          { term: "Soft Probabilities", definition: "Instead of telling the student the final answer is simply 'Cat', the Teacher tells the student 'I am 82% sure it is a Cat, 15% sure it is a Dog, and 3% sure it is a Car.' The student learns infinitely faster by observing these nuanced doubts." }
        ],
      };

    default:
      return {
        title: props.framework === "pytorch" ? "PyTorch Neural Net" : "TensorFlow Neural Net",
        summary: "Dense neural network baseline.",
        flowType: "sequential",
        nodes: [
          { label: "Input Features", bg: "#f4f4f5", border: "#d4d4d8" },
          { label: outputLabel, bg: "#ecfeff", border: "#67e8f9" },
        ],
        layers: [
          "Input Features: The raw columns of data (like an Excel sheet), properly prepared so the AI can read them.",
          "Model Core: The internal 'brain' where the magic happens and hidden patterns are discovered.",
          "Output Head: The final step that turns those discovered patterns into your actual prediction.",
        ],
        terminology: [
          { term: "Features", definition: "The individual columns of data you feed the AI (like 'Age', 'Height', or 'Zip Code')." },
          { term: "Tabular", definition: "Data that lives beautifully in traditional rows and columns, exactly like a standard Excel spreadsheet or SQL database." },
          { term: "Prediction", definition: "The final guess the AI spits out after crunching all the numbers." }
        ],
      };
  }
});

const title = computed(() => modelMeta.value.title);
const summary = computed(() => modelMeta.value.summary);
const flowType = computed(() => modelMeta.value.flowType);
const nodes = computed(() => modelMeta.value.nodes);
const layers = computed(() => modelMeta.value.layers);
const terminology = computed(() => modelMeta.value.terminology);

function parseBullet(bullet: string) {
  const idx = bullet.indexOf(":");
  if (idx < 0) return { term: bullet.trim(), definition: "" };
  return {
    term: bullet.slice(0, idx).trim(),
    definition: bullet.slice(idx + 1).trim(),
  };
}
</script>
