<template>
  <Teleport to="body">
    <div
      v-if="isOpen"
      :class="[
        'fixed inset-0 z-50 flex justify-center p-4',
        position === 'top' ? 'items-start pt-4 md:pt-8' : 'items-center'
      ]"
    >
      <!-- Backdrop -->
      <div
        class="absolute inset-0 bg-black/50 backdrop-blur-sm transition-opacity"
        @click="$emit('close')"
        aria-hidden="true"
      />

      <!-- Content -->
      <div
        class="relative z-50 w-full max-w-3xl rounded-2xl bg-white p-6 shadow-xl ring-1 ring-zinc-900/5 transition-all max-h-[90vh] flex flex-col"
        role="dialog"
        aria-modal="true"
      >
        <div class="mb-4 flex items-center justify-between border-b border-zinc-100 pb-4 shrink-0">
          <h2 class="text-lg font-semibold text-zinc-900">{{ title }}</h2>
          <button
            @click="$emit('close')"
            class="rounded-full p-2 text-zinc-500 hover:bg-zinc-100 hover:text-zinc-700 transition"
            aria-label="Close modal"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        <div class="overflow-y-auto flex-1 pr-1">
          <slot />
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { watch, onBeforeUnmount } from "vue";

const props = withDefaults(
  defineProps<{
    isOpen: boolean;
    title?: string;
    position?: "center" | "top";
  }>(),
  {
    title: "",
    position: "center",
  }
);

const emit = defineEmits<{ close: [] }>();

function handleKeyDown(event: Event) {
  if ((event as KeyboardEvent).key === "Escape") {
    emit("close");
  }
}

function canUseDocumentBody() {
  return typeof document !== "undefined" && typeof window !== "undefined" && !!document.body;
}

function setBodyScrollLock(locked: boolean) {
  if (!canUseDocumentBody()) return;
  document.body.style.overflow = locked ? "hidden" : "";
}

function setEscapeListener(enabled: boolean) {
  if (typeof window === "undefined") return;
  if (enabled) {
    window.addEventListener("keydown", handleKeyDown);
  } else {
    window.removeEventListener("keydown", handleKeyDown);
  }
}

watch(
  () => props.isOpen,
  (open) => {
    setBodyScrollLock(open);
    setEscapeListener(open);
  },
  { immediate: true }
);

onBeforeUnmount(() => {
  setBodyScrollLock(false);
  setEscapeListener(false);
});
</script>
