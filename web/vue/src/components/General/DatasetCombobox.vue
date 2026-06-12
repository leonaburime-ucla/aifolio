<template>
  <div class="relative">
    <input
      v-model="search"
      type="text"
      class="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm text-zinc-900 placeholder:text-zinc-400 focus:border-zinc-500 focus:outline-none"
      placeholder="Search datasets..."
      @focus="onFocus"
      @input="isSearching = true"
      @blur="onBlur"
      @keydown.enter.prevent="commitSearch"
    />
    <ul
      v-if="isOpen && filteredOptions.length > 0"
      class="absolute z-20 mt-1 max-h-48 w-full overflow-auto rounded-md border border-zinc-200 bg-white py-1 shadow-lg"
    >
      <li
        v-for="opt in filteredOptions"
        :key="opt.id"
        class="cursor-pointer px-3 py-2 text-sm text-zinc-700 hover:bg-zinc-100"
        :class="{ 'bg-zinc-100 font-medium': opt.id === selectedId }"
        @mousedown.prevent="selectOption(opt.id)"
      >
        {{ opt.label }}
      </li>
    </ul>
    <p v-if="isOpen && filteredOptions.length === 0" class="mt-1 text-xs text-zinc-500">
      No datasets found.
    </p>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from "vue";

interface DatasetOption {
  id: string;
  label: string;
}

const props = defineProps<{
  options: DatasetOption[];
  selectedId: string | null;
}>();

const emit = defineEmits<{ change: [id: string] }>();

const search = ref("");
const isOpen = ref(false);
const isSearching = ref(false);

watch(
  () => props.selectedId,
  (id) => {
    if (id && !isOpen.value) {
      const match = props.options.find((o) => o.id === id);
      if (match) search.value = match.label;
    }
  },
  { immediate: true }
);

watch(
  () => props.options,
  (opts) => {
    if (props.selectedId && !search.value) {
      const match = opts.find((o) => o.id === props.selectedId);
      if (match) search.value = match.label;
    }
  }
);

const filteredOptions = computed(() => {
  const q = isOpen.value && !isSearching.value ? "" : search.value.toLowerCase();
  if (!q) return props.options;
  return props.options.filter((o) => o.label.toLowerCase().includes(q));
});

function onFocus(event: FocusEvent) {
  isOpen.value = true;
  isSearching.value = false;
  if (event.target instanceof HTMLInputElement) {
    event.target.select();
  }
}

function selectOption(id: string) {
  search.value = props.options.find((o) => o.id === id)?.label ?? id;
  isOpen.value = false;
  isSearching.value = false;
  emit("change", id);
}

function findSearchMatch() {
  const query = search.value.trim().toLowerCase();
  if (!query) return null;
  return props.options.find((o) => o.id.toLowerCase() === query || o.label.toLowerCase() === query) ?? null;
}

function commitSearch() {
  const match = findSearchMatch();
  if (match) selectOption(match.id);
}

function onBlur() {
  setTimeout(() => {
    const match = findSearchMatch();
    if (match && match.id !== props.selectedId) {
      selectOption(match.id);
      return;
    }
    isOpen.value = false;
    isSearching.value = false;
    const selectedMatch = props.options.find((o) => o.id === props.selectedId);
    if (selectedMatch) search.value = selectedMatch.label;
  }, 150);
}
</script>
