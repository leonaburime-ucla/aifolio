<template>
  <div v-if="columns.length > 0" class="rounded-lg border border-zinc-200">
    <div class="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-200 p-3">
      <span class="text-xs text-zinc-500">{{ rows.length }} rows</span>
      <input
        v-model="search"
        type="search"
        class="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm text-zinc-900 placeholder:text-zinc-400 focus:border-zinc-500 focus:outline-none sm:w-64"
        placeholder="Search table..."
      />
    </div>
    <div class="overflow-auto">
      <table class="w-full border-collapse text-xs">
        <thead class="bg-zinc-50">
          <tr>
            <th v-for="col in columns" :key="col" class="whitespace-nowrap border-b border-zinc-200 px-3 py-2 text-left font-semibold text-zinc-700">
              <button type="button" class="inline-flex items-center gap-1" @click="toggleSort(col)">
                <span>{{ col }}</span>
                <span v-if="sortKey === col" aria-hidden="true">{{ sortDirection === 'asc' ? '^' : 'v' }}</span>
              </button>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(row, idx) in pagedRows" :key="rowKey(row, idx)" class="odd:bg-white even:bg-zinc-50">
            <td v-for="col in columns" :key="col" class="whitespace-nowrap border-b border-zinc-100 px-3 py-2 text-zinc-800">
              <slot :name="col" :row="row" :value="row[col]">
                {{ formatCell(row[col]) }}
              </slot>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    <div class="flex items-center justify-between gap-3 p-3">
      <button
        type="button"
        class="rounded-md border border-zinc-300 px-3 py-1.5 text-xs font-medium text-zinc-700 disabled:cursor-not-allowed disabled:opacity-50"
        :disabled="page === 0"
        @click="page -= 1"
      >
        Previous
      </button>
      <span class="text-xs text-zinc-500">Page {{ page + 1 }} of {{ pageCount }}</span>
      <button
        type="button"
        class="rounded-md border border-zinc-300 px-3 py-1.5 text-xs font-medium text-zinc-700 disabled:cursor-not-allowed disabled:opacity-50"
        :disabled="page + 1 >= pageCount"
        @click="page += 1"
      >
        Next
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";

const props = defineProps<{
  rows: Record<string, unknown>[];
  columns: string[];
}>();

const search = ref("");
const sortKey = ref<string | null>(null);
const sortDirection = ref<"asc" | "desc">("asc");
const page = ref(0);
const pageSize = 25;

const filteredRows = computed(() => {
  const query = search.value.toLowerCase().trim();
  if (!query) return props.rows;
  return props.rows.filter((row) =>
    props.columns.some((column) => String(row[column] ?? "").toLowerCase().includes(query))
  );
});

const sortedRows = computed(() => {
  const key = sortKey.value;
  if (!key) return filteredRows.value;
  const direction = sortDirection.value === "asc" ? 1 : -1;
  return [...filteredRows.value].sort((left, right) => compareValues(left[key], right[key]) * direction);
});

const pageCount = computed(() => Math.max(1, Math.ceil(sortedRows.value.length / pageSize)));
const pagedRows = computed(() => {
  const start = Math.min(page.value, pageCount.value - 1) * pageSize;
  return sortedRows.value.slice(start, start + pageSize);
});

watch(
  () => [props.rows, props.columns],
  () => {
    search.value = "";
    sortKey.value = null;
    sortDirection.value = "asc";
    page.value = 0;
  },
  { deep: false }
);

function toggleSort(column: string) {
  if (sortKey.value === column) {
    sortDirection.value = sortDirection.value === "asc" ? "desc" : "asc";
    return;
  }
  sortKey.value = column;
  sortDirection.value = "asc";
}

function compareValues(left: unknown, right: unknown): number {
  const leftNumber = Number(left);
  const rightNumber = Number(right);
  if (Number.isFinite(leftNumber) && Number.isFinite(rightNumber)) return leftNumber - rightNumber;
  return String(left ?? "").localeCompare(String(right ?? ""), undefined, { numeric: true, sensitivity: "base" });
}

function formatCell(value: unknown): string {
  return value == null ? "" : String(value);
}

function rowKey(row: Record<string, unknown>, index: number): string {
  const signature = props.columns
    .slice(0, 3)
    .map((column) => String(row[column] ?? ""))
    .join("|");
  return `${index}:${signature}`;
}
</script>
