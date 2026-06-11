<template>
  <div v-if="columns.length > 0" class="overflow-auto rounded-lg border border-zinc-200">
    <table ref="tableEl" class="display compact stripe w-full text-xs">
      <thead>
        <tr>
          <th v-for="col in columns" :key="col" class="text-left">{{ col }}</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(row, idx) in rows" :key="idx">
          <td v-for="col in columns" :key="col">
            <slot :name="col" :row="row" :value="row[col]">
              {{ row[col] ?? '' }}
            </slot>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch, onBeforeUnmount, nextTick } from "vue";

const props = defineProps<{
  rows: Record<string, unknown>[];
  columns: string[];
}>();

const tableEl = ref<HTMLTableElement | null>(null);
let dt: any = null;

async function initDataTable() {
  if (!tableEl.value || props.columns.length === 0) return;
  if (dt) {
    dt.destroy(false);
    dt = null;
  }
  const DataTable = (await import("datatables.net")).default;
  await import("datatables.net-dt");
  await nextTick();
  if (!tableEl.value) return;
  const isAlready = (DataTable as any).isDataTable(tableEl.value);
  if (isAlready) return;
  dt = new DataTable(tableEl.value, {
    paging: true,
    pageLength: 25,
    scrollX: true,
    searching: true,
    ordering: true,
    info: true,
    autoWidth: false,
  });
}

onMounted(() => {
  if (props.rows.length > 0) initDataTable();
});

watch(
  () => [props.rows, props.columns],
  async () => {
    await nextTick();
    initDataTable();
  },
  { deep: true }
);

onBeforeUnmount(() => {
  if (dt) {
    dt.destroy();
    dt = null;
  }
});
</script>

<style>
@import "datatables.net-dt/css/dataTables.dataTables.min.css";
</style>
