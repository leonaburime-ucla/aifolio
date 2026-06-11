<template>
  <nav class="sticky top-0 z-40 border-b border-zinc-200 bg-white">
    <div class="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
      <div class="flex items-center gap-6 text-sm text-zinc-900">
        <NuxtLink to="/" :class="navItemClass(route.path === '/')">
          AI Chat
        </NuxtLink>

        <NuxtLink to="/agentic-research" :class="navItemClass(route.path === '/agentic-research')">
          Agentic Research
        </NuxtLink>

        <div class="group relative">
          <button
            type="button"
            :class="[
              'flex items-center gap-1 rounded-md px-2 py-1 transition',
              route.path.startsWith('/ml')
                ? 'bg-zinc-300 text-zinc-950'
                : 'text-zinc-900 hover:bg-zinc-100',
            ]"
          >
            Machine Learning
            <span class="text-xs">&#9662;</span>
          </button>
          <div
            class="invisible absolute left-0 top-full pt-2 opacity-0 transition group-hover:visible group-hover:opacity-100 group-focus-within:visible group-focus-within:opacity-100"
          >
            <div class="w-48 rounded-lg border border-zinc-200 bg-white p-2 text-sm text-zinc-700 shadow-lg">
              <NuxtLink
                to="/ml/pytorch"
                class="block rounded-md px-3 py-2 transition hover:bg-zinc-100"
              >
                PyTorch
              </NuxtLink>
              <NuxtLink
                to="/ml/tensorflow"
                class="block rounded-md px-3 py-2 transition hover:bg-zinc-100"
              >
                TensorFlow
              </NuxtLink>
            </div>
          </div>
        </div>

        <NuxtLink to="/ag-ui" :class="navItemClass(route.path === '/ag-ui')">
          Agentic UI
        </NuxtLink>
      </div>

      <NuxtLink to="/" class="text-sm font-semibold text-zinc-900">
        AIFolio in Vue by Leon A.
      </NuxtLink>
    </div>
  </nav>
</template>

<script setup lang="ts">
const route = useRoute();

function navItemClass(isActive: boolean): string {
  return `rounded-md px-2 py-1 transition ${
    isActive ? "bg-zinc-300 text-zinc-950" : "text-zinc-900 hover:bg-zinc-100"
  }`;
}
</script>
