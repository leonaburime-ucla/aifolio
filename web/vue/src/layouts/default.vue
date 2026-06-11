<template>
  <div class="antialiased font-sans">
    <Navbar />
    <slot />
  </div>
</template>

<script setup lang="ts">
import Navbar from "~/components/Nav/Navbar.vue";
</script>
