<script setup lang="ts">
</script>

<template>
  <div class="app-shell">
    <router-view />
  </div>
</template>

<style scoped>
.app-shell {
  height: 100vh;
  display: flex;
  background: var(--bg-page);
  color: var(--text-primary);
  font-family: var(--font-body);
}
</style>
