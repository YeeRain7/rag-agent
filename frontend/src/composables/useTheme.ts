import { ref, onMounted } from 'vue'

const isDark = ref(false)

export function useTheme() {
  onMounted(() => {
    isDark.value = localStorage.getItem('theme') === 'dark'
    apply()
  })

  function toggle() {
    isDark.value = !isDark.value
    localStorage.setItem('theme', isDark.value ? 'dark' : 'light')
    apply()
  }

  function apply() {
    document.documentElement.setAttribute('data-theme', isDark.value ? 'dark' : 'light')
  }

  return { isDark, toggle }
}
