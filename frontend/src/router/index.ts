import { createRouter, createWebHistory } from 'vue-router'
import ChatView from '../views/ChatView.vue'
import KnowledgeView from '../views/KnowledgeView.vue'

const routes = [
  { path: '/', redirect: '/chat' },
  { path: '/chat', component: ChatView },
  { path: '/chat/:sessionId', component: ChatView },
  { path: '/knowledge', component: KnowledgeView },
]

export default createRouter({
  history: createWebHistory(),
  routes
})
