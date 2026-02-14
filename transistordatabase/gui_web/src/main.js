import { createApp } from 'vue'
import './style.css'
import App from './App.vue'

console.log('[main.js] Initializing Vue app...')

const app = createApp(App)

// Global error handler
app.config.errorHandler = (err, instance, info) => {
  console.error('[Vue Error Handler]:', {
    error: err,
    message: err.message,
    stack: err.stack,
    component: instance?.$options?.name || 'Unknown',
    info: info
  })
}

// Global warning handler (development only)
app.config.warnHandler = (msg, instance, trace) => {
  console.warn('[Vue Warning]:', msg, trace)
}

app.mount('#app')
console.log('[main.js] Vue app mounted successfully')
