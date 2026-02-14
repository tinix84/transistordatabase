# Frontend Debugging Guide

## Comprehensive Logging Added

The Vue.js frontend now has extensive logging to track data flow and catch errors.

## What Was Added

### 1. API Service Logging (`src/services/api.js`)

**Initialization:**
```javascript
[API] Initialized with BASE_URL: http://localhost:8002
```

**Request Interceptor:**
```javascript
[API Request] GET http://localhost:8002/api/transistors
```

**Response Interceptor:**
```javascript
[API Response] GET /api/transistors: 200 OK
[API Response] Data type: Array[25]
```

**Transistor API Methods:**
```javascript
[transistorApi.getAll] Fetching all transistors...
[transistorApi.getAll] Success! Received 25 transistors
[transistorApi.getAll] Sample data: {metadata: {...}, electrical: {...}, ...}
```

**Error Logging:**
```javascript
[API Error]: {
  message: "Network Error",
  status: undefined,
  statusText: undefined,
  data: undefined,
  url: "/api/transistors"
}
```

### 2. App.vue Component Logging

**Initialization:**
```javascript
[App.vue] Component mounted, loading transistors...
[loadTransistors] Starting...
[loadTransistors] API returned: Array[25]
[loadTransistors] Data type: object Is Array: true
[loadTransistors] Data length: 25
[loadTransistors] State updated, transistors.value.length: 25
[loadTransistors] Finished, isLoading: false
[App.vue] Transistors loaded, count: 25
[App.vue] Initialization complete
```

**Error Handling:**
```javascript
[loadTransistors] Error: Error: Network Error
[loadTransistors] Error details: {
  message: "Network Error",
  response: undefined,
  stack: "Error: Network Error\n    at createError..."
}
```

### 3. SearchDatabase Component Logging

**Mounting:**
```javascript
[SearchDatabase] Mounted
[SearchDatabase] props.transistors: Array[25]
[SearchDatabase] transistors type: object Is Array: true
[SearchDatabase] transistors length: 25
```

**Prop Changes:**
```javascript
[SearchDatabase] transistors prop changed
[SearchDatabase] Old length: 0 New length: 25
[SearchDatabase] New value sample: {metadata: {...}, ...}
```

**Computed Properties:**
```javascript
[SearchDatabase] filteredTransistors computing with 25 transistors
```

**Warnings (if data is missing):**
```javascript
[SearchDatabase] availableTypes: transistors is not an array: undefined
```

### 4. Global Vue Error Handler (`src/main.js`)

**Initialization:**
```javascript
[main.js] Initializing Vue app...
[main.js] Vue app mounted successfully
```

**Error Handling:**
```javascript
[Vue Error Handler]: {
  error: TypeError(...),
  message: "Cannot read property 'map' of undefined",
  stack: "TypeError: Cannot read property...",
  component: "SearchDatabase",
  info: "render"
}
```

**Warnings:**
```javascript
[Vue Warning]: Failed to resolve component: transistor-card
```

## How to Use This Logging

### 1. Open Browser DevTools

1. Open `http://localhost:5173` in your browser
2. Press `F12` to open DevTools
3. Go to **Console** tab

### 2. Watch the Initialization Sequence

You should see this sequence on page load:

```
[main.js] Initializing Vue app...
[API] Initialized with BASE_URL: http://localhost:8002
[main.js] Vue app mounted successfully
[App.vue] Component mounted, loading transistors...
[loadTransistors] Starting...
[API Request] GET http://localhost:8002/api/transistors
[API Response] GET /api/transistors: 200 OK
[API Response] Data type: Array[25]
[transistorApi.getAll] Fetching all transistors...
[transistorApi.getAll] Success! Received 25 transistors
[transistorApi.getAll] Sample data: {metadata: {name: "Fuji_...", ...}, ...}
[loadTransistors] API returned: [Object, Object, ...]
[loadTransistors] Data type: object Is Array: true
[loadTransistors] Data length: 25
[loadTransistors] State updated, transistors.value.length: 25
[loadTransistors] Finished, isLoading: false
[App.vue] Transistors loaded, count: 25
[SearchDatabase] Mounted
[SearchDatabase] props.transistors: [Array[25]]
[SearchDatabase] transistors type: object Is Array: true
[SearchDatabase] transistors length: 25
[SearchDatabase] filteredTransistors computing with 25 transistors
[App.vue] Initialization complete
```

### 3. Check for Errors

Look for these patterns:

**Network Errors:**
```javascript
[API Error]: {message: "Network Error", ...}
```
→ Backend is not reachable on port 8002

**CORS Errors:**
```
Access to XMLHttpRequest at 'http://localhost:8002/api/transistors'
from origin 'http://localhost:5173' has been blocked by CORS policy
```
→ Backend CORS configuration issue

**Data Type Errors:**
```javascript
[loadTransistors] Data type: string Is Array: false
```
→ API returned wrong data format (should be Array)

**Component Errors:**
```javascript
[Vue Error Handler]: {component: "SearchDatabase", message: "Cannot read property 'map' of undefined"}
```
→ Component trying to access data that doesn't exist

### 4. Filter Console Logs

Use Chrome DevTools filter to focus on specific areas:

- `[API]` - Show only API-related logs
- `[SearchDatabase]` - Show only SearchDatabase component logs
- `[loadTransistors]` - Show only data loading logs
- `Error` - Show only errors
- `Warn` - Show only warnings

### 5. Network Tab Analysis

Go to **Network** tab in DevTools:

1. Refresh the page (F5)
2. Look for `/api/transistors` request
3. Check:
   - **Status**: Should be `200 OK`
   - **Type**: `xhr` or `fetch`
   - **Size**: Should be ~50-100KB (JSON data)
   - **Time**: Should be <500ms
4. Click on the request to see:
   - **Headers** → Response Headers → Should have `Access-Control-Allow-Origin`
   - **Preview** → Should show array of objects
   - **Response** → Raw JSON data

## Common Issues and Solutions

### Issue 1: No API Requests in Network Tab

**Symptom:**
- No `/api/transistors` request appears
- Console shows: `[loadTransistors] Starting...` but no API logs

**Cause:** JavaScript error before API call

**Solution:**
1. Check Console for red errors
2. Look for `[Vue Error Handler]` messages
3. Fix the error and refresh

### Issue 2: API Request Returns Empty Array

**Symptom:**
```javascript
[transistorApi.getAll] Success! Received 0 transistors
[loadTransistors] Data length: 0
```

**Cause:** Backend database not loaded

**Solution:**
1. Check backend logs: `tail -f /tmp/fastapi-8002.log`
2. Should see: "Transistor XXX generated / loaded successfully!" (25 times)
3. If not, restart backend or check database directory

### Issue 3: CORS Error

**Symptom:**
```
Access-Control-Allow-Origin header is missing
```

**Cause:** Backend CORS not configured for frontend origin

**Solution:**
1. Check `gui_web/backend/main.py` lines 43-52
2. Should include: `"http://localhost:5173"`
3. Restart backend after changing CORS config

### Issue 4: Data Received But Not Displayed

**Symptom:**
```javascript
[loadTransistors] State updated, transistors.value.length: 25
[SearchDatabase] props.transistors: Array[25]
```
But UI shows "0 transistors"

**Cause:** Component rendering issue or template error

**Solution:**
1. Check for Vue warnings in console
2. Look for component errors in `[Vue Error Handler]`
3. Inspect HTML elements in DevTools **Elements** tab
4. Search for `.transistor-card` or `.search-results` in DOM

### Issue 5: Props Not Reaching Child Component

**Symptom:**
```javascript
[App.vue] Transistors loaded, count: 25
[SearchDatabase] props.transistors: undefined
```

**Cause:** Prop binding issue in parent component

**Solution:**
1. Check `App.vue` line 209-215 (SearchDatabase component usage)
2. Verify `:transistors="transistors"` binding exists
3. Check `v-if="currentView === 'search'"` condition

## Defensive Programming Added

### Null/Undefined Checks

All computed properties now check if data exists:

```javascript
if (!props.transistors || !Array.isArray(props.transistors)) {
  console.warn('[SearchDatabase] transistors is not an array:', props.transistors)
  return []
}
```

This prevents crashes when data is loading or missing.

### Error Boundaries

Global error handler catches all Vue errors:

```javascript
app.config.errorHandler = (err, instance, info) => {
  console.error('[Vue Error Handler]:', {...})
}
```

## Testing the Logging

### Quick Test

```bash
# Open browser
google-chrome http://localhost:5173

# Or check from command line
curl -s http://localhost:5173 | grep -o '<title>.*</title>'
```

### Automated Test

```bash
# Run Playwright tests with console logging
uv run pytest tests/test_gui_playwright.py::TestFullWebInterface -v --headed
```

The test will show browser console output in the terminal.

## Next Steps

1. **Refresh your browser** at `http://localhost:5173`
2. **Open DevTools** (F12) → Console tab
3. **Look for the log sequence** shown above
4. **Check Network tab** for `/api/transistors` request
5. **Report back** with:
   - First 20 lines of console logs
   - Any red errors you see
   - Network tab status for `/api/transistors` request

## Summary

All major data flow points now have logging:
- ✅ API initialization
- ✅ API requests/responses
- ✅ Component mounting
- ✅ Data loading
- ✅ Prop changes
- ✅ Computed property calculations
- ✅ Global error handling
- ✅ Defensive null checks

This will help pinpoint exactly where the data flow breaks.
