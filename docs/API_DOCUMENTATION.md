# REST API Documentation

Complete reference for the Transistor Database REST API.

**Base URL**: `http://localhost:8002`

**Interactive Documentation**:
- Swagger UI: http://localhost:8002/docs
- ReDoc: http://localhost:8002/redoc

## Table of Contents

- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [Transistor CRUD](#transistor-crud)
  - [Export Endpoints](#export-endpoints)
  - [Comparison Endpoints](#comparison-endpoints)
  - [Validation Endpoints](#validation-endpoints)
  - [Plot Data Endpoints](#plot-data-endpoints)
- [Data Models](#data-models)
- [Error Handling](#error-handling)
- [Code Examples](#code-examples)
- [Rate Limits](#rate-limits)

---

## Authentication

Currently **no authentication** required. API is designed for local development use.

For production deployments, consider adding:
- API key authentication
- JWT tokens
- OAuth2

---

## Endpoints

### Transistor CRUD

#### List All Transistors

```http
GET /api/transistors
```

**Response**: `200 OK`

```json
[
  {
    "name": "CREE_C3M0060065J",
    "type": "SiC-MOSFET",
    "manufacturer": "CREE",
    "v_abs_max": 650,
    "i_abs_max": 120,
    "i_cont": 60,
    "housing": "TO-247-3"
  },
  ...
]
```

**Query Parameters**:
- `type` (optional): Filter by type ("MOSFET", "SiC-MOSFET", "IGBT", "GaN")
- `manufacturer` (optional): Filter by manufacturer
- `v_min` (optional): Minimum voltage rating
- `v_max` (optional): Maximum voltage rating
- `i_min` (optional): Minimum current rating
- `i_max` (optional): Maximum current rating

**Example**:
```bash
curl "http://localhost:8002/api/transistors?type=SiC-MOSFET&v_min=600"
```

---

#### Get Single Transistor

```http
GET /api/transistors/{name}
```

**Parameters**:
- `name` (path): Transistor name (URL-encoded)

**Response**: `200 OK`

```json
{
  "metadata": {
    "name": "CREE_C3M0060065J",
    "type": "SiC-MOSFET",
    "manufacturer": "CREE",
    "housing": "TO-247-3",
    "datasheet_hyperlink": "https://..."
  },
  "electrical_ratings": {
    "v_abs_max": 650,
    "i_abs_max": 120,
    "i_cont": 60,
    "t_j_max": 175
  },
  "thermal_properties": {
    "r_th_junction_case": 0.24,
    "r_th_case_heatsink": 0.0,
    "housing_area": 0.00054,
    "cooling_area": 0.00054
  },
  "switch": {
    "channel_data": [...],
    "e_on_data": [...],
    "e_off_data": [...],
    "gate_charge_curves": [...]
  },
  "diode": {
    "channel_data": [...],
    "e_rr_data": [...]
  },
  "c_oss": [...],
  "c_iss": [...],
  "c_rss": [...]
}
```

**Errors**:
- `404 Not Found`: Transistor not found

**Example**:
```bash
curl http://localhost:8002/api/transistors/CREE_C3M0060065J
```

---

#### Create Transistor

```http
POST /api/transistors
```

**Request Body**:

```json
{
  "name": "Custom_Transistor_001",
  "type": "MOSFET",
  "manufacturer": "Generic",
  "v_abs_max": 600,
  "i_abs_max": 100,
  "i_cont": 50,
  "t_j_max": 150,
  "r_th_junction_case": 0.5,
  "housing": "TO-220"
}
```

**Minimum Required Fields**:
- `name` (string, unique)
- `type` (enum: "MOSFET", "SiC-MOSFET", "IGBT", "GaN")
- `manufacturer` (string)
- `v_abs_max` (number, > 0)
- `i_abs_max` (number, > 0)

**Optional Fields**:
- `i_cont` (number)
- `t_j_max` (number)
- `r_th_junction_case` (number)
- `housing` (string)
- `datasheet_hyperlink` (string, URL)
- `switch` (object)
- `diode` (object)
- `c_oss`, `c_iss`, `c_rss` (arrays)

**Response**: `201 Created`

```json
{
  "message": "Transistor created successfully",
  "name": "Custom_Transistor_001"
}
```

**Errors**:
- `400 Bad Request`: Validation error
- `409 Conflict`: Transistor with this name already exists

**Example**:
```bash
curl -X POST http://localhost:8002/api/transistors \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test_MOSFET",
    "type": "MOSFET",
    "manufacturer": "Test Corp",
    "v_abs_max": 600,
    "i_abs_max": 100
  }'
```

---

#### Update Transistor

```http
PUT /api/transistors/{name}
```

**Parameters**:
- `name` (path): Transistor name to update

**Request Body**: Same as Create, but `name` field is ignored (use path parameter)

**Response**: `200 OK`

```json
{
  "message": "Transistor updated successfully",
  "name": "Custom_Transistor_001"
}
```

**Errors**:
- `404 Not Found`: Transistor not found
- `400 Bad Request`: Validation error

**Example**:
```bash
curl -X PUT http://localhost:8002/api/transistors/Test_MOSFET \
  -H "Content-Type: application/json" \
  -d '{
    "type": "MOSFET",
    "manufacturer": "Test Corp",
    "v_abs_max": 650,
    "i_abs_max": 120
  }'
```

---

#### Delete Transistor

```http
DELETE /api/transistors/{name}
```

**Parameters**:
- `name` (path): Transistor name to delete

**Response**: `200 OK`

```json
{
  "message": "Transistor deleted successfully",
  "name": "Custom_Transistor_001"
}
```

**Errors**:
- `404 Not Found`: Transistor not found

**Example**:
```bash
curl -X DELETE http://localhost:8002/api/transistors/Test_MOSFET
```

---

### Export Endpoints

#### Export to JSON

```http
GET /api/export/{name}/json
```

**Response**: `200 OK` (application/json)

Returns complete transistor data in JSON format.

**Example**:
```bash
curl http://localhost:8002/api/export/CREE_C3M0060065J/json > transistor.json
```

---

#### Export to MATLAB

```http
GET /api/export/{name}/matlab
```

**Response**: `200 OK` (text/plain)

Returns MATLAB script (.m file content) with transistor data as variables.

**Example**:
```bash
curl http://localhost:8002/api/export/CREE_C3M0060065J/matlab > transistor.m
```

**Generated MATLAB Variables**:
```matlab
% Transistor: CREE_C3M0060065J
v_abs_max = 650;
i_abs_max = 120;
i_cont = 60;
r_ds_on = 0.006;
r_th_jc = 0.24;
% ... and more
```

---

#### Export to PLECS

```http
GET /api/export/{name}/plecs
```

**Response**: `200 OK` (application/xml)

Returns PLECS XML thermal library file.

**Example**:
```bash
curl http://localhost:8002/api/export/CREE_C3M0060065J/plecs > transistor_plecs.xml
```

**Usage**: Import XML file into PLECS → Thermal Library → Import

---

#### Export to Simulink

```http
GET /api/export/{name}/simulink
```

**Response**: `200 OK` (application/octet-stream)

Returns .mat file for Simulink/Simscape.

**Example**:
```bash
curl http://localhost:8002/api/export/CREE_C3M0060065J/simulink > transistor.mat
```

---

#### Export to GeckoCIRCUITS

```http
GET /api/export/{name}/gecko
```

**Response**: `200 OK` (application/xml)

Returns GeckoCIRCUITS .ipes file.

**Example**:
```bash
curl http://localhost:8002/api/export/CREE_C3M0060065J/gecko > transistor.ipes
```

---

#### Export Virtual Datasheet (PDF)

```http
GET /api/export/{name}/datasheet
```

**Response**: `200 OK` (application/pdf)

Returns virtual datasheet PDF with plots and specifications.

**Example**:
```bash
curl http://localhost:8002/api/export/CREE_C3M0060065J/datasheet > datasheet.pdf
```

---

### Comparison Endpoints

#### Compare Transistors

```http
POST /api/compare
```

**Request Body**:

```json
{
  "transistors": [
    "CREE_C3M0060065J",
    "Infineon_IPT60R028G7",
    "Infineon_FF300R12KE3"
  ]
}
```

**Response**: `200 OK`

```json
{
  "comparison": {
    "electrical": {
      "CREE_C3M0060065J": {"v_abs_max": 650, "i_abs_max": 120, "i_cont": 60},
      "Infineon_IPT60R028G7": {"v_abs_max": 600, "i_abs_max": 90, "i_cont": 45},
      "Infineon_FF300R12KE3": {"v_abs_max": 1200, "i_abs_max": 300, "i_cont": 300}
    },
    "thermal": {
      "CREE_C3M0060065J": {"r_th_junction_case": 0.24},
      "Infineon_IPT60R028G7": {"r_th_junction_case": 0.30},
      "Infineon_FF300R12KE3": {"r_th_junction_case": 0.13}
    },
    "switching": {...},
    "best_in_category": {
      "lowest_r_th": "Infineon_FF300R12KE3",
      "highest_voltage": "Infineon_FF300R12KE3",
      "highest_current": "Infineon_FF300R12KE3"
    }
  }
}
```

**Limits**:
- Minimum 2 transistors
- Maximum 3 transistors

**Errors**:
- `400 Bad Request`: Invalid number of transistors
- `404 Not Found`: One or more transistors not found

---

### Validation Endpoints

#### Validate Transistor Data

```http
POST /api/validate
```

**Request Body**: Same as Create Transistor

**Response**: `200 OK`

```json
{
  "valid": true,
  "errors": [],
  "warnings": [
    "Missing thermal resistance data"
  ]
}
```

**Invalid Response**: `200 OK` (validation failures return 200, not 400)

```json
{
  "valid": false,
  "errors": [
    "v_abs_max must be positive",
    "name is required"
  ],
  "warnings": []
}
```

---

### Plot Data Endpoints

#### Get Channel Characteristics Plot Data

```http
GET /api/plots/{name}/channel
```

**Query Parameters**:
- `device` (optional): "switch" or "diode" (default: "switch")

**Response**: `200 OK`

```json
{
  "curves": [
    {
      "v_g": 15,
      "v_ds": [0, 0.5, 1.0, 1.5, 2.0],
      "i_d": [0, 10, 20, 30, 35],
      "t_j": 25
    },
    {
      "v_g": 15,
      "v_ds": [0, 0.5, 1.0, 1.5, 2.0],
      "i_d": [0, 8, 16, 24, 28],
      "t_j": 150
    }
  ]
}
```

---

#### Get Switching Loss Plot Data

```http
GET /api/plots/{name}/switching
```

**Query Parameters**:
- `type` (required): "e_on", "e_off", or "e_rr"

**Response**: `200 OK`

```json
{
  "curves": [
    {
      "i_x": [10, 20, 30, 40, 50],
      "e_x": [0.5, 1.2, 2.1, 3.2, 4.5],
      "v_supply": 400,
      "v_g": 15,
      "r_g": 10,
      "t_j": 25
    }
  ]
}
```

---

#### Get Capacitance Plot Data

```http
GET /api/plots/{name}/capacitance
```

**Query Parameters**:
- `type` (required): "c_oss", "c_iss", or "c_rss"

**Response**: `200 OK`

```json
{
  "v_ds": [0, 10, 25, 50, 100, 200, 400, 600],
  "capacitance": [8500, 3200, 1800, 1200, 800, 500, 350, 280]
}
```

---

## Data Models

### TransistorMetadata

```typescript
{
  name: string;              // Unique identifier
  type: "MOSFET" | "SiC-MOSFET" | "IGBT" | "GaN";
  manufacturer: string;
  housing: string;           // e.g., "TO-247-3", "TO-220"
  datasheet_hyperlink?: string;
  comment?: string;
}
```

### ElectricalRatings

```typescript
{
  v_abs_max: number;         // Max drain-source voltage [V]
  i_abs_max: number;         // Max drain current [A]
  i_cont: number;            // Continuous drain current [A]
  t_j_max: number;           // Max junction temperature [°C]
}
```

### ThermalProperties

```typescript
{
  r_th_junction_case: number;      // Thermal resistance J-C [K/W]
  r_th_case_heatsink?: number;     // Thermal resistance C-H [K/W]
  r_th_heatsink_ambient?: number;  // Thermal resistance H-A [K/W]
  housing_area?: number;           // Housing surface area [m²]
  cooling_area?: number;           // Effective cooling area [m²]
}
```

### ChannelCharacteristics

```typescript
{
  v_g: number;               // Gate voltage [V]
  v_ds: number[];            // Drain-source voltage array [V]
  i_d: number[];             // Drain current array [A]
  t_j: number;               // Junction temperature [°C]
}
```

### SwitchingLossData

```typescript
{
  dataset_type: "graph_i_e" | "graph_r_e";
  i_x: number[];             // Current array [A]
  e_x: number[];             // Energy loss array [J]
  v_supply: number;          // Supply voltage [V]
  v_g: number;               // Gate voltage [V]
  r_g: number;               // Gate resistance [Ω]
  t_j: number;               // Junction temperature [°C]
}
```

### VoltageDependentCapacitance

```typescript
{
  v_ds: number[];            // Voltage array [V]
  capacitance: number[];     // Capacitance array [F]
  t_j?: number;              // Temperature [°C]
}
```

### GateChargeCurve

```typescript
{
  v_supply: number;          // Supply voltage [V]
  i_d: number;               // Drain current [A]
  q_g: number[];             // Gate charge array [C]
  v_gs: number[];            // Gate-source voltage array [V]
  t_j: number;               // Temperature [°C]
}
```

---

## Error Handling

All errors return JSON with consistent format:

```json
{
  "detail": "Error message here"
}
```

### HTTP Status Codes

| Code | Meaning | Usage |
|------|---------|-------|
| 200 | OK | Successful GET, PUT, DELETE |
| 201 | Created | Successful POST |
| 400 | Bad Request | Validation error, malformed JSON |
| 404 | Not Found | Transistor not found |
| 409 | Conflict | Duplicate name on create |
| 422 | Unprocessable Entity | Invalid data type |
| 500 | Internal Server Error | Server-side exception |

### Error Response Examples

**404 Not Found:**
```json
{
  "detail": "Transistor 'Unknown_Transistor' not found"
}
```

**400 Bad Request:**
```json
{
  "detail": "Validation error: v_abs_max must be positive"
}
```

**409 Conflict:**
```json
{
  "detail": "Transistor 'CREE_C3M0060065J' already exists"
}
```

---

## Code Examples

### Python (requests)

```python
import requests

BASE_URL = "http://localhost:8002"

# List all transistors
response = requests.get(f"{BASE_URL}/api/transistors")
transistors = response.json()
print(f"Found {len(transistors)} transistors")

# Get specific transistor
name = "CREE_C3M0060065J"
response = requests.get(f"{BASE_URL}/api/transistors/{name}")
transistor = response.json()
print(f"Voltage rating: {transistor['electrical_ratings']['v_abs_max']}V")

# Create new transistor
new_transistor = {
    "name": "Custom_MOSFET",
    "type": "MOSFET",
    "manufacturer": "Custom",
    "v_abs_max": 600,
    "i_abs_max": 100
}
response = requests.post(
    f"{BASE_URL}/api/transistors",
    json=new_transistor
)
print(f"Created: {response.json()}")

# Export to PLECS
response = requests.get(f"{BASE_URL}/api/export/{name}/plecs")
with open("transistor.xml", "w") as f:
    f.write(response.text)
print("Exported to PLECS")

# Compare transistors
comparison_request = {
    "transistors": [
        "CREE_C3M0060065J",
        "Infineon_IPT60R028G7"
    ]
}
response = requests.post(
    f"{BASE_URL}/api/compare",
    json=comparison_request
)
comparison = response.json()
print(f"Best thermal: {comparison['comparison']['best_in_category']['lowest_r_th']}")
```

### JavaScript (fetch)

```javascript
const BASE_URL = 'http://localhost:8002';

// List all transistors
async function listTransistors() {
  const response = await fetch(`${BASE_URL}/api/transistors`);
  const transistors = await response.json();
  console.log(`Found ${transistors.length} transistors`);
  return transistors;
}

// Get specific transistor
async function getTransistor(name) {
  const response = await fetch(`${BASE_URL}/api/transistors/${name}`);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  }
  return await response.json();
}

// Create transistor
async function createTransistor(data) {
  const response = await fetch(`${BASE_URL}/api/transistors`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  return await response.json();
}

// Export to JSON
async function exportToJSON(name) {
  const response = await fetch(`${BASE_URL}/api/export/${name}/json`);
  const blob = await response.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${name}.json`;
  a.click();
}

// Usage
(async () => {
  const transistors = await listTransistors();
  const cree = await getTransistor('CREE_C3M0060065J');
  console.log('Voltage:', cree.electrical_ratings.v_abs_max);

  await createTransistor({
    name: 'Test_MOSFET',
    type: 'MOSFET',
    manufacturer: 'Test',
    v_abs_max: 600,
    i_abs_max: 100
  });
})();
```

### cURL

```bash
# List all transistors
curl http://localhost:8002/api/transistors

# Filter SiC MOSFETs with V > 1000V
curl "http://localhost:8002/api/transistors?type=SiC-MOSFET&v_min=1000"

# Get specific transistor
curl http://localhost:8002/api/transistors/CREE_C3M0060065J

# Create transistor
curl -X POST http://localhost:8002/api/transistors \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test_MOSFET",
    "type": "MOSFET",
    "manufacturer": "Test Corp",
    "v_abs_max": 600,
    "i_abs_max": 100
  }'

# Update transistor
curl -X PUT http://localhost:8002/api/transistors/Test_MOSFET \
  -H "Content-Type: application/json" \
  -d '{
    "type": "MOSFET",
    "manufacturer": "Test Corp",
    "v_abs_max": 650,
    "i_abs_max": 120
  }'

# Delete transistor
curl -X DELETE http://localhost:8002/api/transistors/Test_MOSFET

# Export to PLECS
curl http://localhost:8002/api/export/CREE_C3M0060065J/plecs > output.xml

# Export to JSON
curl http://localhost:8002/api/export/CREE_C3M0060065J/json > output.json

# Compare transistors
curl -X POST http://localhost:8002/api/compare \
  -H "Content-Type: application/json" \
  -d '{
    "transistors": ["CREE_C3M0060065J", "Infineon_IPT60R028G7"]
  }'

# Validate transistor data
curl -X POST http://localhost:8002/api/validate \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test",
    "type": "MOSFET",
    "manufacturer": "Test",
    "v_abs_max": -100,
    "i_abs_max": 50
  }'
```

---

## Rate Limits

Currently **no rate limits** implemented.

For production use, consider:
- 100 requests per minute per IP
- 1000 requests per hour per IP
- Separate limits for export endpoints (more expensive)

---

## CORS Configuration

The API has CORS enabled for:
- `http://localhost:5173` (Vue dev server)
- `http://localhost:3000`
- `http://127.0.0.1:5173`

For production, update CORS origins in `gui_web/backend/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Performance Considerations

### Caching

Consider implementing caching for:
- GET `/api/transistors` (list endpoint)
- Export endpoints (computationally expensive)

### Pagination

For large databases, implement pagination:

```http
GET /api/transistors?page=1&per_page=50
```

### Async Processing

For long-running export operations, consider:
1. Return task ID immediately
2. Client polls for completion
3. Download when ready

---

## Testing the API

### Automated Testing

```bash
# Run API tests
pytest tests/test_rest_api.py -v

# Run specific test
pytest tests/test_rest_api.py::test_create_transistor -v
```

### Manual Testing with Swagger UI

1. Start backend: `uvicorn transistordatabase.gui_web.backend.main:app --port 8002`
2. Open: http://localhost:8002/docs
3. Click "Try it out" on any endpoint
4. Fill in parameters
5. Click "Execute"
6. View response

### Testing with Postman

1. Import collection from `docs/postman_collection.json` (if available)
2. Or create requests manually
3. Set base URL: `http://localhost:8002`
4. Test CRUD operations

---

## Versioning

Current API version: **v1**

Future versions will be prefixed:
- `/api/v1/transistors`
- `/api/v2/transistors`

---

## Support

- **API Issues**: https://github.com/tinix84/transistordatabase/issues
- **Interactive Docs**: http://localhost:8002/docs
- **User Guide**: [USER_GUIDE.md](USER_GUIDE.md)

---

## Changelog

### v0.6.0 (Current)
- Initial REST API implementation
- CRUD operations for transistors
- Export endpoints (all formats)
- Comparison endpoint
- Validation endpoint
- Plot data endpoints

---

**That's everything you need to know about the Transistor Database REST API!** 🚀

For Python library usage (without REST API), see the main [User Guide](USER_GUIDE.md).
