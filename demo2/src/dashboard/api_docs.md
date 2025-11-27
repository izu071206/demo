# API Documentation

## Base URL
`http://localhost:5000/api`

## Authentication
Currently no authentication required. For production, add authentication middleware.

## Endpoints

### 1. Status & Health

#### GET `/api/status`
Check if inference pipeline is ready.

**Response:**
```json
{
  "status": "ready",
  "model": "RandomForest (Obfuscation Detector)"
}
```

#### GET `/api/health`
Comprehensive health check.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-27T21:00:00",
  "pipeline": "ready",
  "models_available": 2
}
```

### 2. Prediction

#### POST `/api/predict`
Predict obfuscation for a single file.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (binary file)

**Response:**
```json
{
  "id": 1234567890,
  "filename": "sample.exe",
  "is_obfuscated": true,
  "prediction": "Obfuscated",
  "confidence": 0.95,
  "probabilities": {
    "benign": 0.05,
    "obfuscated": 0.95
  },
  "model": "RandomForest (Obfuscation Detector)",
  "feature_count": 287,
  "processing_time": 1.234,
  "top_contributors": [
    {"feature": "api_calls_0", "impact": 0.1234},
    ...
  ]
}
```

#### POST `/api/predict/batch`
Batch prediction for multiple files.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `files[]` (array of binary files)

**Response:**
```json
{
  "results": [
    {
      "filename": "file1.exe",
      "is_obfuscated": true,
      "prediction": "Obfuscated",
      ...
    },
    ...
  ],
  "count": 2
}
```

### 3. Models & Configuration

#### GET `/api/models`
Get list of available models.

**Response:**
```json
{
  "models": [
    {
      "name": "random_forest",
      "type": "random_forest",
      "path": "models/random_forest_model.pkl",
      "size": 1024000
    }
  ],
  "current": "RandomForest (Obfuscation Detector)"
}
```

#### GET `/api/config`
Get current inference configuration.

**Response:**
```json
{
  "inference_config": {
    "model_type": "random_forest",
    "model_path": "models/random_forest_model.pkl",
    ...
  },
  "pipeline_ready": true,
  "feature_dim": 287
}
```

#### POST `/api/config`
Update configuration (requires restart).

### 4. Results & Metrics

#### GET `/api/results`
Get evaluation results for all models.

**Response:**
```json
{
  "random_forest": {
    "accuracy": 0.95,
    "precision": 0.92,
    "recall": 0.88,
    "f1_score": 0.90,
    ...
  }
}
```

#### GET `/api/metrics`
Get detailed metrics with chart availability.

**Response:**
```json
{
  "random_forest": {
    "accuracy": 0.95,
    ...
    "charts": {
      "confusion_matrix": true,
      "roc_curve": true
    }
  }
}
```

### 5. Features

#### GET `/api/features/info`
Get feature extraction information.

**Response:**
```json
{
  "feature_dim": 287,
  "opcode_ngrams": [2, 3, 4],
  "opcode_max_features": 1000,
  "api_max_features": 500,
  "enable_cfg": true
}
```

### 6. History & Statistics

#### GET `/api/history`
Get prediction history.

**Query Parameters:**
- `limit` (optional): Number of records to return (default: 50)

**Response:**
```json
{
  "history": [
    {
      "id": 1234567890,
      "filename": "sample.exe",
      "is_obfuscated": true,
      "timestamp": "2025-11-27T21:00:00",
      ...
    }
  ],
  "count": 10
}
```

#### GET `/api/history/<prediction_id>`
Get detailed information about a specific prediction.

#### DELETE `/api/history`
Clear prediction history.

#### GET `/api/stats`
Get dashboard statistics.

**Response:**
```json
{
  "total_predictions": 100,
  "obfuscated_count": 45,
  "benign_count": 55,
  "errors": 2,
  "start_time": "2025-11-27T20:00:00",
  "model_info": {
    "type": "random_forest",
    "name": "RandomForest (Obfuscation Detector)",
    "loaded": true
  },
  "recent_predictions": 100
}
```

### 7. Charts & Visualizations

#### GET `/api/charts/confusion_matrix/<model_name>`
Get confusion matrix image for a model.

#### GET `/api/charts/roc_curve/<model_name>`
Get ROC curve image for a model.

## Error Responses

All errors follow this format:
```json
{
  "error": "Error type",
  "message": "Detailed error message"
}
```

**HTTP Status Codes:**
- `200`: Success
- `400`: Bad Request
- `404`: Not Found
- `500`: Internal Server Error
- `503`: Service Unavailable

## Rate Limiting

Currently no rate limiting. For production, implement rate limiting middleware.

## CORS

CORS is enabled for all origins. For production, restrict to specific origins.

