- Main source code:
    + Write as class:
        + Remember to commend input output, and the purpose of the class
    + OCR, OD, CD:
        + Each file is a video:
        + Json {
            "video_name": "video1.mp4",
            "frames": [
                {
                    "frame_number": 1,
                    "objects": [
                        {
                            "class": "car",
                            "bbox": [x_min, y_min, x_max, y_max],
                            "confidence": 0.95
                        },
                        ...
                    ]
                },
                ...
            ]
        }
- API
- Documentation:
    + Write a README.md file
    + Random bullsh*t go


# DRES API Usage Guide (v2)

This guide explains how to interact with the DRES (Distributed Retrieval Evaluation Server) **v2 API** to:

- Authenticate (`/login`)
- List evaluations (`/client/evaluation/list`)
- Submit answers (`/submit/{evaluationId}`)

Useful for automated benchmarking tools, submission clients, or your own Python scripts.
---

## 🛠️ Base URL

All requests go through DRES instance:
https://api.siu.edu.vn/siu_chillout_1
---

## 🔐 1. Login

**Endpoint:**
https://api.siu.edu.vn/siu_chillout_1/api/v2/login
type: POST

**Request Body (JSON):**

```json
{
  "username": "your_username",
  "password": "your_password"
}
```
**Response:**
```json
{
  "sessionId": "abcdef1234567890"
}
```
This sessionId is required for all subsequent requests as a session query parameter

## 📋 2. List Evaluations
**Endpoint:**
https://api.siu.edu.vn/siu_chillout_1/api/v2/client/evaluation/list?session=<sessionId>
type: GET

**Response:**
```json
[
  {
    "id": "evaluation-id-123",
    "name": "My Evaluation",
    "status": "ACTIVE",
    "startTime": 1690000000,
    "endTime": null
  }
]

```
- Look for "status": "ACTIVE" to find the currently running evaluation.

## 📝 3. Submit Answers
**Endpoint:**
https://api.siu.edu.vn/siu_chillout_1/api/v2/submit/<evaluationId>?session=<sessionId>
type: POST
**Request Body (JSON):**
```json
{
  "answerSets": [
    {
      "answers": [
        {
          "mediaItemName": "video_name",
          "start": start_time_in_miliseconds,
          "end": end_time_in_miliseconds
        }
      ]
    }
  ]
}

```
- Start and end times can the same if you want to select a single frame.
- Example:
```json
{
  "answerSets": [
    {
      "answers": [
        {
          "mediaItemName": "L01_V001",
          "start": 1000,
          "end": 1000
        }
      ]
    }
  ]
}
```
**Response:**
```json
{
  "status": true,
  "message": "Submission successful"
}
```
