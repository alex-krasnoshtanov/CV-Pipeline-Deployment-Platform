# Frontend Architecture

This document provides a brief overview of the frontend architecture. 

## Stack Overview
- **Framework:** Next.js 16 (React 19)
- **Role:** NPEC researcher UI for uploading images, reviewing segmentation masks, and flagging predictions as good or bad.

## Data Flow
The frontend operates as a thin client that uploads Petri dish images to the `FastAPI` backend endpoint (`POST /infer`). It visualizes the returned `InferenceResult` object, which contains a base64-encoded PNG mask and a list of landmark coordinates.

Researchers can then submit feedback for these predictions, which is stored via the backend to build a dataset for the retraining loop.
