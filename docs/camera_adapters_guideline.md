# Camera adapters guideline

## Goal

Each manufacturer must convert its proprietary format into
`CameraEvent`, without adding manufacturer-specific rules to the
pipeline.

## Mandatory contract

Every adapter must inherit from `CameraAdapter` and implement:

- `can_handle(content_type, body)`
- `normalize(package, body)`

## Adapter responsibilities

- recognize only packages from its own manufacturer;
- extract metadata, image, and bounding boxes;
- convert coordinates to pixels;
- save the original and annotated image;
- return a `CameraEvent`;
- preserve manufacturer-specific information in `extra_data`.

## Pipeline responsibilities

- pick the adapter via the factory;
- ignore events without an image or without valid boxes;
- track people;
- send events to the panel;
- publish normalized events to Kafka.

## Implementation rules

- use type hints;
- use clear names;
- avoid manufacturer-specific logic inside the pipeline;
- don't register adapters outside the factory;
- don't change the schema without versioning;
- test direct JSON and multipart packages;
- keep compatibility with existing routes;
- log enough context;
- don't store passwords or credentials in the code.

## Checklist for a new manufacturer

1. Create `<manufacturer>.py`.
2. Implement `CameraAdapter`.
3. Register it in `CameraAdapterFactory`.
4. Test recognition.
5. Test normalization without an image.
6. Test normalization with an image.
7. Validate the bounding box in pixels.
8. Validate execution in the pipeline.
9. Validate publishing to Kafka.
