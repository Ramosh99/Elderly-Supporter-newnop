# Architecture

```mermaid
flowchart TD
    A[Continuous recorded video] --> B[Timestamped sampling]
    B --> C[YOLO bed segmentation]
    B --> D[YOLO pose and target tracking]
    C --> E[Posture, spatial and motion evidence]
    D --> E
    E --> F[Bounded review planner]
    F --> G[Denser context frames]
    G --> H[Chronological pose reclassification]
    H --> I[Optional Gemini image review]
    I --> J[Bracketed corrections and acceptance checks]
    H --> J
    J --> K[Temporal state tracking]
    K --> L[Bed exit and return confirmation]
    K --> M[Activity timeline and durations]
    L --> N[NORMAL / MONITOR / ALERT]
    M --> O[Final JSON]
    N --> O
    E --> P[Candidate annotated MP4]
    O --> Q[Evaluation against reviewed labels]
```

The planner decides when to gather evidence through rules and budgets. Gemini is a
visual reviewer, not an unrestricted controller. Bed and pose networks are pretrained;
no new neural model is trained. Image-space geometry and temporal rules convert their
outputs to activities. Offline analysis can use earlier and later frames; streaming
would require buffering and delayed confirmation.
