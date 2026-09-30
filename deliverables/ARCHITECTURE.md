# Architecture

```mermaid
flowchart TD
    A[Continuous recorded video] --> B[Timestamped sampling at 3 FPS]
    B --> C[YOLO bed segmentation]
    B --> D[YOLO pose + ByteTrack identity locking]
    C --> E[Posture / spatial / motion evidence per frame]
    D --> E
    E --> F[Iterative review agent\nplan initial windows]
    F --> G[Gather denser frames at 9 FPS]
    G --> H[Assess finding\ntransition? uncertainty? context clear?]
    H -->|transition or uncertainty detected| F
    H --> I[Chronological pose reclassification]
    I --> J[Optional Gemini VLM review\nup to 6 requests × 5 frames]
    J --> K[Bracketed corrections + fragmentation guard]
    I --> K
    K --> L[Temporal state tracking\nhold periods + gap detection]
    L --> M[Bed exit / return event state machine]
    L --> N[Activity timeline and durations]
    M --> O[NORMAL / MONITOR / ALERT decision]
    N --> P[Final JSON output]
    O --> P
    E --> Q[Candidate annotated MP4\nfirst-pass states only]
    P --> R[Evaluation against reviewed ground-truth labels]
```

The review agent uses an iterative reasoning loop: findings from each dense-sampling
window (transition detected, upright near bed, uncertainty remains) trigger backward
or forward follow-up windows before committing. Gemini is a visual reviewer with strict
acceptance gates, not an unrestricted controller. Bed and pose networks are pretrained;
no new neural model is trained. Offline analysis can use earlier and later frames;
streaming would require buffering and delayed confirmation.
