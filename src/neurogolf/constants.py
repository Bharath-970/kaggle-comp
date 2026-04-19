"""Project-wide constants for ARC tensor and ONNX validation."""

GRID_SIZE = 50
COLOR_CHANNELS = 10
IDENTITY_CHANNELS = 16
STATE_CHANNELS = COLOR_CHANNELS + IDENTITY_CHANNELS
BATCH_DIM = 1
INPUT_SHAPE = (BATCH_DIM, STATE_CHANNELS, GRID_SIZE, GRID_SIZE)

# Competition states 1.44MB max ONNX file size.
MAX_ONNX_FILE_BYTES = 1_440_000

BANNED_ONNX_OPS = frozenset(
    {
        "Loop",
        "Scan",
        "NonZero",
        "Unique",
        "Script",
        "Function",
    }
)
