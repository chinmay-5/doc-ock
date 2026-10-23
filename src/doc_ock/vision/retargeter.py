
class Retargeter:
    def __init__(
        self,
        xy_scale: float = 0.30,
        z_scale: float = 0.20,
    ):
        self.xy_scale = xy_scale
        self.z_scale = z_scale
        self.anchor = None

    def reset(self):
        """Forget the reference hand position."""
        self.anchor = None

    def update(self, landmarks):
        """Return relative (x, y, z) offsets in meters.

        The first observation establishes the anchor and returns zero.
        """
        if landmarks is None:
            self.reset()
            return None

        wrist = landmarks[0]

        if self.anchor is None:
            self.anchor = wrist
            return (0.0, 0.0, 0.0)

        dx = wrist[0] - self.anchor[0]
        dy = wrist[1] - self.anchor[1]
        dz = wrist[2] - self.anchor[2]

        return (
            dx * self.xy_scale,
            -dy * self.xy_scale,
            -dz * self.z_scale,
        )
