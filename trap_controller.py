"""
Physical trap actuation via a servo motor (Raspberry Pi), plus a simulated
controller used everywhere else.

Wiring (default, Pi only):
  Servo signal wire -> Raspberry Pi BCM pin 17 (change in config.py)
  Servo power/ground -> external 5V supply is recommended, NOT the Pi's
  5V pin, to avoid brownouts when the servo moves. Share ground with the Pi.
"""

import time
from datetime import datetime, timezone

import config


class TrapController:
    def __init__(self, pin=None):
        from gpiozero import AngularServo

        self.servo = AngularServo(
            pin or config.SERVO_PIN,
            min_angle=0,
            max_angle=180,
            min_pulse_width=0.0005,
            max_pulse_width=0.0025,
        )
        self.state = "closed"
        self.last_activation_utc = None
        self.activation_count = 0
        self.reset()

    def reset(self):
        """Return trap to resting/closed position."""
        self.servo.angle = config.TRAP_CLOSED_ANGLE
        self.state = "closed"

    def activate(self):
        """Trigger the trap mechanism, hold briefly, then reset."""
        print("[trap] ACTIVATING — harmful pest detected")
        self.servo.angle = config.TRAP_OPEN_ANGLE
        self.state = "open"
        self.last_activation_utc = datetime.now(timezone.utc).isoformat()
        self.activation_count += 1
        time.sleep(config.TRAP_ACTIVE_SECONDS)
        self.reset()

    def close(self):
        self.servo.detach()


class SimulatedTrapController:
    """Drop-in replacement for testing away from real hardware.

    Keeps the same observable state (state / last_activation_utc /
    activation_count) so the dashboard can show trap activity.
    """

    def __init__(self):
        self.state = "closed"
        self.last_activation_utc = None
        self.activation_count = 0

    def reset(self):
        self.state = "closed"

    def activate(self):
        print("[trap] ACTIVATING (simulated servo) — harmful pest detected")
        self.state = "open"
        self.last_activation_utc = datetime.now(timezone.utc).isoformat()
        self.activation_count += 1
        time.sleep(config.TRAP_ACTIVE_SECONDS)
        self.reset()

    def close(self):
        pass


def make_trap(simulate: bool = None):
    """Create the right controller for this machine.

    simulate=None -> use real servo on a Raspberry Pi, simulate elsewhere.
    """
    if simulate is None:
        import camera_capture
        simulate = not camera_capture._on_raspberry_pi()

    if not simulate:
        try:
            return TrapController()
        except Exception as e:
            print(f"[trap] Servo/GPIO unavailable ({e}); using simulated trap.")
    return SimulatedTrapController()
