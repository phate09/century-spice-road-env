"""Century: Spice Road - Interactive Pygame GUI.

Launch with: century-gui  OR  python -m century_env.gui
"""

from century_env.gui.app import SpiceRoadGUI


def main() -> None:
    gui = SpiceRoadGUI()
    gui.run()


if __name__ == "__main__":
    main()
