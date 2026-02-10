"""Century: Spice Road - Interactive Textual TUI.

Launch with: century-tui  OR  python -m century_env.tui
"""

from century_env.tui.app import SpiceRoadApp


def main() -> None:
    app = SpiceRoadApp()
    app.run()


if __name__ == "__main__":
    main()
