"""Century: Spice Road - Interactive Textual TUI.

Launch with: century-tui  OR  python -m century_env.tui
"""


def main() -> None:
    from century_env.tui.app import SpiceRoadApp

    app = SpiceRoadApp()
    app.run()


if __name__ == "__main__":
    main()
