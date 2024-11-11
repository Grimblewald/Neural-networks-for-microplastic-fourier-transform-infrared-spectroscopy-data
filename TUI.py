import curses
import os
import sys
from functions.DNNModel import base_model

# Initialize the model object
model = base_model()

def get_yaml_files():
    # List all YAML files in the root directory
    return [file for file in os.listdir('.') if file.endswith('.yaml')]

class ScrollableOutput:
    """Redirects output to a scrollable curses window."""
    def __init__(self, window):
        self.window = window
        self.buffer = []
        self.scroll_position = 0  # Track the current scroll position
        self.max_lines = 40  # Arbitrary limit for output history

    def write(self, text):
        """Add each line of text to the buffer and display."""
        for line in text.splitlines():
            self.add_line(line)
        self.display()  # Refresh display after writing new text

    def add_line(self, line):
        # Add line to buffer; truncate if buffer exceeds max_lines
        self.buffer.append(line)
        if len(self.buffer) > self.max_lines:
            self.buffer = self.buffer[1:]

    def display(self):
        """Display the buffer content within the visible area with scroll."""
        self.window.clear()
        # Determine visible slice based on scroll position and window height
        height, _ = self.window.getmaxyx()
        visible_lines = self.buffer[self.scroll_position:self.scroll_position + height]
        for idx, line in enumerate(visible_lines):
            self.window.addstr(idx, 0, line)
        self.window.refresh()

    def scroll_up(self):
        if self.scroll_position > 0:
            self.scroll_position -= 1
            self.display()

    def scroll_down(self):
        if self.scroll_position + self.window.getmaxyx()[0] < len(self.buffer):
            self.scroll_position += 1
            self.display()

    def flush(self):
        """No-op method to match file-like object interface."""
        pass
        
    def clear(self):
        """Clear the output buffer and reset the scroll position."""
        self.buffer.clear()
        self.scroll_position = 0
        self.window.clear()
        self.window.refresh()


def get_user_input(stdscr, y, x):
    """Prompt user input at specific position in the stdscr window."""
    curses.echo()
    stdscr.move(y, x)
    user_input = stdscr.getstr(y, x).decode()
    curses.noecho()
    return user_input

def main(stdscr):
    # Setup curses screen layout
    stdscr.clear()
    curses.curs_set(1)

    # Define window dimensions
    height, width = stdscr.getmaxyx()
    options_width = width // 3  # Left third for options
    output_width = width - options_width  # Remaining space for output

    # Define sub-windows
    options_win = stdscr.subwin(height, options_width, 0, 0)
    output_win = stdscr.subwin(height, output_width, 0, options_width)

    # Redirect stdout to scrollable output window
    output_display = ScrollableOutput(output_win)
    sys.stdout = output_display

    # Step 1: Display YAML files in options window and prompt user to select
    yaml_files = get_yaml_files()
    if not yaml_files:
        output_display.add_line("No YAML configuration files found.")
        output_display.display()
        output_win.getch()
        return
    
    options_win.addstr("Configuration files found:\n")
    for idx, file in enumerate(yaml_files, start=1):
        options_win.addstr(f"  {idx}. {file}\n")
    options_win.addstr("Select file to load: ")
    options_win.refresh()
    
    # Get YAML file selection from user
    while True:
        config_input = get_user_input(stdscr, options_win.getyx()[0], options_width - 10)
        try:
            config_idx = int(config_input) - 1
            if 0 <= config_idx < len(yaml_files):
                config_file = yaml_files[config_idx]
                break
            else:
                output_display.add_line("Invalid selection. Please try again.")
                output_display.display()
        except ValueError:
            output_display.add_line("Please enter a valid number.")
            output_display.display()

    # Step 2: Load and apply the configuration file
    try:
        model.load_config(config_file)
        output_display.add_line(f"Configuration loaded: {config_file}")
        output_display.display()
    except Exception as e:
        output_display.add_line(f"Failed to load configuration: {e}")
        output_display.display()
        output_win.getch()
        return

    # Define available model actions
    options = [
        "Build Datasets",
        "Build Model",
        "Train Model",
        "Evaluate Model",
        "Graph Evaluations",
        "Save Evaluations to LaTeX Tables",
        "Do All Specified Runs",
        "Exit"
    ]

    # Main TUI loop
    while True:
        # Display options on the left
        options_win.clear()
        options_win.addstr("Options:\n")
        for idx, option in enumerate(options, start=1):
            options_win.addstr(f"  {idx}. {option}\n")
        options_win.addstr("Select Option: ")
        options_win.refresh()

        # Get action from user
        action_input = get_user_input(stdscr, options_win.getyx()[0], options_width - 10)
        try:
            action_idx = int(action_input) - 1
            if 0 <= action_idx < len(options):
                action = options[action_idx]

                if action == "Exit":
                    break

                try:
                    match action:
                        case "Build Datasets":
                            output_display.clear()
                            model.build_datasets()
                            output_display.add_line("Datasets built successfully.")
                        case "Build Model":
                            output_display.clear()
                            model.build_model()
                            output_display.add_line("Model built successfully.")
                        case "Train Model":
                            output_display.clear()
                            model.train()
                            output_display.add_line("\nModel trained successfully.")
                        case "Evaluate Model":
                            output_display.clear()
                            model.evaluate()
                            output_display.add_line("Model evaluated successfully.")
                        case "Graph Evaluations":
                            output_display.clear()
                            model.graph_evaluations()
                            output_display.add_line("Graphs generated successfully.")
                        case "Save Evaluations to LaTeX Tables":
                            output_display.clear()
                            model.evaluations_to_latex()
                            output_display.add_line("Saved evaluations to LaTeX tables.")
                        case "Do All Specified Runs":
                            output_display.clear()
                            model.do_runs()
                            output_display.add_line("All runs completed.")
                    output_display.display()

                except Exception as e:
                    output_display.add_line(f"Error: {e}")
                    output_display.display()
            else:
                output_display.add_line("Invalid selection. Please try again.")
                output_display.display()

        except ValueError:
            output_display.add_line("Invalid input. Enter a number.")
            output_display.display()

        # Scrollable output controls
        key = stdscr.getch()
        if key == curses.KEY_UP:
            output_display.scroll_up()
        elif key == curses.KEY_DOWN:
            output_display.scroll_down()

    curses.curs_set(0)  # Hide cursor before exit

# Run the TUI
curses.wrapper(main)
