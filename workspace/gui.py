"""gui.py
A minimal Tkinter GUI application that demonstrates:
1. A main window with a green background
2. A red button centered in the window
3. Printing a message to the console when the button is clicked

Run this file directly with a Python interpreter (Python 3.x).
"""

import tkinter as tk
from tkinter import ttk


def on_button_click() -> None:
    """Callback triggered when the red button is pressed."""
    print("Button clicked!")


def main() -> None:
    """Create and run the GUI application."""
    # Create the main application window
    root = tk.Tk()
    root.title("Simple Tkinter GUI")

    # Configure the window's background color
    root.configure(bg="green")

    # Ensure the window has a reasonable default size
    root.geometry("300x200")  # Width x Height

    # Create a red button. Use ttk for a modern look, but set style to make it red.
    style = ttk.Style(root)
    style.configure("Red.TButton", foreground="white", background="red")
    # Note: On some platforms ttk may ignore the background option. In that case, use tk.Button instead.

    # Fallback to tk.Button for consistent background color across platforms
    button = tk.Button(
        root,
        text="Click Me",
        bg="red",
        fg="white",
        activebackground="dark red",
        command=on_button_click,
        relief=tk.RAISED,
        padx=20,
        pady=10,
    )

    # Center the button in the window using pack with "expand" and "fill" options
    button.pack(expand=True)

    # Start the Tkinter event loop
    root.mainloop()


if __name__ == "__main__":
    main()
