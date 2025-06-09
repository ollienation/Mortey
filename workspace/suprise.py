"""
A simple Tkinter GUI with a green background that contains a button. Each time the button
is pressed, it changes to a random colour and prints the user's name ("Oliver") to the
console.

Run with:
    python suprise.py
"""

import random
import tkinter as tk
from typing import List


class ColorChangingButtonApp:
    """A GUI application featuring a colour-changing button."""

    # Predefined palette of pleasant colours for the button to cycle through.
    COLOUR_PALETTE: List[str] = [
        "#FF6B6B",  # Soft Red
        "#FFD93D",  # Sunflower Yellow
        "#6BCB77",  # Mint Green
        "#4D96FF",  # Cornflower Blue
        "#9D4EDD",  # Purple
        "#F473B9",  # Pink
        "#FFA15F",  # Orange
        "#00C1D4",  # Teal
    ]

    def __init__(self, user_name: str = "Oliver") -> None:
        self.user_name = user_name

        # Root window configuration
        self.root = tk.Tk()
        self.root.title("Colour-Changing Button")
        self.root.configure(background="#2ECC71")  # Flat green background
        self.root.geometry("300x200")  # Sensible default size
        self.root.resizable(False, False)

        # Initialise button with first colour in the palette
        self.button_colour = random.choice(self.COLOUR_PALETTE)
        self.button = tk.Button(
            self.root,
            text="Click me!",
            command=self.on_button_click,
            bg=self.button_colour,
            fg="white",
            activeforeground="white",
            activebackground=self.darken_colour(self.button_colour),
            font=("Helvetica", 14, "bold"),
            relief=tk.RAISED,
            bd=3,
        )
        # Centre the button in the window
        self.button.pack(expand=True, ipadx=20, ipady=10)

    def run(self) -> None:
        """Start the Tkinter event loop."""
        self.root.mainloop()

    def on_button_click(self) -> None:
        """Handle button click events: change colour and print the user's name."""
        # Choose a new colour different from the current one for variety
        new_colour = self.button_colour
        while new_colour == self.button_colour:
            new_colour = random.choice(self.COLOUR_PALETTE)
        self.button_colour = new_colour

        # Update button colours (normal and active)
        self.button.configure(
            bg=new_colour,
            activebackground=self.darken_colour(new_colour),
        )

        # Print the user's name to the console
        print(self.user_name)

    @staticmethod
    def darken_colour(hex_colour: str, factor: float = 0.85) -> str:
        """Return a slightly darkened variant of an RGB hex colour string."""
        hex_colour = hex_colour.lstrip("#")
        r, g, b = (int(hex_colour[i : i + 2], 16) for i in (0, 2, 4))
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)
        return f"#{r:02X}{g:02X}{b:02X}"


if __name__ == "__main__":
    app = ColorChangingButtonApp(user_name="Oliver")
    app.run()
