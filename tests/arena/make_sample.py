"""Build the original, static arena frame used by the no-game narration check."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    image = Image.new("RGB", (1280, 720), "#11182c")
    draw = ImageDraw.Draw(image)
    title = ImageFont.load_default(size=36)
    body = ImageFont.load_default(size=28)
    small = ImageFont.load_default(size=22)
    draw.text((45, 30), "PLAYER TWO  /  PRACTICE ARENA", font=title, fill="#e8c547")
    draw.text((45, 90), "Ari   HP 80 / 120   SP 40 / 40", font=body, fill="white")
    for x, name, hp, color in [(330, "Practice A", 80, "#9865d9"),
                                (800, "Practice B", 60, "#d98465")]:
        draw.ellipse((x, 200, x + 145, 345), fill=color)
        draw.text((x - 10, 365), f"{name}   HP {hp}", font=body, fill="white")
    draw.rounded_rectangle((890, 425, 1225, 675), radius=12, outline="#4fc3f7", width=3)
    draw.rectangle((905, 440, 1210, 490), fill="#4fc3f7")
    for y, text in [(450, "> Attack"), (510, "Skill"), (565, "Item"), (620, "Guard")]:
        draw.text((925, y), text, font=body, fill="#11182c" if y == 450 else "white")
    draw.rounded_rectangle((45, 490, 845, 675), radius=12, outline="#e8c547", width=3)
    draw.text((70, 515), "YOUR TURN", font=body, fill="#e8c547")
    draw.text((70, 565), "Choose an action. Attack is selected.", font=body, fill="white")
    draw.text((45, 695), "Original static test fixture. Not live gameplay.", font=small, fill="#aab8cf", anchor="ls")
    image.save(Path(__file__).with_name("sample.png"))


if __name__ == "__main__":
    main()
