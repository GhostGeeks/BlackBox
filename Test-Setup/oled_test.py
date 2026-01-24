import board
import busio
import adafruit_ssd1306
from PIL import Image, ImageDraw, ImageFont

# Initialize I2C
i2c = busio.I2C(board.SCL, board.SDA)

# Initialize OLED (128x64)
display = adafruit_ssd1306.SSD1306_I2C(128, 64, i2c)

# Clear display
display.fill(0)
display.show()

# Create image buffer
image = Image.new("1", (display.width, display.height))
draw = ImageDraw.Draw(image)

# Draw text
draw.text((0, 0), "OLDE PARK HOTEL", fill=255)
draw.text((0, 16), "OLED ONLINE", fill=255)
draw.text((0, 32), "I2C OK", fill=255)

# Display image
display.image(image)
display.show()
