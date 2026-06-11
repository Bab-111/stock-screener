from datetime import datetime
import os

# Make sure output folder exists
os.makedirs("output", exist_ok=True)

# Create simple HTML
html = f"""
<!DOCTYPE html>
<html>
<head><title>Test Screener</title></head>
<body>
  <h1>Hello Babak!</h1>
  <p>This is a test run at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.</p>
</body>
</html>
"""

# Save HTML into output/index.html
with open("output/index.html", "w") as f:
    f.write(html)
