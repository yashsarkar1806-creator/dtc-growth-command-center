"""Capture README screenshots of each tab from a running app (python scripts/capture_screenshots.py URL CHROME_PATH)."""
import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
CHROME = sys.argv[2] if len(sys.argv) > 2 else None
TABS = ["Customers", "Channel mix", "Review intelligence", "Campaign brief", "Insights export"]
NAMES = ["01-customers", "02-channel-mix", "03-reviews", "04-campaign-brief", "05-insights"]

with sync_playwright() as p:
    b = p.chromium.launch(executable_path=CHROME, headless=True)
    pg = b.new_page(viewport={"width": 1440, "height": 1500})
    pg.goto(URL)
    pg.wait_for_selector("text=DTC Growth Command Center", timeout=120_000)
    pg.wait_for_timeout(15_000)
    for tab, name in zip(TABS, NAMES):
        pg.get_by_role("tab", name=tab).click()
        pg.wait_for_timeout(6_000)
        if tab == "Campaign brief":
            pg.get_by_role("button", name="Generate AI headline variants").click()
            pg.wait_for_selector("text=/(Hugging Face: HuggingFaceTB|Template fallback)/", timeout=240_000)
            pg.wait_for_timeout(2_000)
        pg.screenshot(path=f"docs/screenshots/{name}.png")
        print("saved", name)
    b.close()
