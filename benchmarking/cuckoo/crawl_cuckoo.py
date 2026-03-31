#!/usr/bin/env python3
"""
Crawl the results from the Cuckoo Meetings page and save the transcript and translation to TXT files.

Usage:
    # First time: save the login session
    python crawl_cuckoo.py --save-session

    # Crawl the specified meeting page
    python crawl_cuckoo.py <meeting_url> <output_stem>

    # Example:
    python crawl_cuckoo.py https://app.cuckoo.so/meetings/abc123 test

Output:
    results/<output_stem>_transcript.txt   ← The transcript (one segment per line)
    results/<output_stem>_translation.txt  ← The translation (one segment per line)

Prerequisites:
    pip install playwright
    playwright install chromium
"""
import os
import sys
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

# ─────────────────────────────────────────────────────────────────────────────
CUCKOO_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SESSION_FILE = CUCKOO_DIR / "crawl/cuckoo_session.json"
OUTPUT_DIR   = CUCKOO_DIR / "crawl"
OUTPUT_DIR.mkdir(exist_ok=True)

# DOM selectors
#
#   div  ← The speech segment (SEL_SEGMENT)
#     └─ div.space-y-1
#          ├─ p.text-sm > span          ← SEL_ORIGINAL
#          └─ div > p.text-lg > span    ← SEL_TRANSLATION
#
SEL_SEGMENT     = "div.space-y-1"
SEL_ORIGINAL    = "p.text-sm > span"
SEL_TRANSLATION = "p.text-lg > span"
# ─────────────────────────────────────────────────────────────────────────────


async def save_session():
    """Once: save the session cookie after manual login, so you don't need to login again."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page    = await context.new_page()

        await page.goto("https://app.cuckoo.so/")
        print("Please login in the browser, then enter any page, and then press Enter in the terminal...")
        input()

        await context.storage_state(path=str(SESSION_FILE))
        await browser.close()
        print(f"✅ Session saved to {SESSION_FILE}")


async def crawl(meeting_url: str, output_stem: str):
    """Load the existing session, and crawl the transcript and translation from the specified meeting page."""
    if not SESSION_FILE.exists():
        print("Session file not found, please login...")
        await save_session()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            storage_state=str(SESSION_FILE)   # Load the existing session
        )
        page = await context.new_page()

        print(f"Opening page: {meeting_url}")
        await page.goto(meeting_url)
        await page.wait_for_timeout(800) # wait for contents to be loaded

        # Wait for the subtitle block to load
        print("Waiting for subtitle block to load...")

        try:
            await page.wait_for_selector(SEL_SEGMENT, timeout=30_000)
        except Exception:
            print("❌ Timeout: subtitle block not found. Please check the URL is correct and the page is fully loaded.")
            await browser.close()
            return

        # crawl all segments of the original and translation
        segments = await page.eval_on_selector_all(
            SEL_SEGMENT,
            """(els, [selOrig, selTrans]) => els.map(el => ({
                original:    el.querySelector(selOrig)?.innerText?.trim()  ?? '',
                translation: el.querySelector(selTrans)?.innerText?.trim() ?? '',
            }))""",
            [SEL_ORIGINAL, SEL_TRANSLATION]
        )

        # Filter empty segments
        originals    = [s["original"]    for s in segments if s["original"]]
        translations = [s["translation"] for s in segments if s["translation"]]

        if not originals:
            print("⚠  No subtitle content found, please check if the selector is still correct.")
            print(f"   SEL_SEGMENT  = {SEL_SEGMENT!r}")
            print(f"   SEL_ORIGINAL = {SEL_ORIGINAL!r}")
            await browser.close()
            return

        orig_path  = OUTPUT_DIR / f"{output_stem}_transcript.txt"
        trans_path = OUTPUT_DIR / f"{output_stem}_translation.txt"

        orig_path.write_text("\n".join(originals),     encoding="utf-8")
        trans_path.write_text("\n".join(translations), encoding="utf-8")

        print(f"✅ Transcript  → {orig_path}  （{len(originals)} segments）")
        print(f"✅ Translation  → {trans_path}  （{len(translations)} segments）")

        await browser.close()


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--save-session":
        asyncio.run(save_session())
    elif len(sys.argv) == 3:
        meeting_url  = sys.argv[1]
        output_stem  = sys.argv[2]
        asyncio.run(crawl(meeting_url, output_stem))
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()