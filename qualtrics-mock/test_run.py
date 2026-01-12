import json
import math
import os
import sys
import time

from playwright.sync_api import sync_playwright


URL = os.environ.get("MOCK_URL", "http://127.0.0.1:8765/")


def wait_for_next_enabled(page):
    page.wait_for_function(
        """
        () => Array.from(document.querySelectorAll('button')).some(
          b => b.textContent.includes('Next trial') && !b.disabled
        )
        """,
        timeout=20000,
    )


def select_matrix_values(page, value="6"):
    # Set 4 rows to the given value (quality, felt_long, hurt_quality, accept)
    keys = ["quality", "felt_long", "hurt_quality", "accept"]
    for k in keys:
        locator = page.locator(f'input[name="{k}"][value="{value}"]')
        locator.first.check()


def set_all_sliders(page, value=3):
    sliders = page.locator('input[type="range"]')
    count = sliders.count()
    for i in range(count):
        v = value + (i % 3)
        sliders.nth(i).evaluate("(el, v) => { el.value = String(v); el.dispatchEvent(new Event('input', { bubbles: true })); }", v)


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(URL, wait_until="load")

        # Start
        page.get_by_text("Start").click()

        # Iterate 5 trials
        for _ in range(5):
            # Ensure we are on a trial screen by checking header
            page.wait_for_selector('h2:has-text("Trial")', timeout=20000)
            # Make matrix selections (also satisfies attention check on P3)
            select_matrix_values(page, value="6")
            # Wait for full render (button enabled) and proceed
            wait_for_next_enabled(page)
            page.get_by_role("button", name="Next trial").click()

        # Sliders
        page.wait_for_selector('h2:has-text("How long did the wait feel")', timeout=10000)
        set_all_sliders(page, value=3)
        page.get_by_role("button", name="Next").click()

        # Wrap-up
        page.wait_for_selector('h2:has-text("Wrap-up")', timeout=10000)
        # Set max acceptable to 6s (first slider on wrap-up)
        slider = page.locator('input[type="range"]').first
        slider.evaluate("(el) => { el.value = '6'; el.dispatchEvent(new Event('input', {bubbles:true})); }")
        page.get_by_role("radio", name="Streaming", exact=True).check()
        page.get_by_role("button", name="Finish").click()

        # Summary
        page.wait_for_selector('h2:has-text("Summary")', timeout=10000)
        pre = page.locator('pre').first
        results_json = pre.inner_text()
        data = json.loads(results_json)

        # Basic validations
        assert len(data.get("trials", [])) == 5, "Expected 5 trials logged"

        # Validate timing accuracy
        allowed_delta_ms = 600  # generous for CI environments
        stream_tft_bounds = (100, 700)
        seen_labels = set()
        for t in data["trials"]:
            seen_labels.add(t["label"])
            target = int(t["latency_ms"]) if isinstance(t["latency_ms"], int) else t["latency_ms"]
            full = t.get("full_render_ms")
            assert full is not None, "full_render_ms missing"
            delta = full - target
            assert abs(delta) <= allowed_delta_ms, f"Full render off target by {delta} ms for {t['label']} ({t['modality']})"

            if t["modality"] == "stream":
                ft = t.get("first_token_ms")
                assert ft is not None, "first_token_ms missing for streaming trial"
                assert stream_tft_bounds[0] <= ft <= stream_tft_bounds[1], f"first_token_ms {ft} outside expected bounds {stream_tft_bounds}"

        print("OK: 5 trials logged; timing within bounds; streaming TFT ok; labels:", sorted(seen_labels))
        print(json.dumps({
            "participant_id": data.get("participant_id"),
            "trial_labels": sorted(seen_labels),
            "wrap": data.get("wrap"),
        }, indent=2))

        browser.close()


if __name__ == "__main__":
    main()

