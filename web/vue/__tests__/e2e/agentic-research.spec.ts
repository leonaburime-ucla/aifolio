import { test, expect } from "@playwright/test";

test.describe("Agentic Research", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/agentic-research");
    await page.waitForSelector("table tbody tr", { timeout: 60_000 });
  });

  test.fixme("dataset auto-loads on mount and populates table", async ({ page }) => {
    // FIXME: DataTables.net creates multiple <table> elements; locator needs
    // a more specific selector (e.g. data-testid on the wrapper) to avoid
    // hitting an internal DT table with empty tbody.
    const rows = page.locator("table tbody tr");
    const rowCount = await rows.count();
    expect(rowCount).toBeGreaterThan(0);
  });

  test("sklearn tools render grouped categories", async ({ page }) => {
    const toolSection = page.locator("text=ML Algorithms + Sample Prompts");
    await expect(toolSection).toBeVisible();

    await page.waitForSelector("text=Decomposition", { timeout: 10_000 });
    const categories = page.locator("text=/Decomposition|Classification|Clustering|Regression/");
    const count = await categories.count();
    expect(count).toBeGreaterThanOrEqual(1);
  });

  test("dataset table shows column headers", async ({ page }) => {
    const table = page.locator("table").first();
    await expect(table).toBeVisible();

    const headers = table.locator("th");
    const headerCount = await headers.count();
    expect(headerCount).toBeGreaterThan(1);
  });

  test("chat sidebar is present with research mode", async ({ page }) => {
    const sidebar = page.locator("text=/AI\\s*Chat/i");
    await expect(sidebar.first()).toBeVisible();

    const input = page.locator("input[aria-label='Chat input']");
    await expect(input).toBeVisible();
  });

  test("send research prompt and receive response", async ({ page }) => {
    const input = page.locator("input[aria-label='Chat input']");
    await input.fill("Run PCA analysis");

    const sendButton = page.locator("button:has-text('Send')");
    await sendButton.click();

    const thinking = page.locator("text=Thinking...");
    await expect(thinking).toBeVisible({ timeout: 5_000 });

    await expect(thinking).toBeHidden({ timeout: 180_000 });

    const assistantMessages = page.locator(".bg-zinc-100:not(:has(svg))");
    const lastMessage = assistantMessages.last();
    await expect(lastMessage).toBeVisible();

    const messageText = await lastMessage.textContent();
    expect(messageText!.length).toBeGreaterThan(10);
  });
});
