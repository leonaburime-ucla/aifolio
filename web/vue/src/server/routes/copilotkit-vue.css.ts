import { readFile } from "node:fs/promises";
import { join } from "node:path";

export default defineEventHandler(async (event) => {
  setResponseHeader(event, "content-type", "text/css; charset=utf-8");

  return readFile(
    join(process.cwd(), "node_modules/@copilotkit/vue/dist/styles.css"),
    "utf8",
  );
});
