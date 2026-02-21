
// Removed require('node-fetch') as Node 24 has native fetch

async function testFetchModels() {
  console.log("Testing fetchModels...");
  try {
    const response = await fetch("http://127.0.0.1:8000/llm/models");
    console.log("Response status:", response.status);

    if (!response.ok) {
      console.error("Response not OK");
      return;
    }

    const data = await response.json();
    console.log("Response data:", JSON.stringify(data, null, 2));

    if (data.status !== "ok" || !data.models) {
      console.error("Invalid data format");
    } else {
      console.log("Models fetched successfully:", data.models.length);
    }

  } catch (error) {
    console.error("Fetch failed:", error);
  }
}

testFetchModels();
