// Background service worker for DetectShieldReview extension

// Helper to open the full analysis report
async function openReportInterface(analysisData) {
  // Store data in local storage for index.html to read via getPendingAnalysis
  await chrome.storage.local.set({ pendingAnalysis: analysisData });

  try {
    const queryParam = `?url=${encodeURIComponent(analysisData.data)}&type=${encodeURIComponent(analysisData.type || "URL")}`;
    const extensionBase = chrome.runtime.getURL("index.html");
    const targetUrl = extensionBase + queryParam;

    const existingTabs = await chrome.tabs.query({ url: extensionBase + "*" });
    if (existingTabs && existingTabs.length > 0) {
      await chrome.tabs.update(existingTabs[0].id, { active: true, url: targetUrl });
      if (existingTabs[0].windowId) {
        await chrome.windows.update(existingTabs[0].windowId, { focused: true });
      }
    } else {
      await chrome.tabs.create({ url: targetUrl, active: true });
    }
  } catch (tabErr) {
    console.error("Failed to open report tab:", tabErr);
  }
}

// Create context menu when extension is installed
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "analyzeLink",
    title: "🛡️ Analyze Link with DetectShieldReview",
    contexts: ["link"]
  });

  chrome.contextMenus.create({
    id: "analyzeText",
    title: "🛡️ Analyze Message with DetectShieldReview",
    contexts: ["selection"]
  });

  chrome.contextMenus.create({
    id: "analyzePage",
    title: "🛡️ Analyze Current Page",
    contexts: ["page"]
  });

  console.log("DetectShieldReview extension installed successfully!");
});

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener((info, tab) => {
  let dataToAnalyze = null;
  let analysisType = null;

  if (info.menuItemId === "analyzeLink") {
    dataToAnalyze = info.linkUrl;
    analysisType = "URL";
  } else if (info.menuItemId === "analyzeText") {
    dataToAnalyze = info.selectionText;
    analysisType = "Message";
  } else if (info.menuItemId === "analyzePage") {
    dataToAnalyze = info.pageUrl;
    analysisType = "URL";
  }

  if (dataToAnalyze) {
    openReportInterface({
      data: dataToAnalyze,
      type: analysisType,
      timestamp: Date.now()
    });
  }
});

// Unified message router
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  // 1. Popup requesting pending analysis data
  if (request.action === "getPendingAnalysis") {
    chrome.storage.local.get(["pendingAnalysis"], (result) => {
      sendResponse(result.pendingAnalysis || null);
      chrome.storage.local.remove("pendingAnalysis");
    });
    return true; // Keep channel open for async response
  }

  // 2. Content script requesting to open full report
  if (request.action === "analyzeFromPage") {
    const analysisData = {
      data: request.data,
      type: request.type || "URL",
      timestamp: Date.now()
    };

    openReportInterface(analysisData).then(() => {
      sendResponse({ success: true });
    }).catch(err => {
      console.error("Error opening report:", err);
      sendResponse({ success: false, error: err.message });
    });
    return true;
  }

  // 3. Quick URL safety check using live FastAPI backend
  if (request.action === "checkUrlSafety") {
    (async () => {
      try {
        const apiBase = "http://localhost:8000";
        const res = await fetch(`${apiBase}/predict-url`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url: request.url })
        });
        if (res.ok) {
          const data = await res.json();
          sendResponse({
            success: true,
            verdict: data.verdict || "Safe",
            confidenceScore: data.confidenceScore || 90,
            explanation: data.explanation || "Verified URL"
          });
          return;
        }
      } catch (err) {
        console.warn("Backend unavailable for quick safety check:", err);
      }
      sendResponse({ success: false });
    })();
    return true;
  }
});
