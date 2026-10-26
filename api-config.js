(() => {
  const isExtension = Boolean(globalThis.chrome?.runtime?.id || window.location.protocol === "chrome-extension:");
  if (isExtension) {
    document.documentElement.classList.add("extension-page");
  }

  const params = new URLSearchParams(window.location.search);
  const queryBase = params.get("api");
  const queryUrl = params.get("url") || params.get("data");
  const storedBase = window.localStorage.getItem("detectshieldApiBaseUrl");

  // When served as a web app, use the current website origin as the API backend
  // When running inside a Chrome extension popup, use the stored URL or localhost:8000
  const defaultBase = isExtension ? "http://localhost:8000" : window.location.origin;
  const configuredBase = window.DETECTSHIELD_API_BASE_URL || queryBase || storedBase || defaultBase;

  try {
    const parsedBase = new URL(configuredBase, window.location.href);
    if (parsedBase.protocol === "http:" || parsedBase.protocol === "https:") {
      window.DETECTSHIELD_API_BASE_URL = `${parsedBase.origin}${parsedBase.pathname}`.replace(/\/+$/, "");
      if (queryBase) window.localStorage.setItem("detectshieldApiBaseUrl", window.DETECTSHIELD_API_BASE_URL);
    }
  } catch {
    window.DETECTSHIELD_API_BASE_URL = defaultBase;
  }

  // If opened with ?url=..., ensure pendingAnalysis is primed in storage for automatic scan
  if (queryUrl && globalThis.chrome?.storage?.local) {
    chrome.storage.local.set({
      pendingAnalysis: {
        data: decodeURIComponent(queryUrl),
        type: params.get("type") || "URL",
        timestamp: Date.now()
      }
    });
  }
})();