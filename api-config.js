(() => {
  const isExtension = Boolean(globalThis.chrome?.runtime?.id || window.location.protocol === "chrome-extension:");
  if (isExtension) {
    document.documentElement.classList.add("extension-page");
  }

  const params = new URLSearchParams(window.location.search);
  const queryBase = params.get("api");
  const queryUrl = params.get("url") || params.get("data");
  let storedBase = window.localStorage.getItem("detectshieldApiBaseUrl");
  // Invalidate any stale stored localhost dev server ports (e.g. 5173)
  if (storedBase && (storedBase.includes(":5173") || storedBase.includes(":3000"))) {
    window.localStorage.removeItem("detectshieldApiBaseUrl");
    storedBase = null;
  }

  const isLocalhost = Boolean(
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1" ||
    window.location.hostname === "::1" ||
    window.location.protocol === "file:"
  );

  // When running locally or in extension, the backend API is hosted at localhost:8000
  // When hosted on a cloud domain (e.g. Render), use the deployed origin
  let defaultBase = "http://localhost:8000";
  if (!isExtension && !isLocalhost && window.location.origin && window.location.origin.startsWith("http")) {
    defaultBase = window.location.origin;
  }
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