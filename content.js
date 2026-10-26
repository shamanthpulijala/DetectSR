// Content script for DetectShieldReview extension
// This runs on all web pages to enable context menu functionality

// Listen for messages from the extension
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "getPageUrl") {
    sendResponse({ url: window.location.href });
  }
  return true;
});

// Optional: Add visual indicators for analyzed links
// This can be extended to mark links that have been checked
console.log("DetectShieldReview content script loaded");
