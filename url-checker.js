// URL Checker Content Script - Shows inline safety check

// Create floating safety indicator
function createSafetyIndicator() {
  let indicator = document.getElementById('detectshield-indicator');
  if (indicator) return indicator;

  indicator = document.createElement('div');
  indicator.id = 'detectshield-indicator';
  indicator.style.cssText = `
    position: fixed;
    top: 60px;
    right: 20px;
    z-index: 2147483647;
    background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    border: 2px solid #06b6d4;
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 8px 32px rgba(6, 182, 212, 0.35);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    color: white;
    display: none;
    min-width: 300px;
    max-width: 420px;
    animation: dsSlideIn 0.3s ease-out;
  `;
  
  indicator.innerHTML = `
    <style>
      @keyframes dsSlideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
      }
      @keyframes dsSpin {
        to { transform: rotate(360deg); }
      }
      @keyframes dsPulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
      }
      #detectshield-indicator .loading {
        animation: dsPulse 1.5s ease-in-out infinite;
      }
      #view-full-report:hover {
        opacity: 0.9;
        transform: translateY(-1px);
      }
    </style>
    <div id="indicator-content">
      <div class="loading" style="display: flex; align-items: center; gap: 10px;">
        <div style="width: 18px; height: 18px; border: 2px solid #06b6d4; border-top-color: transparent; border-radius: 50%; animation: dsSpin 0.8s linear infinite;"></div>
        <span style="font-size: 13px; color: #cbd5e1;">DetectShield: Checking URL safety...</span>
      </div>
    </div>
  `;
  
  document.body.appendChild(indicator);
  return indicator;
}

// Show safety result
function showSafetyResult(url, verdict, explanation) {
  const indicator = createSafetyIndicator();
  indicator.style.display = 'block';
  
  const vLower = String(verdict || 'caution').toLowerCase();
  const colors = {
    'safe': { bg: '#10b981', border: '#10b981', icon: '✅', title: 'SAFE' },
    'caution': { bg: '#fb923c', border: '#fb923c', icon: '⚠️', title: 'CAUTION' },
    'dangerous': { bg: '#ef4444', border: '#ef4444', icon: '🚫', title: 'DANGEROUS' }
  };
  
  const style = colors[vLower] || colors.caution;
  indicator.style.borderColor = style.border;
  
  const content = document.getElementById('indicator-content');
  content.innerHTML = `
    <div style="display: flex; flex-direction: column; gap: 10px;">
      <div style="display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 22px;">${style.icon}</span>
        <div style="flex: 1; overflow: hidden;">
          <div style="font-weight: 700; font-size: 14px; color: ${style.bg}; letter-spacing: 0.5px;">${style.title}</div>
          <div style="font-size: 11px; color: #94a3b8; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${url}">${url}</div>
        </div>
        <button id="close-indicator" style="background: none; border: none; color: #94a3b8; cursor: pointer; font-size: 18px; line-height: 1; padding: 0 4px;" title="Dismiss">✕</button>
      </div>
      <div style="font-size: 12px; color: #e2e8f0; line-height: 1.4;">${explanation}</div>
      <button id="view-full-report" style="
        background: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 100%);
        border: none;
        color: white;
        padding: 9px 14px;
        border-radius: 8px;
        cursor: pointer;
        font-size: 13px;
        font-weight: 600;
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        box-shadow: 0 4px 12px rgba(6, 182, 212, 0.3);
        transition: all 0.2s ease;
      ">
        <span>View Full Report</span>
        <span>🛡️</span>
      </button>
    </div>
  `;
  
  // Close button
  document.getElementById('close-indicator').addEventListener('click', (e) => {
    e.stopPropagation();
    indicator.style.display = 'none';
  });
  
  // View full report button
  const reportBtn = document.getElementById('view-full-report');
  reportBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    reportBtn.innerHTML = '<span>Opening Full Report...</span> <span>⏳</span>';
    reportBtn.style.opacity = '0.85';
    reportBtn.disabled = true;

    // Send request to background script to store pending analysis and launch UI
    chrome.runtime.sendMessage({
      action: 'analyzeFromPage',
      data: url,
      type: 'URL'
    }, (response) => {
      // Direct fallback if messaging had an issue
      if (chrome.runtime.lastError || !response || !response.success) {
        try {
          const targetUrl = chrome.runtime.getURL(`index.html?url=${encodeURIComponent(url)}&type=URL`);
          window.open(targetUrl, '_blank');
        } catch (err) {
          console.warn('Fallback window.open failed:', err);
        }
      }
      setTimeout(() => {
        indicator.style.display = 'none';
      }, 600);
    });
  });
  
  // Auto-hide after 12 seconds
  if (window._dsHideTimer) clearTimeout(window._dsHideTimer);
  window._dsHideTimer = setTimeout(() => {
    if (indicator && indicator.style.display !== 'none') {
      indicator.style.display = 'none';
    }
  }, 12000);
}

// Extract URL from selected text
function extractURL(text) {
  if (!text || typeof text !== 'string') return null;
  const clean = text.trim();
  const urlPattern = /(https?:\/\/[^\s]+)|(www\.[^\s]+)|([a-zA-Z0-9-]+\.[a-zA-Z]{2,}[^\s]*)/gi;
  const matches = clean.match(urlPattern);
  
  if (matches && matches.length > 0) {
    let url = matches[0].replace(/[.,;:)]+$/, ''); // Strip trailing punctuation
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
      url = 'https://' + url;
    }
    return url;
  }
  return null;
}

// Safety check with live AI backend integration
async function quickCheck(url) {
  // 1. Try checking via live FastAPI AI model in background worker
  try {
    const aiResult = await new Promise((resolve) => {
      chrome.runtime.sendMessage({ action: 'checkUrlSafety', url }, (res) => {
        if (chrome.runtime.lastError || !res || !res.success) {
          resolve(null);
        } else {
          resolve(res);
        }
      });
    });

    if (aiResult) {
      const v = String(aiResult.verdict).toLowerCase();
      let icon = v === 'safe' ? '✅' : v === 'dangerous' ? '🚫' : '⚠️';
      return {
        verdict: aiResult.verdict,
        explanation: `${icon} ${aiResult.explanation} (Confidence: ${aiResult.confidenceScore}%)`
      };
    }
  } catch (err) {
    // Background not responding, fallback to heuristics
  }

  // 2. Fallback heuristics if backend unreachable
  try {
    const urlObj = new URL(url);
    const domain = urlObj.hostname.toLowerCase();
    
    const suspiciousPatterns = [
      /paypa1/i, /g00gle/i, /amaz0n/i, /micr0soft/i,
      /\.tk$/i, /\.ml$/i, /\.ga$/i, /\.cf$/i,
      /login.*verify/i, /secure.*account/i, /update.*payment/i
    ];
    
    const hasSuspiciousPattern = suspiciousPatterns.some(pattern => 
      domain.match(pattern) || url.match(pattern)
    );
    
    if (hasSuspiciousPattern) {
      return {
        verdict: 'Dangerous',
        explanation: '⚠️ Suspicious patterns commonly used in phishing attacks detected.'
      };
    }
    
    if (!url.startsWith('https://')) {
      return {
        verdict: 'Caution',
        explanation: '⚠️ This site does not use HTTPS encryption.'
      };
    }
    
    const safeDomains = [
      'google.com', 'youtube.com', 'facebook.com', 'twitter.com', 'x.com',
      'github.com', 'stackoverflow.com', 'wikipedia.org', 'amazon.com',
      'whatsapp.com', 'instagram.com', 'discord.com', 'apple.com',
      'microsoft.com', 'linkedin.com', 'paypal.com', 'uber.com', 'netflix.com'
    ];
    
    const isSafeDomain = safeDomains.some(safe => domain === safe || domain.endsWith('.' + safe));
    
    if (isSafeDomain) {
      return {
        verdict: 'Safe',
        explanation: '✅ Verified, trusted web domain.'
      };
    }
    
    return {
      verdict: 'Caution',
      explanation: '⚠️ Unknown website. Click "View Full Report" for detailed AI analysis.'
    };
    
  } catch (error) {
    return {
      verdict: 'Caution',
      explanation: '⚠️ URL format unverified. Click "View Full Report" to analyze.'
    };
  }
}

// Trigger safety scan on detected URL
async function handleUrlDetection(rawText) {
  const url = extractURL(rawText);
  if (!url) return;

  const indicator = createSafetyIndicator();
  indicator.style.display = 'block';

  const result = await quickCheck(url);
  showSafetyResult(url, result.verdict, result.explanation);
}

// Listen for mouse selection
document.addEventListener('mouseup', () => {
  const selectedText = window.getSelection().toString().trim();
  if (selectedText.length > 0) {
    handleUrlDetection(selectedText);
  }
});

// Listen for clipboard copy
document.addEventListener('copy', () => {
  const selectedText = window.getSelection().toString().trim();
  if (selectedText.length > 0) {
    handleUrlDetection(selectedText);
  }
});

// Listen for messages from background script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'checkURL') {
    quickCheck(request.url).then(result => {
      showSafetyResult(request.url, result.verdict, result.explanation);
      sendResponse({ success: true });
    });
    return true;
  }
});

console.log('DetectShieldReview URL Checker loaded ✓');
