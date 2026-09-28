/**
 * Indian Equity AI - Landing Page Interactive Scripts
 * Provides interactive pipeline stepping, responsive mobile navigation,
 * dynamic glass navbar scroll effects, and accessibility enhancements.
 */

document.addEventListener('DOMContentLoaded', () => {
  // 1. Mobile Navigation Toggle
  const mobileToggle = document.getElementById('mobile-toggle');
  const navMenu = document.getElementById('nav-menu');

  if (mobileToggle && navMenu) {
    mobileToggle.addEventListener('click', () => {
      const isExpanded = navMenu.classList.toggle('active');
      mobileToggle.setAttribute('aria-expanded', isExpanded ? 'true' : 'false');
    });

    // Close menu when clicking any nav link
    navMenu.querySelectorAll('.nav-link').forEach(link => {
      link.addEventListener('click', () => {
        navMenu.classList.remove('active');
        mobileToggle.setAttribute('aria-expanded', 'false');
      });
    });
  }

  // 2. Sticky Navbar Glassmorphism Scroll Elevation
  const navbar = document.getElementById('navbar');
  const handleScroll = () => {
    if (window.scrollY > 40) {
      navbar.style.backgroundColor = 'rgba(8, 12, 21, 0.92)';
      navbar.style.boxShadow = '0 10px 30px rgba(0, 0, 0, 0.5)';
    } else {
      navbar.style.backgroundColor = 'rgba(14, 20, 36, 0.75)';
      navbar.style.boxShadow = 'none';
    }
  };
  window.addEventListener('scroll', handleScroll, { passive: true });
  handleScroll();

  // 3. Interactive Dataflow Pipeline Stages
  const pipelineStepsData = {
    1: {
      badge: "Stage 01 • Data Layer",
      title: "Market Data Ingestion & Validation",
      desc: "Daily historical OHLCV data is retrieved for Indian Nifty 50 equities via Yahoo Finance, while real-time tick updates stream via Upstox V3 Market Data Feed using binary Protobuf decoding. All series undergo quantitative data cleaning to eliminate negative volumes, zero prices, duplicate timestamps, and unadjusted split anomalies.",
      meta1: "YFinance (Daily) + Upstox V3 (Ticks)",
      meta2: "Nifty 50 Universe (RELIANCE.NS, TCS.NS, etc.)",
      meta3: "Monotonic strictly chronological ordering"
    },
    2: {
      badge: "Stage 02 • Quantitative Engineering",
      title: "Feature Engineering & Preprocessing",
      desc: "Calculates over 25 technical indicators including RSI, MACD, Bollinger Bands, and Exponential Moving Averages alongside volume metrics and 5-day autoregressive return lags. Data scaling (MinMaxScaler) is strictly fitted on past training data to guarantee zero future-lookahead leakage.",
      meta1: "RSI(14), MACD(12,26,9), BB(20,2), EMA(9,21,50,200)",
      meta2: "Lagged returns t-1 to t-5 + OHLC volatility",
      meta3: "Causal windowing with strictly no forward leakage"
    },
    3: {
      badge: "Stage 03 • Dual Machine Learning",
      title: "LSTM & XGBoost Model Processing",
      desc: "Two distinct algorithms process the engineered inputs concurrently: a 2-layer Stacked LSTM Neural Network evaluates 60-step temporal sequences to model long-range autocorrelation, while a regularized XGBoost Regressor captures non-linear tabular interactions and feature thresholds.",
      meta1: "LSTM: 64 & 32 units, 20% Dropout, Adam lr=0.001",
      meta2: "XGBoost: 200 estimators, max depth 5, L1/L2 reg",
      meta3: "Complementary sequential & tabular inference"
    },
    4: {
      badge: "Stage 04 • Quantitative Synthesis",
      title: "Prediction & Probability Modeling",
      desc: "Inference outputs are synthesized to predict tomorrow's closing price (t+1) and the probability of an upward vs downward movement. The models produce continuous rupee projections alongside confidence intervals to gauge model certainty.",
      meta1: "Point estimate for next trading day Close",
      meta2: "Directional probability scoring (Up vs Down)",
      meta3: "Cross-model agreement & consensus verification"
    },
    5: {
      badge: "Stage 05 • Decision Engine",
      title: "Confluence Signal Generation",
      desc: "A quantitative rule engine translates predictive price targets and technical momentum indicators into actionable research signals: BUY, HOLD, or SELL. A BUY signal is issued only when both models predict positive returns and technical filters (e.g. RSI not overbought, bullish MACD) confirm confluence.",
      meta1: "BUY: Positive delta + RSI < 70 + Model Confluence",
      meta2: "SELL: Negative delta + RSI > 30 + Downward Consensus",
      meta3: "HOLD: Divergent model signals or conflicting indicators"
    },
    6: {
      badge: "Stage 06 • Visualization & Backtest",
      title: "Interactive Streamlit Dashboard",
      desc: "The cloud-hosted Streamlit dashboard renders interactive Plotly candlestick charts, real-time tick metrics, model comparison tables, walk-forward cumulative return curves, and explainability breakdown cards for institutional-grade presentation.",
      meta1: "Streamlit Cloud Native (Python 3.12)",
      meta2: "Interactive Plotly Financial Candlesticks",
      meta3: "Walk-forward backtest equity curves & metrics"
    }
  };

  const pipelineNodes = document.querySelectorAll('.pipeline-node');
  const stepBadge = document.getElementById('step-badge');
  const stepTitle = document.getElementById('step-title');
  const stepDesc = document.getElementById('step-desc');
  const stepMeta1 = document.getElementById('step-meta-1');
  const stepMeta2 = document.getElementById('step-meta-2');
  const stepMeta3 = document.getElementById('step-meta-3');
  const detailsCard = document.getElementById('pipeline-details');

  pipelineNodes.forEach(node => {
    node.addEventListener('click', () => {
      const stepNumber = node.getAttribute('data-step');
      const data = pipelineStepsData[stepNumber];

      if (!data) return;

      // Update active button state
      pipelineNodes.forEach(n => n.classList.remove('active'));
      node.classList.add('active');

      // Subtle fade animation
      if (detailsCard) {
        detailsCard.style.opacity = '0';
        detailsCard.style.transform = 'translateY(4px)';

        setTimeout(() => {
          stepBadge.textContent = data.badge;
          stepTitle.textContent = data.title;
          stepDesc.textContent = data.desc;
          stepMeta1.textContent = data.meta1;
          stepMeta2.textContent = data.meta2;
          stepMeta3.textContent = data.meta3;

          detailsCard.style.transition = 'opacity 250ms ease, transform 250ms ease';
          detailsCard.style.opacity = '1';
          detailsCard.style.transform = 'translateY(0)';
        }, 150);
      }
    });
  });

  // 4. Smooth Anchor Link Scrolling with Offset for Sticky Header
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      const targetId = this.getAttribute('href');
      if (targetId === '#' || targetId === '') return;

      const targetEl = document.querySelector(targetId);
      if (targetEl) {
        e.preventDefault();
        const headerOffset = 80;
        const elementPosition = targetEl.getBoundingClientRect().top;
        const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

        window.scrollTo({
          top: offsetPosition,
          behavior: 'smooth'
        });
      }
    });
  });
});
