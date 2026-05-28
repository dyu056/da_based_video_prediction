import './styles.css';

const cases = [
  {
    id: 'basketball',
    title: 'Basketball Shooting',
    prompt: 'A basketball player shooting',
    slug: 'a_basketball_player_shooting',
    baseline: 'Clean court and player, but motion remains local.',
    ours: 'Clearer shooting trajectory and follow-through; mild appearance drift.',
    takeaway: 'Retrieved motion helps resolve arm extension and ball-release timing.'
  },
  {
    id: 'swing',
    title: 'Playground Swing',
    prompt: 'A child swinging on a playground swing',
    slug: 'a_child_swinging_on_a_playground_swing',
    baseline: 'Stable face, ropes, and background; forward motion is modest.',
    ours: 'Stronger depth and camera motion, with some blur and identity softening.',
    takeaway: 'Observation supplies a concrete oscillatory trajectory.'
  },
  {
    id: 'dog',
    title: 'Dog Fetching',
    prompt: 'A dog fetching a ball',
    slug: 'a_dog_fetching_a_ball',
    baseline: 'Mostly stationary dog; stable but weak dynamics.',
    ours: 'More body turning and approach motion, with softer detail.',
    takeaway: 'Retrieval increases action energy when the prompt is underspecified.'
  },
  {
    id: 'horse',
    title: 'Horse Galloping',
    prompt: 'A horse galloping across a field',
    slug: 'a_horse_galloping_across_a_field',
    baseline: 'Plausible gallop pose; conservative lateral movement.',
    ours: 'More energetic tracking and body motion; late frames may drift.',
    takeaway: 'Aligned motion evidence improves dynamics but can weaken framing.'
  },
  {
    id: 'surfing',
    title: 'Surfing',
    prompt: 'A surfer riding a small wave',
    slug: 'a_surfer_riding_a_small_wave',
    baseline: 'Readable wave scene with moderate motion.',
    ours: 'Stronger board, wave, and camera motion, with subject drift.',
    takeaway: 'Observation helps coordinate water deformation and subject movement.'
  },
  {
    id: 'running',
    title: 'City Running',
    prompt: 'People running on a city skywalk',
    slug: 'people_running_on_a_city_skywalk',
    baseline: 'Foreground runner moves while road and crowd stay static.',
    ours: 'Stronger forward motion and global scene flow, with mild blur.',
    takeaway: 'Scene-level observation supports camera parallax and background dynamics.'
  }
];

const slides = [
  {
    type: 'cover',
    eyebrow: 'Data Assimilation + Machine Learning Final Project',
    title: 'Scene-Assimilated Video World Models',
    subtitle: 'Retrieval-guided data assimilation for image-to-video generation'
  },
  {
    type: 'concept',
    eyebrow: 'Problem',
    title: 'Image-to-video generation is an open-loop forecast',
    image: '/media/assets/observation_analysis_paradigm_manual.png',
    points: [
      'The model receives one anchor image and a short prompt.',
      'It must roll out a plausible future scene from its learned prior.',
      'Short prompts omit timing, contact, acceleration, camera motion, and background flow.'
    ],
    claim: 'The missing piece is an explicit observation update during rollout.'
  },
  {
    type: 'concept',
    eyebrow: 'Paradigm',
    title: 'Retrieval becomes a sparse scene observation',
    image: '/media/assets/method_analysis_embedded_manual.png',
    points: [
      'Forecast tokens come from the base image-to-video generator.',
      'Observation tokens come from one retrieved real-video trajectory.',
      'The analysis state is the forecast representation after in-context fusion.'
    ],
    formula: 'zᵃₜ = Aθ(zᶠₜ, O, I, p)'
  },
  {
    type: 'concept',
    eyebrow: 'Evidence',
    title: 'A toy ETKF study checks the information claim',
    image: '/media/assets/etkf_information_metrics.png',
    points: [
      'Moving MNIST gives a controlled latent-state assimilation setting.',
      'Increasing ETKF ensemble size reduces the forecast-analysis information gap.',
      'This motivates attention as a tractable learned analysis operator for video DiTs.'
    ],
    metrics: [
      ['8 members', '295.4926', '0.5709'],
      ['100 members', '105.9172', '0.2477'],
      ['500 members', '19.6006', '0.0622']
    ]
  },
  {
    type: 'method',
    eyebrow: 'ML System',
    title: 'LanguageBind retrieval + Video-as-Prompt path',
    image: '/media/assets/retrieval_vap_architecture_imagegen.png',
    points: [
      'OpenVid-style memory stores real videos and captions.',
      'LanguageBind embeds prompt, anchor image, and candidate video in one space.',
      'The in-context DiT branch fuses retrieved observation tokens with forecast tokens.'
    ]
  },
  {
    type: 'qualitative',
    eyebrow: 'Qualitative Analysis',
    title: 'Side-by-side video comparison',
    caseIndex: 0
  },
  {
    type: 'summary',
    eyebrow: 'Reading the cases',
    title: 'Observation improves motion when retrieval is compatible',
    columns: [
      ['Motion gains', 'Follow-through, body-state transition, water motion, road flow, camera parallax.'],
      ['Costs', 'Blur, subject drift, identity softening, framing instability when retrieval mismatches.'],
      ['Criterion', 'A useful analysis update adds motion evidence without overwriting anchor-state variables.']
    ]
  },
  {
    type: 'takeaways',
    eyebrow: 'Conclusion',
    title: 'Takeaways',
    points: [
      'Treat the base video generator as the forecast model.',
      'Treat retrieved real clips as sparse observations of scene dynamics.',
      'Treat in-context attention as a learned analysis operator.',
      'Evaluate retrieval as state estimation, not just as extra prompt context.'
    ]
  }
];

let currentSlide = 0;
let currentCase = 0;
let syncedPlayback = true;
let isPlaying = false;

const app = document.querySelector('#app');

function mediaPath(slug, kind) {
  if (kind === 'reference') return `/media/cases/${slug}/reference.mp4`;
  if (kind === 'anchor') return `/media/cases/${slug}/anchor.png`;
  return `/media/cases/${slug}/${kind}.mp4`;
}

function render() {
  const slide = slides[currentSlide];
  app.innerHTML = `
    <main class="deck">
      <header class="topbar">
        <div>
          <span class="deck-label">RGDA Presentation</span>
          <span class="progress-text">${currentSlide + 1} / ${slides.length}</span>
        </div>
        <nav class="slide-nav" aria-label="Slide navigation">
          <button class="icon-button" data-action="prev" aria-label="Previous slide">‹</button>
          <button class="icon-button" data-action="next" aria-label="Next slide">›</button>
        </nav>
      </header>
      <section class="slide slide-${slide.type}">
        ${renderSlide(slide)}
      </section>
      <footer class="progress"><span style="width:${((currentSlide + 1) / slides.length) * 100}%"></span></footer>
    </main>
  `;

  bindDeckControls();
  if (slide.type === 'qualitative') bindQualitativeControls();
}

function renderSlide(slide) {
  if (slide.type === 'cover') {
    return `
      <div class="cover-grid">
        <div class="cover-copy">
          <p class="eyebrow">${slide.eyebrow}</p>
          <h1>${slide.title}</h1>
          <p class="subtitle">${slide.subtitle}</p>
          <div class="role-row">
            <span>Forecast</span>
            <span>Observation</span>
            <span>Analysis</span>
          </div>
        </div>
        <img class="cover-image" src="/media/assets/intro_concept_imagegen.png" alt="Conceptual video world model illustration" />
      </div>
    `;
  }

  if (slide.type === 'concept' || slide.type === 'method') {
    return `
      <div class="split">
        <div class="copy-block">
          <p class="eyebrow">${slide.eyebrow}</p>
          <h2>${slide.title}</h2>
          <ul class="point-list">${slide.points.map((point) => `<li>${point}</li>`).join('')}</ul>
          ${slide.formula ? `<div class="formula">${slide.formula}</div>` : ''}
          ${slide.claim ? `<p class="claim">${slide.claim}</p>` : ''}
          ${slide.metrics ? renderMetrics(slide.metrics) : ''}
        </div>
        <figure class="figure-panel">
          <img src="${slide.image}" alt="${slide.title}" />
        </figure>
      </div>
    `;
  }

  if (slide.type === 'qualitative') {
    return renderQualitativeSlide();
  }

  if (slide.type === 'summary') {
    return `
      <p class="eyebrow">${slide.eyebrow}</p>
      <h2>${slide.title}</h2>
      <div class="summary-grid">
        ${slide.columns.map(([heading, body]) => `
          <article class="summary-card">
            <h3>${heading}</h3>
            <p>${body}</p>
          </article>
        `).join('')}
      </div>
      <img class="wide-strip" src="/media/assets/qualitative_six_case_late_panel.png" alt="Six-case qualitative panel" />
    `;
  }

  return `
    <div class="takeaway-layout">
      <div>
        <p class="eyebrow">${slide.eyebrow}</p>
        <h2>${slide.title}</h2>
        <ul class="point-list big">${slide.points.map((point) => `<li>${point}</li>`).join('')}</ul>
      </div>
      <div class="closing-card">
        Retrieval should be evaluated as an estimator: does aligned evidence close the missing-dynamics gap without overwriting image-conditioned state?
      </div>
    </div>
  `;
}

function renderMetrics(rows) {
  return `
    <table class="metrics">
      <thead><tr><th>ETKF ensemble</th><th>D<sub>KL</sub> ↓</th><th>Residual I(Z;Y) ↓</th></tr></thead>
      <tbody>${rows.map((row) => `<tr><td>${row[0]}</td><td>${row[1]}</td><td>${row[2]}</td></tr>`).join('')}</tbody>
    </table>
  `;
}

function renderQualitativeSlide() {
  const item = cases[currentCase];
  return `
    <div class="qual-header">
      <div>
        <p class="eyebrow">Qualitative Analysis</p>
        <h2>${item.title}</h2>
        <p class="prompt">Prompt: ${item.prompt}</p>
      </div>
      <div class="case-controls">
        <select id="caseSelect" aria-label="Select qualitative case">
          ${cases.map((c, index) => `<option value="${index}" ${index === currentCase ? 'selected' : ''}>${c.title}</option>`).join('')}
        </select>
        <button class="control-button" data-action="toggle-play">${isPlaying ? 'Pause all' : 'Play all'}</button>
        <label class="toggle">
          <input type="checkbox" id="syncToggle" ${syncedPlayback ? 'checked' : ''} />
          <span>Sync</span>
        </label>
      </div>
    </div>

    <div class="comparison-grid">
      ${renderVideoPanel('Retrieved observation', 'reference', item.slug, 'Real scene dynamics used as sparse observation.', '')}
      ${renderVideoPanel('Wan2.1 baseline', 'baseline', item.slug, item.baseline, 'baseline')}
      ${renderVideoPanel('Scene-assimilated', 'ours', item.slug, item.ours, 'ours')}
    </div>

    <div class="analysis-row">
      <img src="${mediaPath(item.slug, 'anchor')}" alt="${item.title} anchor image" />
      <p>${item.takeaway}</p>
    </div>
  `;
}

function renderVideoPanel(title, kind, slug, note, accent) {
  return `
    <article class="video-panel ${accent}">
      <div class="video-title">
        <span>${title}</span>
      </div>
      <video src="${mediaPath(slug, kind)}" muted loop playsinline preload="metadata"></video>
      <p>${note}</p>
    </article>
  `;
}

function bindDeckControls() {
  app.querySelector('[data-action="prev"]').addEventListener('click', previousSlide);
  app.querySelector('[data-action="next"]').addEventListener('click', nextSlide);
}

function bindQualitativeControls() {
  const select = app.querySelector('#caseSelect');
  select.addEventListener('change', (event) => {
    currentCase = Number(event.target.value);
    isPlaying = false;
    render();
  });

  app.querySelector('#syncToggle').addEventListener('change', (event) => {
    syncedPlayback = event.target.checked;
  });

  app.querySelector('[data-action="toggle-play"]').addEventListener('click', () => {
    isPlaying = !isPlaying;
    const videos = [...app.querySelectorAll('video')];
    if (syncedPlayback) videos.forEach((video) => { video.currentTime = 0; });
    videos.forEach((video) => {
      if (isPlaying) video.play();
      else video.pause();
    });
    app.querySelector('[data-action="toggle-play"]').textContent = isPlaying ? 'Pause all' : 'Play all';
  });

  app.querySelectorAll('video').forEach((video) => {
    video.addEventListener('play', () => {
      if (!syncedPlayback) return;
      const originTime = video.currentTime;
      app.querySelectorAll('video').forEach((peer) => {
        if (peer !== video && Math.abs(peer.currentTime - originTime) > 0.25) peer.currentTime = originTime;
      });
    });
  });
}

function nextSlide() {
  currentSlide = Math.min(slides.length - 1, currentSlide + 1);
  isPlaying = false;
  render();
}

function previousSlide() {
  currentSlide = Math.max(0, currentSlide - 1);
  isPlaying = false;
  render();
}

document.addEventListener('keydown', (event) => {
  if (event.key === 'ArrowRight' || event.key === 'PageDown' || event.key === ' ') nextSlide();
  if (event.key === 'ArrowLeft' || event.key === 'PageUp') previousSlide();
  if (event.key >= '1' && event.key <= String(cases.length) && slides[currentSlide].type === 'qualitative') {
    currentCase = Number(event.key) - 1;
    isPlaying = false;
    render();
  }
});

render();
