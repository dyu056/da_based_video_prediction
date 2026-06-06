import './styles.css';

const mediaRoot = `${import.meta.env.BASE_URL}media`;

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
    type: 'problemVideos',
    eyebrow: 'Problem',
    title: 'Short prompts leave motion under-specified',
    lead: 'Image-to-video generators can preserve the first frame, but the event dynamics often remain too conservative when the prompt is short.',
    points: [
      'The model sees one anchor image and a compact text command.',
      'Important latent variables are missing: contact timing, acceleration, camera flow, and background motion.',
      'The result is visually stable but physically weak rollout.'
    ],
    videos: [
      {
        slug: 'a_basketball_player_shooting',
        label: 'Basketball',
        note: 'The shooting pose is clean, but the action barely develops.'
      },
      {
        slug: 'a_surfer_riding_a_small_wave',
        label: 'Surfing',
        note: 'Wave, board, and camera motion remain moderate.'
      },
      {
        slug: 'people_running_on_a_city_skywalk',
        label: 'Running',
        note: 'Foreground motion appears, while scene flow stays weak.'
      }
    ]
  },
  {
    type: 'concept',
    eyebrow: 'Motivation',
    title: 'Video generation can be viewed as open-loop forecasting',
    image: 'observation_analysis_paradigm_manual.png',
    points: [
      'The base generator is a strong learned prior over plausible videos.',
      'But after rollout begins, there is no explicit update from real scene evidence.',
      'Retrieval gives a lightweight way to bring empirical dynamics into the state.'
    ],
    claim: 'The core motivation is to turn retrieval from prompt context into a state-estimation signal.'
  },
  {
    type: 'formula',
    eyebrow: 'Data Assimilation Formulation',
    title: 'Forecast, observation, analysis',
    points: [
      ['Forecast', 'The open-loop latent video rollout from the image and prompt.'],
      ['Observation', 'A retrieved real video that carries sparse scene dynamics.'],
      ['Analysis', 'The corrected latent representation after fusing forecast and observation tokens.']
    ],
    equations: [
      'p(x_t | y_{1:t}) ∝ p(y_t | x_t) p(x_t | y_{1:t-1})',
      'xᵃ = xᶠ + K(y - xᶠ),  K = Pᶠ / (Pᶠ + R)',
      'zᵃₜ = Aθ(zᶠₜ, O, I, p)'
    ],
    explanation: 'We do not claim attention is an exact Kalman filter. The equations define the role of the components: the generator forecasts, retrieval observes, and the in-context pathway performs the learned analysis update.'
  },
  {
    type: 'concept',
    eyebrow: 'Preliminary Test',
    title: 'Moving MNIST checks whether observation closes the information gap',
    image: 'etkf_information_metrics.png',
    points: [
      'A latent ETKF update is applied to a controlled Moving MNIST predictor.',
      'If assimilation is working, the corrected analysis distribution should approach the observation-conditioned rollout.',
      'The result supports the information claim before moving to video DiTs.'
    ],
    metrics: [
      ['8 members', '295.4926', '0.5709'],
      ['100 members', '105.9172', '0.2477'],
      ['500 members', '19.6006', '0.0622']
    ]
  },
  {
    type: 'method',
    eyebrow: 'Main Method',
    title: 'Scene-assimilated analysis inside the video DiT',
    image: 'method_analysis_embedded_manual.png',
    points: [
      'Forecast tokens come from the base image-to-video generator.',
      'Observation tokens come from one retrieved real-video trajectory.',
      'Forecast-side attention reads both streams, producing analysis tokens for denoising.'
    ],
    claim: 'The retrieved clip is not a target to copy; it is sparse evidence for under-specified dynamics.'
  },
  {
    type: 'experiment',
    eyebrow: 'Experimental Method',
    title: 'Retrieval-guided Wan2.1 pilot',
    image: 'retrieval_vap_architecture_imagegen.png',
    steps: [
      ['Query', 'Embed the anchor image and short prompt with LanguageBind.'],
      ['Retrieve', 'Select one nearest OpenVid-style real clip as the observation trajectory.'],
      ['Generate', 'Compare Wan2.1 image-to-video baseline with scene-assimilated inference.'],
      ['Inspect', 'Use six prompt cases to analyze motion gains and spatial consistency costs.']
    ],
    settings: [
      'Same prompt family and anchor-image setup',
      'Baseline: image + prompt only',
      'Ours: image + prompt + retrieved observation video',
      'Qualitative focus: action phase, camera motion, background flow, identity drift',
      'Implementation: LanguageBind retrieval + Video-as-Prompt in-context pathway'
    ]
  },
  {
    type: 'qualitative',
    eyebrow: 'Qualitative Analysis',
    title: 'Side-by-side video comparison',
    caseIndex: 0
  },
  {
    type: 'discussion',
    eyebrow: 'Discussion I',
    title: 'What improves, and why?',
    columns: [
      ['Motion evidence', 'Retrieved clips provide contact timing, pose transition, camera parallax, and scene-level flow.'],
      ['State-estimation view', 'The useful signal is not visual copying; it is correction of missing latent dynamics.'],
      ['Best case', 'Retrieval is aligned in viewpoint, subject scale, background, and motion phase.']
    ],
    image: 'qualitative_six_case_late_panel.png'
  },
  {
    type: 'discussion',
    eyebrow: 'Discussion II',
    title: 'What can go wrong?',
    columns: [
      ['Over-trust', 'A mismatched observation can add blur, identity drift, or framing instability.'],
      ['Metric design', 'Motion quality must be reported together with spatial anchoring and physical consistency.'],
      ['Next step', 'Add retrieval confidence, viewpoint filtering, and geometry-aware scene state.']
    ],
    image: 'observation_analysis_paradigm_manual.png'
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
  },
  {
    type: 'qa',
    eyebrow: 'Q and A',
    title: 'Questions?',
    subtitle: 'Scene-assimilated video world models: retrieval as observation, attention as analysis, generation as corrected forecasting.'
  }
];

let currentSlide = 0;
let currentCase = 0;
let syncedPlayback = true;
let isPlaying = false;

const app = document.querySelector('#app');

function mediaPath(slug, kind) {
  if (kind === 'reference') return `${mediaRoot}/cases/${slug}/reference.mp4`;
  if (kind === 'anchor') return `${mediaRoot}/cases/${slug}/anchor.png`;
  return `${mediaRoot}/cases/${slug}/${kind}.mp4`;
}

function assetPath(fileName) {
  return `${mediaRoot}/assets/${fileName}`;
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
  if (slide.type === 'problemVideos') bindProblemVideoControls();
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
        <img class="cover-image" src="${assetPath('intro_concept_imagegen.png')}" alt="Conceptual video world model illustration" />
      </div>
    `;
  }

  if (slide.type === 'problemVideos') {
    return renderProblemVideos(slide);
  }

  if (slide.type === 'formula') {
    return renderFormulaSlide(slide);
  }

  if (slide.type === 'experiment') {
    return renderExperimentSlide(slide);
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
          <img src="${assetPath(slide.image)}" alt="${slide.title}" />
        </figure>
      </div>
    `;
  }

  if (slide.type === 'qualitative') {
    return renderQualitativeSlide();
  }

  if (slide.type === 'discussion') {
    return `
      <div class="discussion-layout">
        <div>
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
        </div>
        <figure class="figure-panel">
          <img src="${assetPath(slide.image)}" alt="${slide.title}" />
        </figure>
      </div>
    `;
  }

  if (slide.type === 'qa') {
    return `
      <div class="qa-layout">
        <p class="eyebrow">${slide.eyebrow}</p>
        <h2>${slide.title}</h2>
        <p>${slide.subtitle}</p>
      </div>
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

function renderProblemVideos(slide) {
  return `
    <div class="problem-layout">
      <div class="problem-copy">
        <p class="eyebrow">${slide.eyebrow}</p>
        <h2>${slide.title}</h2>
        <p class="lead">${slide.lead}</p>
        <ul class="point-list compact">${slide.points.map((point) => `<li>${point}</li>`).join('')}</ul>
        <button class="control-button problem-play" data-action="problem-play">Play examples</button>
      </div>
      <div class="example-video-grid">
        ${slide.videos.map((video) => `
          <article class="video-panel baseline">
            <div class="video-title"><span>${video.label}</span></div>
            <video src="${mediaPath(video.slug, 'baseline')}" muted loop playsinline preload="metadata"></video>
            <p>${video.note}</p>
          </article>
        `).join('')}
      </div>
    </div>
  `;
}

function renderFormulaSlide(slide) {
  return `
    <div class="formula-layout">
      <div>
        <p class="eyebrow">${slide.eyebrow}</p>
        <h2>${slide.title}</h2>
        <div class="equation-stack">
          ${slide.equations.map((equation) => `<div class="formula">${equation}</div>`).join('')}
        </div>
      </div>
      <div class="formula-explain">
        ${slide.points.map(([term, meaning]) => `
          <article>
            <h3>${term}</h3>
            <p>${meaning}</p>
          </article>
        `).join('')}
        <p class="claim">${slide.explanation}</p>
      </div>
    </div>
  `;
}

function renderExperimentSlide(slide) {
  return `
    <div class="experiment-layout">
      <div>
        <p class="eyebrow">${slide.eyebrow}</p>
        <h2>${slide.title}</h2>
        <div class="step-grid">
          ${slide.steps.map(([label, body]) => `
            <article class="step-card">
              <span>${label}</span>
              <p>${body}</p>
            </article>
          `).join('')}
        </div>
      </div>
      <div>
        <figure class="figure-panel compact-figure">
          <img src="${assetPath(slide.image)}" alt="${slide.title}" />
        </figure>
        <ul class="settings-list">
          ${slide.settings.map((setting) => `<li>${setting}</li>`).join('')}
        </ul>
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

function bindProblemVideoControls() {
  const button = app.querySelector('[data-action="problem-play"]');
  button.addEventListener('click', () => {
    const videos = [...app.querySelectorAll('video')];
    const shouldPlay = videos.some((video) => video.paused);
    videos.forEach((video) => {
      if (shouldPlay) {
        video.currentTime = 0;
        video.play();
      } else {
        video.pause();
      }
    });
    button.textContent = shouldPlay ? 'Pause examples' : 'Play examples';
  });
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
