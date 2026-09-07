/*
  script.js — AI Document Summarization System
  =============================================
  Frontend logic for the Web-Based AI Document Summarization System.

  Features:
  - Drag-and-drop & file input validation (PDF, DOCX, TXT up to 20 MB)
  - Summary length: Short, Medium, Long (placeholder shown, user must select)
  - Download options accordion
  - Asynchronous upload + polling with 4-step vertical progress stepper
  - Key Topics keyword chips
  - Key Highlights bullet list
  - Structured summary rendering (## Headings, **bold** terms, paragraphs)
  - Text-to-Speech (Read Aloud)
  - Code listings with copy buttons
  - Extracted figures gallery
  - Download in TXT, PDF, DOCX (placeholder shown, user must select)
  - Copy to clipboard & Print

  FIXES (2026-07-27):
  Bug H — resetBtn now clears lengthSelect.value and restores placeholder class
           so the dropdown shows "Choose Summary Length…" again after reset.
  Bug I — Stepper stage mapping corrected: 'extracting' maps to step index 1
           (was incorrectly mapping to 0, so step 2 was never shown active).
  Bug J — cancelBtn.disabled is always reset in a .finally() handler so
           a failed cancel fetch doesn't leave the button permanently greyed.
  Bug K — pollJobStatus tracks consecutive network errors and stops polling
           after 5 failures, then shows a user-visible error.
  Bug L — activeJobId is reset to null in the resetBtn handler so that
           subsequent jobs don't reuse the old job's image URLs.
  Bug F — Download format validation now marks the select as aria-invalid
           and shows a visible hint text when the user clicks Download
           without choosing a format.
*/

'use strict';

document.addEventListener('DOMContentLoaded', function() {

  // --- DOM Elements ---
  const dropzone        = document.getElementById('dropzone');
  const fileInput       = document.getElementById('fileInput');
  const validationMsg   = document.getElementById('validationMsg');
  const fileChip        = document.getElementById('fileChip');
  const fileChipName    = document.getElementById('fileChipName');
  const fileChipMeta    = document.getElementById('fileChipMeta');
  const fileTypeIcon    = document.getElementById('fileTypeIcon');
  const fileChipRemove  = document.getElementById('fileChipRemove');

  const lengthSelect    = document.getElementById('lengthSelect');
  const lengthHint      = document.getElementById('lengthHint');
  const summarizeBtn    = document.getElementById('summarizeBtn');
  const btnHint         = document.getElementById('btnHint');

  const advancedToggle  = document.getElementById('advancedToggle');
  const advancedBody    = document.getElementById('advancedBody');

  const progressPanel   = document.getElementById('progressPanel');
  const progressFill    = document.getElementById('progressFill');
  const progressLabel   = document.getElementById('progressLabel');
  const progressEta     = document.getElementById('progressEta');
  const cancelBtn       = document.getElementById('cancelBtn');
  const progressTrack   = document.getElementById('progressTrack');

  const errorPanel      = document.getElementById('errorPanel');
  const errorPanelText  = document.getElementById('errorPanelText');
  const errorCloseBtn   = document.getElementById('errorCloseBtn');

  const resultPanel     = document.getElementById('resultPanel');
  const statOrigWords   = document.getElementById('statOrigWords');
  const statSumWords    = document.getElementById('statSumWords');
  const statReduction   = document.getElementById('statReduction');
  const statTimeSaved   = document.getElementById('statTimeSaved');

  const keywordsBox     = document.getElementById('keywordsBox');
  const keywordsChips   = document.getElementById('keywordsChips');
  const highlightsBox   = document.getElementById('highlightsBox');
  const highlightsList  = document.getElementById('highlightsList');
  const resultText      = document.getElementById('resultText');
  const figuresBox      = document.getElementById('figuresBox');
  const figuresGrid     = document.getElementById('figuresGrid');
  const codeBox         = document.getElementById('codeBox');
  const codeList        = document.getElementById('codeList');

  const copyBtn         = document.getElementById('copyBtn');
  const printBtn        = document.getElementById('printBtn');
  const previewBtn      = document.getElementById('previewBtn');
  const resetBtn        = document.getElementById('resetBtn');
  const downloadBtn     = document.getElementById('downloadBtn');
  const downloadFormat  = document.getElementById('downloadFormat');

  // --- State ---
  let selectedFile      = null;
  let activeJobId       = null;  // reset to null on job complete/cancel/reset
  let pollInterval      = null;
  let pollErrorCount    = 0;     // Bug K: track consecutive network errors
  const MAX_POLL_ERRORS = 5;
  let startTime         = 0;
  let currentResult     = null;

  // --- Length Selector ---
  if (lengthSelect) {
    lengthSelect.addEventListener('change', function() {
      const val = lengthSelect.value;
      if (val) {
        lengthSelect.classList.remove('option-select--placeholder');
        lengthSelect.removeAttribute('aria-invalid');
      }
      clearLengthError();
      checkReadyState();
    });
  }

  // --- 3. File Validation & Upload UI ---
  function validateFile(file) {
    if (!file) return false;
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['pdf', 'docx', 'txt'].includes(ext)) {
      showError('Invalid file format. Please choose a PDF, DOCX, or TXT file.');
      return false;
    }
    if (file.size > 20 * 1024 * 1024) {
      showError('File size exceeds the 20 MB limit. Please upload a smaller file.');
      return false;
    }
    if (file.size === 0) {
      showError('The selected file is empty.');
      return false;
    }
    clearError();
    return true;
  }

  function handleFileSelection(file) {
    if (!validateFile(file)) return;
    selectedFile = file;
    if (fileChip && fileChipName && fileChipMeta && fileTypeIcon) {
      const ext = file.name.split('.').pop().toUpperCase();
      fileTypeIcon.textContent = ext;
      fileChipName.textContent = file.name;
      const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
      fileChipMeta.textContent = `${ext} · ${sizeMB} MB`;
      fileChip.hidden = false;
    }
    checkReadyState();
  }

  function checkReadyState() {
    const hasFile   = selectedFile !== null;
    const hasLength = lengthSelect && lengthSelect.value !== '';
    const isReady   = hasFile && hasLength;
    if (summarizeBtn) {
      summarizeBtn.disabled = !isReady;
    }
  }

  function showLengthError() {
    if (lengthSelect) {
      lengthSelect.setAttribute('aria-invalid', 'true');
      lengthSelect.classList.add('option-select--error');
    }
  }

  function clearLengthError() {
    if (lengthSelect) {
      lengthSelect.removeAttribute('aria-invalid');
      lengthSelect.classList.remove('option-select--error');
    }
  }

  // Dropzone Events
  if (dropzone) {
    dropzone.addEventListener('click', () => fileInput && fileInput.click());
    dropzone.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        fileInput && fileInput.click();
      }
    });
    ['dragenter', 'dragover'].forEach(evt => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.classList.add('dropzone--dragover');
      });
    });
    ['dragleave', 'drop'].forEach(evt => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.classList.remove('dropzone--dragover');
      });
    });
    dropzone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length > 0) handleFileSelection(files[0]);
    });
  }

  if (fileInput) {
    fileInput.addEventListener('change', () => {
      if (fileInput.files && fileInput.files.length > 0) {
        handleFileSelection(fileInput.files[0]);
      }
    });
  }

  if (fileChipRemove) {
    fileChipRemove.addEventListener('click', () => {
      selectedFile = null;
      if (fileInput) fileInput.value = '';
      if (fileChip) fileChip.hidden = true;
      checkReadyState();
    });
  }

  // --- 4. 4-Step Vertical Progress Stepper ---
  // Maps backend status values to step IDs.
  // Steps: reading → extracting → summarizing → done
  const STEP_IDS = ['reading', 'extracting', 'summarizing', 'done'];

  function setStepState(stepId, state) {
    // state: 'pending' | 'active' | 'done'
    const el = document.getElementById(`step-${stepId}`);
    if (!el) return;
    el.classList.remove('step--pending', 'step--active', 'step--done');
    el.classList.add(`step--${state}`);
  }

  function updateStepperToStage(backendStatus) {
    // FIX Bug I: Map backend status to the correct step index.
    // Previous mapping had 'extracting' → index 0 (same as 'reading'),
    // so step 2 ("Analysing Content") was never shown as active.
    //
    // Correct mapping:
    //   extracting  → index 0 active (step 1: Reading Document)
    //   queued      → index 1 active (step 2: Analysing Content)
    //   summarizing → index 2 active (step 3: Generating Summary)
    //   condensing  → index 2 active (step 3: Generating Summary)
    //   done        → index 4       (all steps done)
    let activeIndex;
    if (backendStatus === 'extracting') {
      activeIndex = 0;   // Step 1: Reading Document
    } else if (backendStatus === 'queued') {
      activeIndex = 1;   // Step 2: Analysing Content
    } else if (backendStatus === 'summarizing' || backendStatus === 'reading') {
      activeIndex = 2;   // Step 3: Generating Summary
    } else if (backendStatus === 'condensing' || backendStatus === 'finalizing') {
      activeIndex = 3;   // Step 4: Finalizing Result
    } else if (backendStatus === 'done') {
      activeIndex = 4;   // All done
    } else {
      activeIndex = 1;   // Default: step 2
    }

    STEP_IDS.forEach((id, idx) => {
      if (idx < activeIndex) setStepState(id, 'done');
      else if (idx === activeIndex) setStepState(id, 'active');
      else setStepState(id, 'pending');
    });
  }

  // Initialize steps as pending
  STEP_IDS.forEach(id => setStepState(id, 'pending'));

  // --- 5. Job Submission & Polling ---
  if (summarizeBtn) {
    summarizeBtn.addEventListener('click', startSummarization);
  }

  function startSummarization() {
    if (!selectedFile) return;

    // Validate length selection — show inline error if not chosen
    if (!lengthSelect || !lengthSelect.value) {
      showLengthError();
      if (lengthSelect) lengthSelect.focus();
      return;
    }

    clearError();
    clearLengthError();
    stopTTS();
    if (resultPanel) resultPanel.hidden = true;

    // Reset polling state
    pollErrorCount = 0;

    const formData = new FormData();
    formData.append('document', selectedFile);
    formData.append('summary_length', lengthSelect.value);

    // Show Progress UI
    STEP_IDS.forEach(id => setStepState(id, 'pending'));
    setStepState('reading', 'active');
    if (progressPanel) progressPanel.hidden = false;
    updateProgressUI('Reading document…', 5, '');
    startTime = Date.now();
    summarizeBtn.disabled = true;

    fetch('/summarize', {
      method: 'POST',
      body: formData,
    })
    .then(res => res.json().then(data => ({ status: res.status, data })))
    .then(({ status, data }) => {
      if (status !== 200) throw new Error(data.error || 'Failed to start summarization.');
      activeJobId = data.job_id;
      pollInterval = setInterval(pollJobStatus, 1000);
    })
    .catch(err => {
      if (progressPanel) progressPanel.hidden = true;
      showError(err.message);
      checkReadyState();
    });
  }

  function pollJobStatus() {
    if (!activeJobId) return;

    fetch(`/status/${activeJobId}`)
      .then(res => res.json())
      .then(job => {
        // Successful response — reset the consecutive error counter
        pollErrorCount = 0;

        if (job.error && !job.status) {
          // Server returned an error object with no status field
          clearInterval(pollInterval);
          if (progressPanel) progressPanel.hidden = true;
          showError(job.error);
          checkReadyState();
          return;
        }

        const status   = job.status;
        const detail   = job.stage_detail || 'Processing…';

        if (status === 'extracting') {
          updateStepperToStage('extracting');
          updateProgressUI(detail, 15, calculateEta(15));

        } else if (status === 'queued') {
          updateStepperToStage('queued');
          updateProgressUI(detail, 10, calculateEta(10));

        } else if (status === 'reading') {
          updateStepperToStage('reading');
          const cur = job.progress_current || 1;
          const tot = job.progress_total  || 1;
          const pct = Math.min(30, 15 + Math.round((cur / tot) * 15));
          updateProgressUI(detail, pct, calculateEta(pct));

        } else if (status === 'summarizing') {
          updateStepperToStage('summarizing');
          const cur = job.progress_current || 1;
          const tot = job.progress_total  || 1;
          const pct = Math.min(85, 30 + Math.round((cur / tot) * 55));
          if (progressLabel) progressLabel.textContent = detail;
          updateProgressUI(detail, pct, calculateEta(pct));

        } else if (status === 'condensing' || status === 'finalizing') {
          updateStepperToStage('condensing');
          const cur = job.progress_current || 1;
          const tot = job.progress_total  || 1;
          const pct = Math.min(95, 85 + Math.round((cur / tot) * 10));
          if (progressLabel) progressLabel.textContent = detail;
          updateProgressUI(detail, pct, calculateEta(pct));

        } else if (status === 'done') {
          clearInterval(pollInterval);
          // Mark all steps done
          STEP_IDS.forEach(id => setStepState(id, 'done'));
          updateProgressUI('Summary ready!', 100, '');
          setTimeout(() => {
            if (progressPanel) progressPanel.hidden = true;
            displayResults(job);
          }, 800);

        } else if (status === 'cancelled') {
          clearInterval(pollInterval);
          if (progressPanel) progressPanel.hidden = true;
          STEP_IDS.forEach(id => setStepState(id, 'pending'));
          if (window.showToast) window.showToast('Summarization cancelled.', 'info');
          checkReadyState();

        } else if (status === 'error') {
          clearInterval(pollInterval);
          if (progressPanel) progressPanel.hidden = true;
          showError(job.error || 'An error occurred during summarization.');
          checkReadyState();
        }
      })
      .catch(() => {
        // FIX Bug K: Track consecutive network errors.
        // After MAX_POLL_ERRORS consecutive failures, stop polling and
        // tell the user — previously the catch block was empty and
        // polling ran forever with silent failures.
        pollErrorCount++;
        if (pollErrorCount >= MAX_POLL_ERRORS) {
          clearInterval(pollInterval);
          if (progressPanel) progressPanel.hidden = true;
          showError('Lost connection to the server. Please check your network and try again.');
          checkReadyState();
        }
      });
  }

  function calculateEta(percent) {
    if (percent <= 5) return 'Estimating time…';
    const elapsedSeconds = (Date.now() - startTime) / 1000;
    const totalEstimatedSeconds = elapsedSeconds / (percent / 100);
    const remainingSeconds = Math.max(0, Math.round(totalEstimatedSeconds - elapsedSeconds));
    if (remainingSeconds < 10) return 'Almost done…';
    if (remainingSeconds < 60) return `~${remainingSeconds}s remaining`;
    const mins = Math.ceil(remainingSeconds / 60);
    return `~${mins} min remaining`;
  }

  function updateProgressUI(label, percent, etaText) {
    if (progressLabel) progressLabel.textContent = label;
    if (progressFill) progressFill.style.width = percent + '%';
    if (progressTrack) progressTrack.setAttribute('aria-valuenow', percent);
    if (progressEta) progressEta.textContent = etaText;
  }

  if (cancelBtn) {
    cancelBtn.addEventListener('click', function() {
      if (!activeJobId) return;
      cancelBtn.disabled = true;
      fetch(`/cancel/${activeJobId}`, { method: 'POST' })
        .then(() => {
          clearInterval(pollInterval);
          if (progressPanel) progressPanel.hidden = true;
          STEP_IDS.forEach(id => setStepState(id, 'pending'));
          checkReadyState();
        })
        .catch(() => {
          // FIX Bug J: Always re-enable the cancel button, even on fetch error.
          // Previously a network error on the cancel request left the button
          // permanently disabled.
          if (window.showToast) window.showToast('Could not reach server to cancel. Please wait.', 'info');
        })
        .finally(() => {
          cancelBtn.disabled = false;
        });
    });
  }

  function cleanDisplayString(str) {
    if (!str) return '';
    return str
      .replace(/[\u25A0-\u25FF\u2B1B\u2B1C\uE000-\uF8FF]/g, '')
      .replace(/[■□▪▫◾◽◼◻⬛⬜▮▯▬▲►▼◄◊○●◦]/g, '')
      .replace(/[■▪•●]\s*\d{1,4}\b/g, '')
      .replace(/\s+/g, ' ')
      .trim();
  }

  // --- 6. Rendering Results ---
  function displayResults(job) {
    currentResult = job;
    if (resultPanel) {
      resultPanel.hidden = false;
      // Trigger fade-in by removing/re-adding class
      resultPanel.classList.remove('result-visible');
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          resultPanel.classList.add('result-visible');
        });
      });
    }

    // Statistics
    if (statOrigWords) statOrigWords.textContent = (job.original_word_count || 0).toLocaleString();
    if (statSumWords)  statSumWords.textContent  = (job.summary_word_count || 0).toLocaleString();
    if (statReduction) statReduction.textContent  = `${job.reduction_percent || 0}%`;
    const timeSaved = Math.max(1, (job.original_reading_time || 0) - (job.summary_reading_time || 0));
    if (statTimeSaved) statTimeSaved.textContent = timeSaved;

    // ── 1. Document Information ──────────────────────────────────────────────
    const docInfoBox  = document.getElementById('docInfoBox');
    const docInfoList = document.getElementById('docInfoList');
    if (docInfoBox && docInfoList) {
      docInfoList.innerHTML = '';
      const meta = job.doc_metadata || {};
      const fieldLabels = [
        ['title',       'Title'],
        ['course_code', 'Course Code'],
        ['course_name', 'Course / Subject'],
        ['author',      'Author / Student'],
        ['lecturer',    'Lecturer / Instructor'],
        ['department',  'Department'],
        ['university',  'Institution'],
        ['date',        'Date / Semester'],
      ];
      let metaCount = 0;
      fieldLabels.forEach(([key, label]) => {
        if (meta[key]) {
          const dt = document.createElement('dt');
          dt.className = 'doc-info-label';
          dt.textContent = label;
          const dd = document.createElement('dd');
          dd.className = 'doc-info-value';
          dd.textContent = meta[key];
          docInfoList.appendChild(dt);
          docInfoList.appendChild(dd);
          metaCount++;
        }
      });
      // Also show the source filename if available
      if (job.suggested_filename) {
        const dt = document.createElement('dt');
        dt.className = 'doc-info-label';
        dt.textContent = 'Source File';
        const dd = document.createElement('dd');
        dd.className = 'doc-info-value';
        dd.textContent = job.suggested_filename;
        docInfoList.appendChild(dt);
        docInfoList.appendChild(dd);
        metaCount++;
      }
      docInfoBox.hidden = (metaCount === 0);
    }

    // ── 2. Key Topics (keywords) ─────────────────────────────────────────────
    if (keywordsBox && keywordsChips) {
      keywordsChips.innerHTML = '';
      const kwList = job.keywords || [];
      if (kwList.length > 0) {
        kwList.forEach(kw => {
          const cleanKw = cleanDisplayString(kw);
          if (cleanKw) {
            const span = document.createElement('span');
            span.className = 'keyword-chip';
            span.textContent = cleanKw;
            keywordsChips.appendChild(span);
          }
        });
        keywordsBox.hidden = false;
      } else {
        keywordsBox.hidden = true;
      }
    }

    // ── 3. Key Highlights ────────────────────────────────────────────────────
    if (highlightsBox && highlightsList) {
      highlightsList.innerHTML = '';
      const hs = job.highlights || [];
      if (hs.length > 0) {
        hs.forEach(h => {
          const cleanH = cleanDisplayString(h);
          if (cleanH) {
            const li = document.createElement('li');
            li.textContent = cleanH;
            highlightsList.appendChild(li);
          }
        });
        highlightsBox.hidden = false;
      } else {
        highlightsBox.hidden = true;
      }
    }

    // ── 4. Summary Text (clean — no metadata mixed in) ───────────────────────
    if (resultText) {
      resultText.innerHTML = formatSummaryHTML(job.summary || '');
    }

    // Figures & Diagrams
    if (figuresBox && figuresGrid) {
      figuresGrid.innerHTML = '';
      const imgs = job.images_meta || [];
      if (imgs.length > 0 && activeJobId) {
        imgs.forEach((img, idx) => {
          const figure  = document.createElement('figure');
          figure.className = 'figure-item';
          const imgEl   = document.createElement('img');
          imgEl.src     = `/image/${activeJobId}/${idx}`;
          imgEl.alt     = `Extracted figure from page ${img.page}`;
          imgEl.loading = 'lazy';
          const caption = document.createElement('figcaption');
          caption.textContent = `Page ${img.page}`;
          figure.appendChild(imgEl);
          figure.appendChild(caption);
          figuresGrid.appendChild(figure);
        });
        figuresBox.hidden = false;
      } else {
        figuresBox.hidden = true;
      }
    }

    // Code Listings
    if (codeBox && codeList) {
      codeList.innerHTML = '';
      const blocks = job.code_blocks || [];
      if (blocks.length > 0) {
        blocks.forEach((codeText, idx) => {
          const wrapper = document.createElement('div');
          wrapper.className = 'code-listing-wrapper';
          const row   = document.createElement('div');
          row.className = 'code-listing-label-row';
          const label = document.createElement('span');
          label.className = 'code-listing-label';
          label.textContent = `Code Block ${idx + 1}`;
          const copyCodeBtn = document.createElement('button');
          copyCodeBtn.type = 'button';
          copyCodeBtn.className = 'btn-copy-code';
          copyCodeBtn.textContent = 'Copy';
          copyCodeBtn.addEventListener('click', () => {
            navigator.clipboard.writeText(codeText).then(() => {
              copyCodeBtn.textContent = 'Copied!';
              setTimeout(() => { copyCodeBtn.textContent = 'Copy'; }, 2000);
            });
          });
          row.appendChild(label);
          row.appendChild(copyCodeBtn);
          const pre = document.createElement('pre');
          pre.className = 'code-listing';
          pre.textContent = codeText;
          wrapper.appendChild(row);
          wrapper.appendChild(pre);
          codeList.appendChild(wrapper);
        });
        codeBox.hidden = false;
      } else {
        codeBox.hidden = true;
      }
    }

    // Set default download format to PDF
    if (downloadFormat && !downloadFormat.value) {
      downloadFormat.value = 'pdf';
    }

    // Scroll to results
    resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    if (window.showToast) window.showToast('Summary generated successfully!', 'success');

    // ── Save live metrics to localStorage for the Performance page ──────────
    try {
      const metrics = {
        filename:        job.filename        || 'Unknown document',
        originalWords:   job.original_word_count || 0,
        summaryWords:    job.summary_word_count  || 0,
        reduction:       job.reduction_percent   || 0,
        originalReading: job.original_reading_time || 0,
        summaryReading:  job.summary_reading_time  || 0,
        sections:        job.section_count        || 0,
        date:            new Date().toLocaleString(),
      };
      localStorage.setItem('lastSummaryMetrics', JSON.stringify(metrics));
    } catch (_) { /* localStorage unavailable — silently skip */ }
  }

  // --- 7. Summary HTML Formatter ---
  // Handles:
  //   ## Heading        → <h3 class="summary-heading">
  //   ### Sub-heading   → <h4 class="summary-subheading">
  //   ---               → <hr class="summary-divider"> + closes metadata block
  //   **bold text**     → <strong>
  //   - bullet item     → <li> in <ul class="summary-bullet-list">
  //   1. numbered item  → <li> in <ol class="summary-numbered-list">
  //   Regular text      → <p class="summary-paragraph">
  function formatSummaryHTML(text) {
    if (!text) return '<p class="summary-empty">No summary available.</p>';

    const lines = text.split('\n');
    let html        = '';
    let inList      = false;   // ul active
    let inOl        = false;   // ol active
    let inMetaBlock = true;    // content before first --- is metadata

    if (inMetaBlock) html += '<div class="summary-meta-block">';

    function closeListsIfOpen() {
      if (inList) { html += '</ul>'; inList = false; }
      if (inOl)   { html += '</ol>'; inOl   = false; }
    }

    lines.forEach(line => {
      const trimmed = line.trim();

      if (!trimmed) {
        closeListsIfOpen();
        return; // skip blank lines
      }

      if (trimmed.startsWith('#### ')) {
        closeListsIfOpen();
        html += `<h5 class="summary-subheading summary-subheading--deep">${applyInlineMarkdown(trimmed.slice(5))}</h5>`;

      } else if (trimmed.startsWith('### ')) {
        closeListsIfOpen();
        html += `<h4 class="summary-subheading">${applyInlineMarkdown(trimmed.slice(4))}</h4>`;

      } else if (trimmed.startsWith('## ')) {
        closeListsIfOpen();
        html += `<h3 class="summary-heading">${applyInlineMarkdown(trimmed.slice(3))}</h3>`;

      } else if (trimmed === '---') {
        closeListsIfOpen();
        if (inMetaBlock) {
          html += '</div>'; // close summary-meta-block
          inMetaBlock = false;
        }
        html += '<hr class="summary-divider" aria-hidden="true">';

      } else if (trimmed.startsWith('- ')) {
        if (inOl) { html += '</ol>'; inOl = false; }
        if (!inList) { html += '<ul class="summary-bullet-list">'; inList = true; }
        html += `<li>${applyInlineMarkdown(trimmed.slice(2))}</li>`;

      } else if (/^\d+\.\s/.test(trimmed)) {
        // Numbered list item: "1. item text"
        if (inList) { html += '</ul>'; inList = false; }
        if (!inOl)  { html += '<ol class="summary-numbered-list">'; inOl = true; }
        const itemText = trimmed.replace(/^\d+\.\s+/, '');
        html += `<li>${applyInlineMarkdown(itemText)}</li>`;

      } else {
        closeListsIfOpen();
        html += `<p class="summary-paragraph">${applyInlineMarkdown(trimmed)}</p>`;
      }
    });

    closeListsIfOpen();
    if (inMetaBlock) html += '</div>'; // close if no divider was found
    return html;
  }

  // Convert **bold** markers to <strong> inline
  function applyInlineMarkdown(str) {
    const escaped = escapeHtml(str);
    // Replace **text** with <strong>text</strong>
    return escaped.replace(/\*\*(.+?)\*\*/g, '<strong class="summary-bold">$1</strong>');
  }

  function escapeHtml(str) {
    const cleaned = str
      .replace(/\uF0B7/g, '•')
      .replace(/[■□▪▫◾◽◼◻⬛⬜▮▯▬•●]\s*\d{1,4}\b/g, '')
      .replace(/[\u25A0-\u25FF\u2B1B\u2B1C\u25AE\u25AF\u25AC]/g, '')
      .replace(/[■□▪▫◾◽◼◻⬛⬜▮▯▬▲►▼◄◊○●◦]/g, '')
      .replace(/[\uE000-\uF8FF]/g, '')
      .replace(/[\u2013\u2014\u2212\u2011]/g, '-')
      .replace(/[\u201C\u201D]/g, '"')
      .replace(/[\u2018\u2019]/g, "'")
      .replace(/[\u2190\u2191\u2192\u2193\u2194\u21D2\u21E2\u27A1\u2793\u2794\u279E]/g, '')
      .replace(/(?<![A-Za-z0-9])®+(?=\s|$)/g, '')
      .replace(/\uFFFD/g, '');
    return cleaned
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }


  // --- 8. Copy helper & plain-text extraction ---
  function getSummaryPlainText() {
    if (!currentResult) return '';
    // Strip markdown markers so clipboard output is clean plain text
    return (currentResult.summary || '')
      .replace(/##+ /g, '')
      .replace(/\*\*/g, '')
      .replace(/^- /gm, '')
      .trim();
  }

  // stopTTS() is a no-op stub kept for compatibility with reset/submit handlers
  function stopTTS() {}


  // --- 9. Preview Modal ---
  const previewModal       = document.getElementById('summaryPreviewModal');
  const previewModalBody   = document.getElementById('previewModalBody');
  const previewModalClose  = document.getElementById('previewModalClose');
  const previewModalCloseB = document.getElementById('previewModalCloseBtn');
  const previewModalBdrop  = document.getElementById('previewModalBackdrop');
  const previewWordCount   = document.getElementById('previewWordCount');

  function openPreviewModal() {
    if (!currentResult || !previewModal) return;
    previewModalBody.innerHTML = formatSummaryHTML(currentResult.summary || '');
    const wc = (currentResult.summary || '').split(/\s+/).filter(Boolean).length;
    if (previewWordCount) previewWordCount.textContent = `${wc.toLocaleString()} words`;
    previewModal.hidden = false;
    document.body.style.overflow = 'hidden';
    previewModalClose.focus();
  }

  function closePreviewModal() {
    if (!previewModal) return;
    previewModal.hidden = true;
    document.body.style.overflow = '';
    if (previewBtn) previewBtn.focus();
  }

  if (previewBtn)      previewBtn.addEventListener('click', openPreviewModal);
  if (previewModalClose)  previewModalClose.addEventListener('click', closePreviewModal);
  if (previewModalCloseB) previewModalCloseB.addEventListener('click', closePreviewModal);
  if (previewModalBdrop)  previewModalBdrop.addEventListener('click', closePreviewModal);
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && previewModal && !previewModal.hidden) closePreviewModal();
  });


  // --- 10. Actions: Copy, Print, Download, Reset ---
  if (copyBtn) {
    copyBtn.addEventListener('click', function() {
      if (!currentResult || !currentResult.summary) return;
      const plainText = getSummaryPlainText();
      navigator.clipboard.writeText(plainText).then(() => {
        copyBtn.textContent = 'Copied!';
        if (window.showToast) window.showToast('Summary copied to clipboard.', 'success');
        setTimeout(() => {
          copyBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg> Copy`;
        }, 2000);
      }).catch(() => {
        if (window.showToast) window.showToast('Could not copy to clipboard.', 'info');
      });
    });
  }

  if (printBtn) {
    printBtn.addEventListener('click', function() { window.print(); });
  }

  if (downloadBtn) {
    downloadBtn.addEventListener('click', function() {
      const jobId = activeJobId || (currentResult && currentResult.job_id);
      if (!jobId) {
        if (window.showToast) window.showToast('No summary available to download.', 'info');
        return;
      }
      const fmt = (downloadFormat && downloadFormat.value) ? downloadFormat.value : 'pdf';
      const originalHtml = downloadBtn.innerHTML;

      downloadBtn.disabled = true;
      downloadBtn.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="spin" aria-hidden="true"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg> Downloading…`;
      if (window.showToast) window.showToast(`Preparing ${fmt.toUpperCase()} download…`, 'info');

      const downloadUrl = `/download/${jobId}?format=${encodeURIComponent(fmt)}`;
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.setAttribute('download', '');
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      setTimeout(() => {
        downloadBtn.disabled = false;
        downloadBtn.innerHTML = originalHtml;
      }, 1500);
    });
  }

  if (resetBtn) {
    resetBtn.addEventListener('click', function() {
      stopTTS();
      selectedFile = null;
      activeJobId  = null;   // FIX Bug L: Reset job ID so stale images aren't reused
      currentResult = null;
      if (fileInput) fileInput.value = '';
      if (fileChip) fileChip.hidden = true;
      if (resultPanel) {
        resultPanel.hidden = true;
        resultPanel.classList.remove('result-visible');
      }
      // FIX Bug H: Reset the length dropdown to the placeholder state so
      // the user must make a deliberate selection before the next job.
      if (lengthSelect) {
        lengthSelect.value = '';
        lengthSelect.classList.add('option-select--placeholder');
        // Clear saved pref so next load also starts fresh
        localStorage.removeItem('pref_length');
      }
      if (lengthHint) {
        lengthHint.textContent = '';
        lengthHint.classList.remove('option-hint--error');
      }
      clearLengthError();
      clearError();
      checkReadyState();
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }

  // --- 10. Error Helpers ---
  function showError(msg) {
    if (errorPanelText && errorPanel) {
      errorPanelText.textContent = msg;
      errorPanel.hidden = false;
      // Scroll error into view if it's off-screen
      errorPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  function clearError() {
    if (errorPanel) errorPanel.hidden = true;
  }

  if (errorCloseBtn) {
    errorCloseBtn.addEventListener('click', clearError);
  }

  // --- Initial State ---
  checkReadyState();

});
