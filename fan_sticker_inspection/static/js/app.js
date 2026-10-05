document.addEventListener('DOMContentLoaded', () => {

    // ---- DOM Elements ----
    const canvas        = document.getElementById('viewport-canvas');
    const ctx           = canvas.getContext('2d');
    const video         = document.getElementById('camera-stream');
    const placeholder   = document.getElementById('viewport-placeholder');
    const fileInput     = document.getElementById('file-input');

    const btnUpload     = document.getElementById('btn-upload');
    const btnCamera     = document.getElementById('btn-camera');
    const btnCapture    = document.getElementById('btn-capture');
    const btnSave       = document.getElementById('btn-save');
    const btnInspect    = document.getElementById('btn-inspect');
    const btnReset      = document.getElementById('btn-reset');

    const statusEl      = document.getElementById('app-status');
    const resultEmpty   = document.getElementById('result-empty');
    const resultContent = document.getElementById('result-content');
    const resultBadge   = document.getElementById('result-badge');
    const resultPred    = document.getElementById('result-prediction');
    const resultConf    = document.getElementById('result-confidence');
    const resultMarker  = document.getElementById('result-marker-info');
    const panelResult   = document.getElementById('panel-result');

    // ---- State ----
    let currentImage = null;   // HTMLImageElement
    let currentBlob  = null;   // Blob for upload
    let mediaStream  = null;
    let apiResult    = null;   // response from /api/predict
    let lastImageSource = 'UPLOAD'; // Track source

    // ---- Status Management ----
    function setStatus(state) {
        statusEl.textContent = state;
        statusEl.className = 'header-status';
        const map = {
            'READY': 'status-ready',
            'IMAGE READY': 'status-ready',
            'CAMERA READY': 'status-ready',
            'INSPECTING': 'status-inspecting',
            'PASS': 'status-pass',
            'FAIL': 'status-fail',
            'UNCERTAIN': 'status-inspecting',
            'ERROR': 'status-error',
        };
        statusEl.classList.add(map[state] || 'status-ready');
    }

    // ---- Show/Hide Helpers ----
    function showCanvas() {
        canvas.style.display = 'block';
        video.style.display = 'none';
        placeholder.style.display = 'none';
    }
    function showVideo() {
        canvas.style.display = 'none';
        video.style.display = 'block';
        placeholder.style.display = 'none';
    }
    function showPlaceholder() {
        canvas.style.display = 'none';
        video.style.display = 'none';
        placeholder.style.display = 'block';
    }

    // ---- Draw image onto canvas, optionally with actual YOLO bounding box ----
    function renderCanvas() {
        if (!currentImage) return;

        canvas.width  = currentImage.naturalWidth;
        canvas.height = currentImage.naturalHeight;
        ctx.drawImage(currentImage, 0, 0);

        if (apiResult && apiResult.result === 'PASS' && apiResult.box) {
            const b = apiResult.box;
            
            const color = '#4caf50';
            const fillColor = 'rgba(76,175,80,0.15)';

            const w = b.x2 - b.x1;
            const h = b.y2 - b.y1;

            // Draw bounding box
            ctx.beginPath();
            ctx.rect(b.x1, b.y1, w, h);
            ctx.fillStyle = fillColor;
            ctx.fill();
            
            ctx.strokeStyle = color;
            ctx.lineWidth = Math.max(3, Math.round(canvas.width / 400));
            ctx.stroke();

            // Label
            const fontSize = Math.max(16, Math.round(canvas.width / 60));
            ctx.font = `bold ${fontSize}px sans-serif`;
            
            // Draw label background for better visibility
            const labelText = `STICKER ${(apiResult.confidence * 100).toFixed(1)}%`;
            const textWidth = ctx.measureText(labelText).width;
            
            ctx.fillStyle = color;
            const labelY = b.y1 - fontSize * 0.5;
            
            if (labelY > fontSize) {
                ctx.fillRect(b.x1, b.y1 - fontSize - 8, textWidth + 10, fontSize + 8);
                ctx.fillStyle = '#000';
                ctx.textAlign = 'left';
                ctx.fillText(labelText, b.x1 + 5, b.y1 - 5);
            } else {
                // If box is too high up, put label inside/below
                ctx.fillRect(b.x1, b.y1, textWidth + 10, fontSize + 8);
                ctx.fillStyle = '#000';
                ctx.textAlign = 'left';
                ctx.fillText(labelText, b.x1 + 5, b.y1 + fontSize + 2);
            }
        }

        showCanvas();
    }

    // ---- Result Panel ----
    function showResult(data) {
        resultEmpty.style.display   = 'none';
        resultContent.style.display = 'block';

        resultBadge.textContent = data.result;
        resultPred.textContent  = data.prediction;
        
        if (data.result === 'PASS') {
            resultConf.textContent  = (data.confidence * 100).toFixed(1) + '%';
            resultMarker.textContent = 'Actual YOLO detection box drawn in green.';
        } else {
            resultConf.textContent  = '--';
            resultMarker.textContent = '';
        }

        // Color class
        resultContent.className = 'result-content';
        if (data.result === 'PASS')       resultContent.classList.add('result-pass');
        else if (data.result === 'FAIL')  resultContent.classList.add('result-fail');
        else                              resultContent.classList.add('result-uncertain');
    }

    function clearResult() {
        resultEmpty.style.display   = 'block';
        resultContent.style.display = 'none';
        resultContent.className     = 'result-content';
    }

    // ---- Camera ----
    function stopCamera() {
        if (mediaStream) {
            mediaStream.getTracks().forEach(t => t.stop());
            mediaStream = null;
        }
        video.srcObject = null;
        btnCapture.style.display = 'none';
    }

    // ---- Reset ----
    function resetAll() {
        stopCamera();
        currentImage = null;
        currentBlob  = null;
        apiResult    = null;
        fileInput.value = '';
        btnInspect.disabled = true;
        btnSave.style.display = 'none';
        clearResult();
        showPlaceholder();
        setStatus('READY');
    }

    // ========== EVENT HANDLERS ==========

    // Upload
    btnUpload.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;

        stopCamera();
        apiResult = null;
        currentBlob = file;
        lastImageSource = 'UPLOAD';

        const reader = new FileReader();
        reader.onload = (ev) => {
            const img = new Image();
            img.onload = () => {
                currentImage = img;
                renderCanvas();
                btnInspect.disabled = false;
                btnSave.style.display = '';
                clearResult();
                setStatus('IMAGE READY');
            };
            img.src = ev.target.result;
        };
        reader.readAsDataURL(file);
    });

    // Camera
    btnCamera.addEventListener('click', async () => {
        resetAll();
        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment', width: { ideal: 1280 } }
            });
            video.srcObject = mediaStream;
            showVideo();
            btnCapture.style.display = '';
            btnInspect.disabled = true;
            btnSave.style.display = 'none';
            setStatus('CAMERA READY');
        } catch (err) {
            alert('Camera access denied or not available.\n' + err.message);
            setStatus('ERROR');
        }
    });

    // Capture
    btnCapture.addEventListener('click', () => {
        if (!mediaStream) return;

        // Grab frame
        canvas.width  = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);

        canvas.toBlob((blob) => {
            currentBlob = blob;
            lastImageSource = 'CAMERA';
            const img = new Image();
            img.onload = () => {
                currentImage = img;
                apiResult = null;
                stopCamera();
                renderCanvas();
                btnInspect.disabled = false;
                btnSave.style.display = '';
                clearResult();
                setStatus('IMAGE READY');
            };
            img.src = URL.createObjectURL(blob);
        }, 'image/jpeg', 0.92);
    });

    // Save Image
    btnSave.addEventListener('click', () => {
        if (!currentBlob) return;
        const url = URL.createObjectURL(currentBlob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `captured_${Date.now()}.jpg`;
        document.body.appendChild(a);
        a.click();
        URL.revokeObjectURL(url);
        a.remove();
    });

    // Inspect
    btnInspect.addEventListener('click', async () => {
        if (!currentBlob) return;

        setStatus('INSPECTING');
        btnInspect.disabled = true;

        const formData = new FormData();
        formData.append('image', currentBlob, 'inspection.jpg');
        formData.append('source', lastImageSource);

        try {
            const resp = await fetch('/api/predict', { method: 'POST', body: formData });

            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                throw new Error(err.error || `HTTP ${resp.status}`);
            }

            const data = await resp.json();
            if (data.error) throw new Error(data.error);

            apiResult = data;
            renderCanvas();
            showResult(data);
            setStatus(data.result);

        } catch (err) {
            console.error('Inspection error:', err);
            alert('Inspection failed: ' + err.message);
            setStatus('ERROR');
            btnInspect.disabled = false;
        }
    });

    // Reset
    btnReset.addEventListener('click', resetAll);

    // Initial state
    setStatus('READY');
});
