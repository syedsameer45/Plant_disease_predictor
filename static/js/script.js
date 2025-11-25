// static/js/script.js - FIXED VERSION

class PlantDiseaseDetector {
    constructor() {
        this.currentFile = null;
        this.lastResponse = null;
        this.currentLanguage = 'te';
        this.initializeApp();
    }

    initializeApp() {
        this.bindEvents();
        this.loadHistory();
        console.log('🌿 Plant Disease Detector Initialized');
    }

    bindEvents() {
        // File upload events
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const analyzeBtn = document.getElementById('analyzeBtn');

        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        uploadArea.addEventListener('drop', this.handleFileDrop.bind(this));
        
        fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        analyzeBtn.addEventListener('click', this.analyzeImage.bind(this));

        // Language tabs
        document.querySelectorAll('.language-tab').forEach(tab => {
            tab.addEventListener('click', (e) => this.switchLanguage(e.target.dataset.lang));
        });

        // Action buttons
        document.getElementById('speakBtn').addEventListener('click', this.speakTreatment.bind(this));
        document.getElementById('saveBtn').addEventListener('click', this.saveToHistory.bind(this));
        document.getElementById('downloadBtn').addEventListener('click', this.downloadHeatmap.bind(this));
    }

    handleDragOver(e) {
        e.preventDefault();
        document.getElementById('uploadArea').classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        document.getElementById('uploadArea').classList.remove('dragover');
    }

    handleFileDrop(e) {
        e.preventDefault();
        document.getElementById('uploadArea').classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        if (e.target.files.length > 0) {
            this.processFile(e.target.files[0]);
        }
    }

    processFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('❌ Please select an image file (JPEG, PNG, etc.)');
            return;
        }

        this.currentFile = file;
        
        // Update file info
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('fileSize').textContent = this.formatFileSize(file.size);
        document.getElementById('fileType').textContent = file.type;
        document.getElementById('fileInfo').style.display = 'block';

        // Enable analyze button
        document.getElementById('analyzeBtn').disabled = false;

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            document.getElementById('previewImage').src = e.target.result;
        };
        reader.readAsDataURL(file);

        console.log('✅ File selected:', file.name);
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async analyzeImage() {
        if (!this.currentFile) {
            alert('❌ Please select an image first.');
            return;
        }

        const analyzeBtn = document.getElementById('analyzeBtn');
        const loadingText = document.getElementById('loadingText');

        analyzeBtn.disabled = true;
        analyzeBtn.textContent = '⏳ Analyzing...';
        loadingText.style.display = 'block';

        try {
            const formData = new FormData();
            formData.append('file', this.currentFile);
            formData.append('lang', this.currentLanguage);

            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();
            this.lastResponse = data;
            this.displayResults(data);
            
            console.log('✅ Analysis completed:', data);

        } catch (error) {
            console.error('❌ Analysis failed:', error);
            alert('Analysis failed: ' + error.message);
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = '🔍 Analyze Image';
            loadingText.style.display = 'none';
        }
    }

    displayResults(data) {
        // Show result section
        document.getElementById('resultSection').classList.add('active');

        // Basic info
        document.getElementById('diseaseTitle').textContent = data.disease || 'Unknown';
        document.getElementById('confidenceScore').textContent = 
            typeof data.confidence === 'number' ? (data.confidence * 100).toFixed(1) + '%' : 'N/A';

        // Treatment data
        const treatment = data.treatment || {};
        
        // Summary
        document.getElementById('treatmentSummary').textContent = 
            treatment.treatment_summary || 'No treatment information available.';

        // Steps
        const stepsList = document.getElementById('treatmentSteps');
        stepsList.innerHTML = '';
        const steps = treatment.step_by_step || [];
        if (steps.length > 0) {
            steps.forEach(step => {
                const li = document.createElement('li');
                li.textContent = step;
                stepsList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = 'No specific steps available.';
            stepsList.appendChild(li);
        }

        // Pesticides
        const pesticideTable = document.querySelector('#pesticideTable tbody');
        pesticideTable.innerHTML = '';
        const pesticides = treatment.pesticides || [];
        
        if (pesticides.length > 0) {
            pesticides.forEach(pesticide => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${pesticide.name || 'Unknown'}</td>
                    <td>${pesticide.dosage || '-'}</td>
                    <td>${pesticide.frequency || '-'}</td>
                    <td>${pesticide.notes || '-'}</td>
                `;
                pesticideTable.appendChild(row);
            });
        } else {
            const row = document.createElement('tr');
            row.innerHTML = `<td colspan="4" style="text-align: center; color: #666;">No pesticide recommendations available</td>`;
            pesticideTable.appendChild(row);
        }

        // Translations
        const translations = treatment.translations || {};
        document.getElementById('transTe').textContent = translations.te || '-';
        document.getElementById('transEn').textContent = translations.en || '-';
        document.getElementById('transHi').textContent = translations.hi || '-';

        // Heatmap
        if (data.gradcam_base64) {
            document.getElementById('heatmapImage').src = 
                `data:image/jpeg;base64,${data.gradcam_base64}`;
        }

        // Scroll to results
        document.getElementById('resultSection').scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start' 
        });
    }

    switchLanguage(lang) {
        this.currentLanguage = lang;
        
        // Update active tab
        document.querySelectorAll('.language-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.lang === lang);
        });

        console.log('🌐 Language switched to:', lang);

        // Re-analyze if we have a current file
        if (this.currentFile && this.lastResponse) {
            this.analyzeImage();
        }
    }

    async speakTreatment() {
        if (!this.lastResponse) {
            alert('❌ Please analyze an image first.');
            return;
        }

        const treatment = this.lastResponse.treatment || {};
        const text = treatment.treatment_summary || 'No treatment information available.';

        try {
            const response = await fetch('/tts', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: new URLSearchParams({
                    text: text,
                    lang: this.currentLanguage
                })
            });

            if (response.ok) {
                const audio = new Audio();
                audio.src = URL.createObjectURL(await response.blob());
                audio.play();
            }
        } catch (error) {
            console.error('❌ TTS failed:', error);
            alert('Text-to-speech failed. Please try again.');
        }
    }

    saveToHistory() {
        if (!this.lastResponse || !this.currentFile) {
            alert('❌ Please analyze an image first.');
            return;
        }

        const history = this.loadHistory();
        history.unshift({
            timestamp: new Date().toISOString(),
            fileName: this.currentFile.name,
            disease: this.lastResponse.disease,
            confidence: this.lastResponse.confidence
        });

        // Keep only last 10 items
        if (history.length > 10) {
            history.splice(10);
        }

        localStorage.setItem('plantAnalysisHistory', JSON.stringify(history));
        this.displayHistory();
        
        alert('✅ Analysis saved to history!');
    }

    loadHistory() {
        try {
            return JSON.parse(localStorage.getItem('plantAnalysisHistory') || '[]');
        } catch {
            return [];
        }
    }

    displayHistory() {
        const history = this.loadHistory();
        const historyList = document.getElementById('historyList');

        if (history.length === 0) {
            historyList.innerHTML = `
                <p style="color: #666; text-align: center; padding: 20px;">
                    No analysis history yet. Analyze some images to see them here.
                </p>
            `;
            return;
        }

        historyList.innerHTML = history.map(item => `
            <div class="history-item">
                <div>
                    <strong>${item.fileName}</strong>
                    <div style="color: #666; font-size: 0.9rem;">
                        ${item.disease} • ${(item.confidence * 100).toFixed(1)}% • 
                        ${new Date(item.timestamp).toLocaleDateString()}
                    </div>
                </div>
                <button class="action-btn" onclick="app.viewHistoryItem('${item.timestamp}')" 
                        style="padding: 5px 10px; font-size: 0.8rem;">
                    View
                </button>
            </div>
        `).join('');
    }

    viewHistoryItem(timestamp) {
        const history = this.loadHistory();
        const item = history.find(h => h.timestamp === timestamp);
        
        if (item) {
            // Simulate displaying historical result
            document.getElementById('diseaseTitle').textContent = item.disease;
            document.getElementById('confidenceScore').textContent = (item.confidence * 100).toFixed(1) + '%';
            document.getElementById('resultSection').classList.add('active');
            document.getElementById('resultSection').scrollIntoView({ behavior: 'smooth' });
        }
    }

    downloadHeatmap() {
        if (!this.lastResponse || !this.lastResponse.gradcam_base64) {
            alert('❌ No heatmap available to download.');
            return;
        }

        const link = document.createElement('a');
        link.href = `data:image/jpeg;base64,${this.lastResponse.gradcam_base64}`;
        link.download = `${this.lastResponse.disease || 'plant'}_heatmap.jpg`;
        link.click();
    }
}

// Initialize the application
const app = new PlantDiseaseDetector();

// Make app globally available for history buttons
window.app = app;
