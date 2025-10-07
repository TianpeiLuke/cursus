/**
 * Cursus Config UI - JavaScript Client
 * Universal configuration management interface
 * Enhanced with multi-page wizard interface
 */

class CursusConfigUI {
    constructor() {
        this.apiBase = '/api/config-ui';
        this.currentConfig = null;
        this.availableConfigs = {};
        this.currentFormData = {};
        
        // Enhanced state management
        this.pendingRequests = new Set();
        this.requestCache = new Map();
        this.debounceTimers = new Map();
        this.validationErrors = {};
        this.isDirty = false;
        this.isLoading = false;
        
        // Multi-page wizard state
        this.currentSection = 'dag-input';
        this.workflowSteps = [];
        this.currentWorkflowStep = 0;
        this.workflowData = {};
        this.pipelineDAG = null;
        this.analysisResult = null;
        
        this.init();
    }

    init() {
        this.bindEvents();
        this.initializeMultiPageInterface();
        this.setupBeforeUnloadHandler();
        this.loadDAGCatalog();
        this.showStatus('Welcome to Cursus Config UI - Multi-Page Wizard Interface', 'info');
    }

    // Multi-page interface methods
    initializeMultiPageInterface() {
        // Initialize the multi-page wizard interface
        this.showSection('dag-input');
        this.updateProgressIndicator();
    }
    
    showSection(sectionName) {
        // Hide all sections
        document.querySelectorAll('.config-section').forEach(section => {
            section.style.display = 'none';
            section.classList.remove('active');
        });
        
        // Show the requested section
        const targetSection = document.getElementById(`${sectionName}-section`);
        if (targetSection) {
            targetSection.style.display = 'block';
            targetSection.classList.add('active');
            this.currentSection = sectionName;
        }
    }
    
    updateProgressIndicator() {
        // Update progress dots and text based on current workflow step
        const progressDots = document.getElementById('progress-dots');
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        
        if (progressDots && this.workflowSteps.length > 0) {
            const totalSteps = this.workflowSteps.length;
            const currentStep = this.currentWorkflowStep;
            
            // Update dots
            const dots = progressDots.querySelectorAll('.dot');
            dots.forEach((dot, index) => {
                if (index <= currentStep) {
                    dot.textContent = '●';
                    dot.classList.add('active');
                } else {
                    dot.textContent = '○';
                    dot.classList.remove('active');
                }
            });
            
            // Update progress bar
            if (progressFill) {
                const percentage = ((currentStep + 1) / totalSteps) * 100;
                progressFill.style.width = `${percentage}%`;
            }
            
            // Update progress text
            if (progressText) {
                progressText.textContent = `Step ${currentStep + 1} of ${totalSteps}`;
            }
        }
    }

    bindEvents() {
        // Multi-page wizard events
        this.bindMultiPageEvents();
        
        // Legacy single-page events (for backward compatibility)
        this.bindLegacyEvents();
    }
    
    bindMultiPageEvents() {
        // DAG Input Section
        const analyzeBtn = document.getElementById('analyze-dag-btn');
        if (analyzeBtn) {
            analyzeBtn.addEventListener('click', () => this.analyzePipelineDAGFromInput());
        }
        
        const dagFileInput = document.getElementById('dag-file-input');
        if (dagFileInput) {
            dagFileInput.addEventListener('change', (e) => this.handleDAGFileUpload(e));
        }
        
        const dagCatalogSelect = document.getElementById('dag-catalog-select');
        if (dagCatalogSelect) {
            dagCatalogSelect.addEventListener('change', (e) => this.handleDAGCatalogSelection(e));
        }
        
        const dagTextarea = document.getElementById('dag-definition');
        if (dagTextarea) {
            dagTextarea.addEventListener('input', (e) => this.handleDAGTextInput(e));
        }
        
        // Upload zone drag and drop
        const uploadZone = document.getElementById('upload-zone');
        if (uploadZone) {
            this.setupDragAndDrop(uploadZone);
        }
        
        // DAG Analysis Section
        const startWorkflowBtn = document.getElementById('start-workflow-btn');
        if (startWorkflowBtn) {
            startWorkflowBtn.addEventListener('click', () => this.startConfigurationWorkflow());
        }
        
        const backToDAGBtn = document.getElementById('back-to-dag-btn');
        if (backToDAGBtn) {
            backToDAGBtn.addEventListener('click', () => this.showSection('dag-input'));
        }
        
        // Workflow Navigation
        const prevStepBtn = document.getElementById('prev-step-btn');
        if (prevStepBtn) {
            prevStepBtn.addEventListener('click', () => this.previousWorkflowStep());
        }
        
        const nextStepBtn = document.getElementById('next-step-btn');
        if (nextStepBtn) {
            nextStepBtn.addEventListener('click', () => this.nextWorkflowStep());
        }
    }
    
    bindLegacyEvents() {
        // Discovery
        const discoverBtn = document.getElementById('discover-btn');
        if (discoverBtn) {
            discoverBtn.addEventListener('click', () => this.discoverConfigs());
        }
        
        // Configuration creation
        const createWidgetBtn = document.getElementById('create-widget-btn');
        if (createWidgetBtn) {
            createWidgetBtn.addEventListener('click', () => this.createConfigWidget());
        }
        
        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });
    }

    // DAG Input handlers
    async analyzePipelineDAGFromInput() {
        const dagText = document.getElementById('dag-definition').value.trim();
        const dagFile = document.getElementById('dag-file-input').files[0];
        const dagCatalog = document.getElementById('dag-catalog-select').value;
        
        let dagData = null;
        
        try {
            if (dagText) {
                // Parse DAG from text input
                dagData = JSON.parse(dagText);
            } else if (dagFile) {
                // Read DAG from uploaded file
                const fileContent = await this.readFileAsText(dagFile);
                if (dagFile.name.endsWith('.json')) {
                    dagData = JSON.parse(fileContent);
                } else {
                    // For Python files, we'd need backend processing
                    this.showStatus('Python DAG files require backend processing', 'info');
                    return;
                }
            } else if (dagCatalog) {
                // Load DAG from catalog
                dagData = await this.loadDAGFromCatalog(dagCatalog);
            } else {
                this.showStatus('Please provide a DAG definition, upload a file, or select from catalog', 'warning');
                return;
            }
            
            this.showLoading(true);
            
            // Analyze the DAG
            this.analysisResult = await this.analyzePipelineDAG(dagData);
            this.pipelineDAG = dagData;
            
            // Display analysis results
            this.displayDAGAnalysisInSection(this.analysisResult);
            this.showSection('dag-analysis');
            
            this.showStatus('DAG analysis completed successfully', 'success');
            
        } catch (error) {
            console.error('DAG analysis error:', error);
            this.showStatus(`DAG analysis failed: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }
    
    previewDAGStructure() {
        const dagText = document.getElementById('dag-definition').value.trim();
        
        if (!dagText) {
            this.showStatus('Please provide a DAG definition to preview', 'warning');
            return;
        }
        
        try {
            const dagData = JSON.parse(dagText);
            this.showDAGPreviewModal(dagData);
        } catch (error) {
            this.showStatus(`Invalid DAG JSON: ${error.message}`, 'error');
        }
    }
    
    handleDAGFileUpload(event) {
        const file = event.target.files[0];
        const fileNameSpan = document.getElementById('dag-file-name');
        
        if (file) {
            fileNameSpan.textContent = file.name;
            
            // Clear other inputs when file is selected
            document.getElementById('dag-definition').value = '';
            document.getElementById('dag-catalog-select').value = '';
        } else {
            fileNameSpan.textContent = 'No file selected';
        }
    }
    
    handleDAGCatalogSelection(event) {
        const selectedValue = event.target.value;
        
        if (selectedValue) {
            // Clear other inputs when catalog is selected
            document.getElementById('dag-definition').value = '';
            document.getElementById('dag-file-input').value = '';
            const fileNameSpan = document.getElementById('dag-file-name');
            if (fileNameSpan) fileNameSpan.textContent = 'No file selected';
        }
        
        // Enable/disable analyze button
        this.updateAnalyzeButtonState();
    }
    
    handleDAGTextInput(event) {
        const value = event.target.value.trim();
        
        if (value) {
            // Clear other inputs when text is entered
            document.getElementById('dag-file-input').value = '';
            document.getElementById('dag-catalog-select').value = '';
            const fileNameSpan = document.getElementById('dag-file-name');
            if (fileNameSpan) fileNameSpan.textContent = 'No file selected';
        }
        
        // Enable/disable analyze button
        this.updateAnalyzeButtonState();
    }
    
    setupDragAndDrop(uploadZone) {
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadZone.addEventListener(eventName, this.preventDefaults, false);
            document.body.addEventListener(eventName, this.preventDefaults, false);
        });
        
        // Highlight drop area when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadZone.addEventListener(eventName, () => uploadZone.classList.add('dragover'), false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadZone.addEventListener(eventName, () => uploadZone.classList.remove('dragover'), false);
        });
        
        // Handle dropped files
        uploadZone.addEventListener('drop', (e) => this.handleDrop(e), false);
        
        // Handle click to open file dialog
        uploadZone.addEventListener('click', () => {
            document.getElementById('dag-file-input').click();
        });
    }
    
    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            const fileInput = document.getElementById('dag-file-input');
            fileInput.files = files;
            
            // Trigger change event
            const event = new Event('change', { bubbles: true });
            fileInput.dispatchEvent(event);
        }
    }
    
    updateAnalyzeButtonState() {
        const analyzeBtn = document.getElementById('analyze-dag-btn');
        if (!analyzeBtn) return;
        
        const dagText = document.getElementById('dag-definition').value.trim();
        const dagFile = document.getElementById('dag-file-input').files[0];
        const dagCatalog = document.getElementById('dag-catalog-select').value;
        
        const hasInput = dagText || dagFile || dagCatalog;
        
        analyzeBtn.disabled = !hasInput;
        
        if (hasInput) {
            analyzeBtn.classList.remove('disabled');
        } else {
            analyzeBtn.classList.add('disabled');
        }
    }
    
    async readFileAsText(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = (e) => reject(e);
            reader.readAsText(file);
        });
    }
    
    async loadDAGFromCatalog(catalogName) {
        // Mock DAG data for catalog selection
        const catalogDAGs = {
            'xgboost_complete_e2e': {
                nodes: [
                    { name: 'cradle_data_loading', type: 'processing' },
                    { name: 'tabular_preprocessing_training', type: 'processing' },
                    { name: 'xgboost_training', type: 'training' },
                    { name: 'xgboost_model_creation', type: 'model' },
                    { name: 'model_registration', type: 'registration' }
                ],
                edges: []
            },
            'tabular_preprocessing': {
                nodes: [
                    { name: 'tabular_preprocessing_training', type: 'processing' }
                ],
                edges: []
            },
            'model_training': {
                nodes: [
                    { name: 'xgboost_training', type: 'training' },
                    { name: 'xgboost_model_creation', type: 'model' }
                ],
                edges: []
            }
        };
        
        return catalogDAGs[catalogName] || null;
    }
    
    displayDAGAnalysisInSection(analysisResult) {
        const container = document.getElementById('dag-analysis-content');
        
        container.innerHTML = `
            <div class="dag-analysis-results">
                <div class="discovered-steps">
                    <h3>🔍 Discovered Pipeline Steps:</h3>
                    <div class="step-cards">
                        ${(analysisResult.discovered_steps || []).map(step => `
                            <div class="step-card">
                                <div class="step-name">${step.step_name}</div>
                                <div class="step-type">${step.step_type}</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
                
                <div class="required-configs">
                    <h3>⚙️ Required Configurations (Only These Will Be Shown):</h3>
                    <div class="config-cards">
                        ${(analysisResult.required_configs || []).map(config => `
                            <div class="config-card">
                                <div class="config-icon">✅</div>
                                <div class="config-name">${config.config_class_name}</div>
                            </div>
                        `).join('')}
                    </div>
                    <p class="hidden-configs">❌ Hidden: ${analysisResult.hidden_configs_count || 0} other config types not needed</p>
                </div>
                
                <div class="workflow-summary">
                    <h3>📋 Configuration Workflow:</h3>
                    <p>Base Config → Processing Config → ${(analysisResult.required_configs || []).length} Specific Configs</p>
                    <div class="workflow-progress">
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: 0%"></div>
                        </div>
                        <div class="progress-text">Ready to start (0/${analysisResult.total_steps || 0} steps)</div>
                    </div>
                </div>
            </div>
        `;
    }

    async analyzePipelineDAG(pipelineDAG) {
        // Mock analysis for now - in real implementation this would call the backend
        return {
            discovered_steps: pipelineDAG.nodes || [],
            required_configs: (pipelineDAG.nodes || []).map(node => ({
                config_class_name: `${node.name}Config`,
                step_name: node.name,
                is_specialized: node.name === 'cradle_data_loading'
            })),
            workflow_steps: [
                { step_number: 1, title: 'Base Configuration', type: 'base', config_class_name: 'BasePipelineConfig' },
                { step_number: 2, title: 'Processing Configuration', type: 'processing', config_class_name: 'ProcessingStepConfigBase' },
                ...(pipelineDAG.nodes || []).map((node, index) => ({
                    step_number: index + 3,
                    title: `${node.name}Config`,
                    type: 'specific',
                    config_class_name: `${node.name}Config`,
                    is_specialized: node.name === 'cradle_data_loading'
                }))
            ],
            total_steps: 2 + (pipelineDAG.nodes || []).length,
            hidden_configs_count: 47
        };
    }
    
    startConfigurationWorkflow() {
        // Start the step-by-step configuration workflow
        if (!this.analysisResult || !this.analysisResult.workflow_steps) {
            this.showStatus('No workflow steps available', 'error');
            return;
        }
        
        this.workflowSteps = this.analysisResult.workflow_steps;
        this.currentWorkflowStep = 0;
        this.workflowData = {};
        
        // Switch to workflow section and render first step
        this.showSection('workflow');
        this.renderCurrentWorkflowStep();
    }
    
    async renderCurrentWorkflowStep() {
        // Render the current configuration step
        if (this.currentWorkflowStep >= this.workflowSteps.length) {
            this.showSection('completion');
            this.renderWorkflowCompletion();
            return;
        }
        
        const step = this.workflowSteps[this.currentWorkflowStep];
        
        // Update workflow header
        document.getElementById('workflow-step-title').textContent = 
            `🏗️ Configuration Workflow - Step ${step.step_number} of ${this.workflowSteps.length}`;
        document.getElementById('workflow-step-subtitle').textContent = 
            `📋 ${step.title}`;
        
        // Update progress
        this.updateWorkflowProgress();
        
        // Render step content
        const stepContent = document.getElementById('workflow-step-content');
        await this.renderWorkflowStepContent(stepContent, step);
        
        // Update navigation buttons
        const prevBtn = document.getElementById('prev-step-btn');
        const nextBtn = document.getElementById('next-step-btn');
        
        if (prevBtn) {
            prevBtn.disabled = this.currentWorkflowStep === 0;
        }
        
        if (nextBtn) {
            nextBtn.textContent = this.currentWorkflowStep === this.workflowSteps.length - 1 
                ? 'Complete Configuration →' 
                : 'Continue to Next Step →';
        }
    }
    
    updateWorkflowProgress() {
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        const progressDots = document.getElementById('progress-dots');
        
        if (progressFill) {
            const percentage = ((this.currentWorkflowStep + 1) / this.workflowSteps.length) * 100;
            progressFill.style.width = `${percentage}%`;
        }
        
        if (progressText) {
            progressText.textContent = `Step ${this.currentWorkflowStep + 1} of ${this.workflowSteps.length}`;
        }
        
        if (progressDots) {
            const dots = progressDots.querySelectorAll('.dot');
            dots.forEach((dot, index) => {
                if (index <= this.currentWorkflowStep) {
                    dot.textContent = '●';
                    dot.classList.add('active');
                } else {
                    dot.textContent = '○';
                    dot.classList.remove('active');
                }
            });
        }
    }
    
    async renderWorkflowStepContent(container, step) {
        // Render content for a specific workflow step
        if (step.type === 'base') {
            await this.renderBaseConfigStep(container);
        } else if (step.type === 'processing') {
            await this.renderProcessingConfigStep(container);
        } else if (step.type === 'specific') {
            await this.renderSpecificConfigStep(container, step);
        }
    }
    
    async renderBaseConfigStep(container) {
        container.innerHTML = `
            <div class="config-step base-config-step">
                <div class="step-description">
                    <h4>📋 Base Pipeline Configuration (Required for All Steps)</h4>
                    <p>Common configuration fields that will be inherited by all pipeline steps</p>
                </div>
                
                <div class="field-tier essential-tier">
                    <h4>🔥 Essential User Inputs (Tier 1)</h4>
                    <div class="form-row">
                        <div class="field-group required">
                            <label class="form-label">👤 author *</label>
                            <input type="text" class="form-control" id="field-author" placeholder="your-username">
                            <div class="field-description">Pipeline author or owner</div>
                            <div class="field-error" id="error-author"></div>
                        </div>
                        <div class="field-group required">
                            <label class="form-label">🪣 bucket *</label>
                            <input type="text" class="form-control" id="field-bucket" placeholder="your-s3-bucket">
                            <div class="field-description">S3 bucket for pipeline assets</div>
                            <div class="field-error" id="error-bucket"></div>
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="field-group required">
                            <label class="form-label">🔐 role *</label>
                            <input type="text" class="form-control" id="field-role" placeholder="arn:aws:iam::123456789012:role/SageMakerRole">
                            <div class="field-description">AWS IAM role for SageMaker execution</div>
                            <div class="field-error" id="error-role"></div>
                        </div>
                        <div class="field-group required">
                            <label class="form-label">🌍 region *</label>
                            <select class="form-control" id="field-region">
                                <option value="">Select region...</option>
                                <option value="NA">NA</option>
                                <option value="EU">EU</option>
                                <option value="APAC">APAC</option>
                            </select>
                            <div class="field-description">Deployment region</div>
                            <div class="field-error" id="error-region"></div>
                        </div>
                    </div>
                </div>
                
                <div class="field-tier system-tier">
                    <h4>⚙️ System Inputs (Tier 2)</h4>
                    <div class="form-row">
                        <div class="field-group">
                            <label class="form-label">🎯 service_name</label>
                            <input type="text" class="form-control" id="field-service_name" value="AtoZ">
                            <div class="field-description">Service name for pipeline identification</div>
                            <div class="field-error" id="error-service_name"></div>
                        </div>
                        <div class="field-group">
                            <label class="form-label">📅 pipeline_version</label>
                            <input type="text" class="form-control" id="field-pipeline_version" value="1.0.0">
                            <div class="field-description">Pipeline version for tracking</div>
                            <div class="field-error" id="error-pipeline_version"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        this.bindStepFormEvents('BasePipelineConfig');
    }
    
    async renderProcessingConfigStep(container) {
        container.innerHTML = `
            <div class="config-step processing-config-step">
                <div class="step-description">
                    <h4>📋 Processing Configuration (For Processing-Based Steps)</h4>
                    <p>Configuration for SageMaker processing jobs</p>
                </div>
                
                <div class="inherited-config">
                    <h4>💾 Inherited from Base Config</h4>
                    <div class="inherited-summary">
                        <span>• 👤 Author: <strong id="inherited-author">-</strong></span>
                        <span>• 🪣 Bucket: <strong id="inherited-bucket">-</strong></span>
                        <span>• 🔐 Role: <strong id="inherited-role">-</strong></span>
                        <span>• 🌍 Region: <strong id="inherited-region">-</strong></span>
                    </div>
                </div>
                
                <div class="field-tier system-tier">
                    <h4>⚙️ Processing-Specific Fields</h4>
                    <div class="form-row">
                        <div class="field-group">
                            <label class="form-label">🖥️ instance_type</label>
                            <select class="form-control" id="field-instance_type">
                                <option value="ml.m5.large">ml.m5.large</option>
                                <option value="ml.m5.xlarge">ml.m5.xlarge</option>
                                <option value="ml.m5.2xlarge" selected>ml.m5.2xlarge</option>
                                <option value="ml.m5.4xlarge">ml.m5.4xlarge</option>
                            </select>
                            <div class="field-description">EC2 instance type for processing</div>
                            <div class="field-error" id="error-instance_type"></div>
                        </div>
                        <div class="field-group">
                            <label class="form-label">📊 volume_size</label>
                            <input type="number" class="form-control" id="field-volume_size" value="500">
                            <div class="field-description">EBS volume size in GB</div>
                            <div class="field-error" id="error-volume_size"></div>
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="field-group">
                            <label class="form-label">📁 processing_source_dir</label>
                            <input type="text" class="form-control" id="field-processing_source_dir" value="src/processing">
                            <div class="field-description">Source directory for processing code</div>
                            <div class="field-error" id="error-processing_source_dir"></div>
                        </div>
                        <div class="field-group">
                            <label class="form-label">🎯 entry_point</label>
                            <input type="text" class="form-control" id="field-entry_point" value="main.py">
                            <div class="field-description">Entry point script for processing</div>
                            <div class="field-error" id="error-entry_point"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Update inherited values display
        this.updateInheritedValuesDisplay();
        this.bindStepFormEvents('ProcessingStepConfigBase');
    }
    
    async renderSpecificConfigStep(container, step) {
        if (step.is_specialized) {
            // Handle specialized configurations (e.g., CradleDataLoadConfig)
            container.innerHTML = `
                <div class="specialized-config-step">
                    <h4>🎛️ Specialized Configuration</h4>
                    <p>This step uses a specialized wizard interface:</p>
                    <div class="specialized-features">
                        <ul>
                            <li>1️⃣ Data Sources Configuration</li>
                            <li>2️⃣ Transform Specification</li>
                            <li>3️⃣ Output Configuration</li>
                            <li>4️⃣ Cradle Job Settings</li>
                        </ul>
                    </div>
                    <button class="btn btn-primary" onclick="window.cursusUI.openSpecializedWizard('${step.config_class_name}')">
                        Open ${step.config_class_name} Wizard
                    </button>
                    <p class="workflow-note"><small>(Base config will be pre-filled automatically)</small></p>
                </div>
            `;
        } else {
            // Handle standard configurations
            container.innerHTML = `
                <div class="config-step specific-config-step">
                    <div class="step-description">
                        <h4>📋 ${step.title} (Step: ${step.step_name || 'unknown'})</h4>
                        <p>Configuration for ${step.config_class_name}</p>
                    </div>
                    
                    <div class="inherited-config">
                        <h4>💾 Inherited Configuration</h4>
                        <div class="inherited-summary">
                            <p>Auto-filled from Base + Processing Config</p>
                        </div>
                    </div>
                    
                    <div class="field-tier essential-tier">
                        <h4>🎯 Step-Specific Fields</h4>
                        <div class="form-row">
                            <div class="field-group">
                                <label class="form-label">🏷️ step_name</label>
                                <input type="text" class="form-control" id="field-step_name" value="${step.step_name || ''}">
                                <div class="field-description">Name for this pipeline step</div>
                                <div class="field-error" id="error-step_name"></div>
                            </div>
                            <div class="field-group">
                                <label class="form-label">🎯 config_type</label>
                                <input type="text" class="form-control" id="field-config_type" value="${step.config_class_name}" readonly>
                                <div class="field-description">Configuration class type</div>
                                <div class="field-error" id="error-config_type"></div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        this.bindStepFormEvents(step.config_class_name);
    }
    
    bindStepFormEvents(configName) {
        // Bind event listeners for step form fields
        const formFields = document.querySelectorAll('#workflow-step-content .form-control');
        formFields.forEach(field => {
            const fieldName = field.id.replace('field-', '');
            
            field.addEventListener('change', () => this.updateWorkflowFormData(configName, fieldName, field));
            field.addEventListener('input', () => this.updateWorkflowFormData(configName, fieldName, field));
        });
    }
    
    updateWorkflowFormData(configName, fieldName, input) {
        if (!this.workflowData[configName]) {
            this.workflowData[configName] = {};
        }
        
        let value = input.value;
        
        try {
            if (input.type === 'checkbox') {
                value = input.checked;
            } else if (input.type === 'number') {
                value = value === '' ? null : parseFloat(value);
            }
            
            this.workflowData[configName][fieldName] = value;
            this.clearWorkflowFieldError(fieldName);
            this.markFormDirty();
            
        } catch (error) {
            this.showWorkflowFieldError(fieldName, `Invalid input: ${error.message}`);
        }
    }
    
    showWorkflowFieldError(fieldName, message) {
        const errorDiv = document.getElementById(`error-${fieldName}`);
        if (errorDiv) {
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
        }
    }
    
    clearWorkflowFieldError(fieldName) {
        const errorDiv = document.getElementById(`error-${fieldName}`);
        if (errorDiv) {
            errorDiv.textContent = '';
            errorDiv.style.display = 'none';
        }
    }
    
    updateInheritedValuesDisplay() {
        // Update inherited values display in processing step
        const baseData = this.workflowData['BasePipelineConfig'] || {};
        
        const authorSpan = document.getElementById('inherited-author');
        const bucketSpan = document.getElementById('inherited-bucket');
        const roleSpan = document.getElementById('inherited-role');
        const regionSpan = document.getElementById('inherited-region');
        
        if (authorSpan) authorSpan.textContent = baseData.author || '-';
        if (bucketSpan) bucketSpan.textContent = baseData.bucket || '-';
        if (roleSpan) roleSpan.textContent = baseData.role || '-';
        if (regionSpan) regionSpan.textContent = baseData.region || '-';
    }
    
    nextWorkflowStep() {
        // Validate current step
        if (!this.validateCurrentWorkflowStep()) {
            this.showStatus('Please fix validation errors before continuing', 'error');
            return;
        }
        
        // Save current step data
        this.saveCurrentWorkflowStep();
        
        // Move to next step
        this.currentWorkflowStep++;
        this.renderCurrentWorkflowStep();
    }
    
    previousWorkflowStep() {
        if (this.currentWorkflowStep > 0) {
            this.currentWorkflowStep--;
            this.renderCurrentWorkflowStep();
        }
    }
    
    validateCurrentWorkflowStep() {
        const requiredFields = document.querySelectorAll('#workflow-step-content .field-group.required .form-control');
        let isValid = true;
        
        requiredFields.forEach(field => {
            const fieldName = field.id.replace('field-', '');
            let value = field.value;
            
            if (field.type === 'checkbox') {
                value = field.checked;
            }
            
            if (!value || (typeof value === 'string' && value.trim() === '')) {
                this.showWorkflowFieldError(fieldName, `${fieldName} is required`);
                field.classList.add('error');
                isValid = false;
            } else {
                this.clearWorkflowFieldError(fieldName);
                field.classList.remove('error');
            }
        });
        
        return isValid;
    }
    
    saveCurrentWorkflowStep() {
        const currentStep = this.workflowSteps[this.currentWorkflowStep];
        if (currentStep) {
            currentStep.completed = true;
            console.log(`Step ${currentStep.title} completed`);
        }
    }
    
    renderWorkflowCompletion() {
        const container = document.getElementById('completion-content');
        
        container.innerHTML = `
            <div class="completion-summary">
                <h3>📋 Configuration Summary:</h3>
                <ul class="completed-configs">
                    ${this.workflowSteps.map(step => `
                        <li class="${step.completed ? 'completed' : 'incomplete'}">
                            ${step.completed ? '✅' : '⏳'} ${step.title}
                        </li>
                    `).join('')}
                </ul>
            </div>
            
            <div class="export-options">
                <h3>💾 Export Options:</h3>
                <div class="export-buttons">
                    <button class="btn btn-success btn-large" onclick="window.cursusUI.saveAllMerged()">
                        💾 Save All Merged
                        <small>Creates unified hierarchical JSON (Recommended)</small>
                    </button>
                    <button class="btn btn-info" onclick="window.cursusUI.exportIndividualConfigs()">
                        📤 Export Individual
                        <small>Individual JSON files for each configuration</small>
                    </button>
                </div>
            </div>
        `;
    }

    // Utility methods
    setupBeforeUnloadHandler() {
        window.addEventListener('beforeunload', (e) => {
            if (this.isDirty) {
                e.preventDefault();
                e.returnValue = 'You have unsaved changes. Are you sure you want to leave?';
                return e.returnValue;
            }
        });
    }

    markFormDirty() {
        this.isDirty = true;
    }

    markFormClean() {
        this.isDirty = false;
    }

    showLoading(show) {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.style.display = show ? 'flex' : 'none';
        }
    }

    showStatus(message, type = 'info') {
        const container = document.getElementById('status-messages');
        if (!container) return;
        
        const statusDiv = document.createElement('div');
        statusDiv.className = `status-message ${type}`;
        statusDiv.textContent = message;
        
        container.appendChild(statusDiv);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (statusDiv.parentNode) {
                statusDiv.parentNode.removeChild(statusDiv);
            }
        }, 5000);
        
        // Remove on click
        statusDiv.addEventListener('click', () => {
            if (statusDiv.parentNode) {
                statusDiv.parentNode.removeChild(statusDiv);
            }
        });
    }

    // DAG Catalog loading
    async loadDAGCatalog() {
        try {
            console.log('Loading DAG catalog from backend...');
            
            const response = await fetch(`${this.apiBase}/catalog/dags`);
            if (!response.ok) {
                throw new Error(`Failed to load DAG catalog: ${response.status}`);
            }
            
            const catalogData = await response.json();
            console.log(`Loaded ${catalogData.count} DAGs from catalog:`, catalogData.dags);
            
            // Populate the dropdown
            this.populateDAGCatalogDropdown(catalogData.dags);
            
        } catch (error) {
            console.error('Failed to load DAG catalog:', error);
            this.showStatus('Failed to load DAG catalog. Using fallback options.', 'warning');
            
            // Fallback to hardcoded options
            this.populateDAGCatalogDropdown([
                {
                    id: 'xgboost_complete_e2e',
                    display_name: 'XGBoost - Complete End-to-End Pipeline',
                    description: 'Complete XGBoost workflow with training, calibration, and registration',
                    complexity: 'complex',
                    framework: 'xgboost'
                },
                {
                    id: 'pytorch_standard_e2e',
                    display_name: 'PyTorch - Standard End-to-End Pipeline',
                    description: 'Standard PyTorch training and evaluation workflow',
                    complexity: 'medium',
                    framework: 'pytorch'
                },
                {
                    id: 'dummy_e2e_basic',
                    display_name: 'Dummy - Basic End-to-End Pipeline',
                    description: 'Simple dummy pipeline for testing',
                    complexity: 'simple',
                    framework: 'dummy'
                }
            ]);
        }
    }
    
    populateDAGCatalogDropdown(dags) {
        const select = document.getElementById('dag-catalog-select');
        if (!select) {
            console.warn('DAG catalog select element not found');
            return;
        }
        
        // Clear existing options except the first one
        while (select.children.length > 1) {
            select.removeChild(select.lastChild);
        }
        
        // Group DAGs by framework for better organization
        const dagsByFramework = {};
        dags.forEach(dag => {
            const framework = dag.framework || 'other';
            if (!dagsByFramework[framework]) {
                dagsByFramework[framework] = [];
            }
            dagsByFramework[framework].push(dag);
        });
        
        // Add options grouped by framework
        Object.keys(dagsByFramework).sort().forEach(framework => {
            // Add framework group header
            const optgroup = document.createElement('optgroup');
            optgroup.label = `${framework.charAt(0).toUpperCase() + framework.slice(1)} Pipelines`;
            
            dagsByFramework[framework].forEach(dag => {
                const option = document.createElement('option');
                option.value = dag.id;
                option.textContent = `${dag.display_name} (${dag.complexity})`;
                option.title = dag.description;
                
                // Store DAG data for later use
                option.dataset.dagData = JSON.stringify(dag);
                
                optgroup.appendChild(option);
            });
            
            select.appendChild(optgroup);
        });
        
        console.log(`Populated dropdown with ${dags.length} DAG options`);
    }
    
    async loadDAGFromCatalog(catalogId) {
        try {
            // Get the selected option to retrieve stored DAG data
            const select = document.getElementById('dag-catalog-select');
            const selectedOption = select.querySelector(`option[value="${catalogId}"]`);
            
            if (selectedOption && selectedOption.dataset.dagData) {
                const dagInfo = JSON.parse(selectedOption.dataset.dagData);
                console.log('Loading DAG from catalog:', dagInfo);
                
                // Return the DAG structure
                return dagInfo.dag_structure || {
                    nodes: dagInfo.dag_structure?.nodes || [],
                    edges: dagInfo.dag_structure?.edges || []
                };
            } else {
                // Fallback to hardcoded data if not found
                console.warn(`DAG ${catalogId} not found in catalog, using fallback`);
                return this.getFallbackDAGData(catalogId);
            }
            
        } catch (error) {
            console.error('Error loading DAG from catalog:', error);
            return this.getFallbackDAGData(catalogId);
        }
    }
    
    getFallbackDAGData(catalogId) {
        // Fallback DAG data for testing
        const fallbackDAGs = {
            'xgboost_complete_e2e': {
                nodes: [
                    { name: 'CradleDataLoading_training', type: 'processing' },
                    { name: 'TabularPreprocessing_training', type: 'processing' },
                    { name: 'XGBoostTraining', type: 'training' },
                    { name: 'ModelCalibration_calibration', type: 'calibration' },
                    { name: 'Package', type: 'packaging' },
                    { name: 'Registration', type: 'registration' },
                    { name: 'Payload', type: 'payload' },
                    { name: 'CradleDataLoading_calibration', type: 'processing' },
                    { name: 'TabularPreprocessing_calibration', type: 'processing' },
                    { name: 'XGBoostModelEval_calibration', type: 'evaluation' }
                ],
                edges: [
                    { from: 'CradleDataLoading_training', to: 'TabularPreprocessing_training' },
                    { from: 'TabularPreprocessing_training', to: 'XGBoostTraining' },
                    { from: 'CradleDataLoading_calibration', to: 'TabularPreprocessing_calibration' },
                    { from: 'XGBoostTraining', to: 'XGBoostModelEval_calibration' },
                    { from: 'TabularPreprocessing_calibration', to: 'XGBoostModelEval_calibration' },
                    { from: 'XGBoostModelEval_calibration', to: 'ModelCalibration_calibration' },
                    { from: 'ModelCalibration_calibration', to: 'Package' },
                    { from: 'XGBoostTraining', to: 'Package' },
                    { from: 'XGBoostTraining', to: 'Payload' },
                    { from: 'Package', to: 'Registration' },
                    { from: 'Payload', to: 'Registration' }
                ]
            },
            'pytorch_standard_e2e': {
                nodes: [
                    { name: 'CradleDataLoading_training', type: 'processing' },
                    { name: 'TabularPreprocessing_training', type: 'processing' },
                    { name: 'PyTorchTraining', type: 'training' },
                    { name: 'PyTorchModelEval', type: 'evaluation' },
                    { name: 'Package', type: 'packaging' },
                    { name: 'Registration', type: 'registration' }
                ],
                edges: [
                    { from: 'CradleDataLoading_training', to: 'TabularPreprocessing_training' },
                    { from: 'TabularPreprocessing_training', to: 'PyTorchTraining' },
                    { from: 'PyTorchTraining', to: 'PyTorchModelEval' },
                    { from: 'PyTorchModelEval', to: 'Package' },
                    { from: 'Package', to: 'Registration' }
                ]
            },
            'dummy_e2e_basic': {
                nodes: [
                    { name: 'DummyDataLoading', type: 'processing' },
                    { name: 'DummyProcessing', type: 'processing' },
                    { name: 'DummyTraining', type: 'training' }
                ],
                edges: [
                    { from: 'DummyDataLoading', to: 'DummyProcessing' },
                    { from: 'DummyProcessing', to: 'DummyTraining' }
                ]
            }
        };
        
        return fallbackDAGs[catalogId] || fallbackDAGs['dummy_e2e_basic'];
    }

    // Placeholder methods for missing functionality
    async discoverConfigs() {
        this.showStatus('Discovery functionality available in legacy mode', 'info');
    }

    async createConfigWidget() {
        this.showStatus('Widget creation available in legacy mode', 'info');
    }

    switchTab(tabName) {
        // Tab switching for legacy mode
        console.log(`Switching to tab: ${tabName}`);
    }

    async saveAllMerged() {
        const sessionConfigs = this.workflowData;
        
        if (Object.keys(sessionConfigs).length === 0) {
            this.showStatus('No configurations to merge', 'warning');
            return;
        }
        
        // Mock merge functionality
        const mergedConfig = {
            shared: sessionConfigs['BasePipelineConfig'] || {},
            processing_shared: sessionConfigs['ProcessingStepConfigBase'] || {},
            specific: {}
        };
        
        // Add specific configs
        Object.entries(sessionConfigs).forEach(([configName, configData]) => {
            if (configName !== 'BasePipelineConfig' && configName !== 'ProcessingStepConfigBase') {
                mergedConfig.specific[configName] = configData;
            }
        });
        
        // Create download
        const jsonString = JSON.stringify(mergedConfig, null, 2);
        const blob = new Blob([jsonString], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = 'config_merged.json';
        link.click();
        
        URL.revokeObjectURL(url);
        
        this.showStatus('Configuration merged and downloaded', 'success');
    }

    exportIndividualConfigs() {
        const sessionConfigs = this.workflowData;
        
        if (Object.keys(sessionConfigs).length === 0) {
            this.showStatus('No configurations to export', 'warning');
            return;
        }
        
        Object.entries(sessionConfigs).forEach(([configName, configData]) => {
            const jsonString = JSON.stringify(configData, null, 2);
            const blob = new Blob([jsonString], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            
            const link = document.createElement('a');
            link.href = url;
            link.download = `${configName}.json`;
            link.click();
            
            URL.revokeObjectURL(url);
        });
        
        this.showStatus(`Exported ${Object.keys(sessionConfigs).length} individual configuration files`, 'success');
    }

    // Placeholder methods for specialized functionality
    openSpecializedWizard(configClassName) {
        this.showStatus(`Opening specialized wizard for ${configClassName}`, 'info');
        // In real implementation, this would open the specialized interface
    }

    showDAGPreviewModal(dagData) {
        this.showStatus('DAG preview functionality coming soon', 'info');
    }

    executeWorkflowPipeline() {
        this.showStatus('Pipeline execution functionality coming soon', 'info');
    }

    saveWorkflowAsTemplate() {
        this.showStatus('Template saving functionality coming soon', 'info');
    }

    modifyWorkflowConfiguration() {
        // Go back to first step to modify
        this.currentWorkflowStep = 0;
        this.renderCurrentWorkflowStep();
        this.showSection('workflow');
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.cursusUI = new CursusConfigUI();
});

// Export for global access
window.CursusConfigUI = CursusConfigUI;
