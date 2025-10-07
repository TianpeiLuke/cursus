/**
 * Cursus Config UI - JavaScript Client
 * Universal configuration management interface
 * Enhanced with robust patterns from Cradle UI
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
        
        this.init();
    }

    init() {
        this.bindEvents();
        this.initializeTabs();
        this.setupBeforeUnloadHandler();
        this.showStatus('Welcome to Cursus Config UI', 'info');
    }

    // Enhanced request management with deduplication and caching
    async makeRequest(url, options = {}, cacheKey = null) {
        // Check cache first
        if (cacheKey && this.requestCache.has(cacheKey)) {
            console.log(`Using cached response for: ${cacheKey}`);
            return this.requestCache.get(cacheKey);
        }
        
        // Prevent duplicate requests
        const requestId = `${options.method || 'GET'}-${url}`;
        if (this.pendingRequests.has(requestId)) {
            console.log(`Request already pending: ${requestId}`);
            return null; // Or return a promise that resolves when the pending request completes
        }
        
        this.pendingRequests.add(requestId);
        this.showLoading(true);
        
        try {
            const response = await fetch(url, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            // Cache successful responses
            if (cacheKey && response.ok) {
                this.requestCache.set(cacheKey, data);
                // Auto-expire cache after 5 minutes
                setTimeout(() => this.requestCache.delete(cacheKey), 5 * 60 * 1000);
            }
            
            return data;
            
        } catch (error) {
            console.error(`Request failed: ${requestId}`, error);
            throw error;
        } finally {
            this.pendingRequests.delete(requestId);
            this.showLoading(false);
        }
    }

    // Debounce utility for validation
    debounce(func, wait) {
        return (...args) => {
            const key = args[0]; // Use first argument as key
            clearTimeout(this.debounceTimers.get(key));
            this.debounceTimers.set(key, setTimeout(() => func.apply(this, args), wait));
        };
    }

    // Setup unsaved changes warning
    setupBeforeUnloadHandler() {
        window.addEventListener('beforeunload', (e) => {
            if (this.isDirty) {
                e.preventDefault();
                e.returnValue = 'You have unsaved changes. Are you sure you want to leave?';
                return e.returnValue;
            }
        });
    }

    // Clear all state when navigating away from forms
    async clearServerConfiguration() {
        try {
            await this.makeRequest(`${this.apiBase}/clear-config`, {
                method: 'POST'
            });
            console.log('Server configuration cleared');
        } catch (error) {
            console.error('Error clearing server configuration:', error);
        }
    }

    bindEvents() {
        // Discovery
        document.getElementById('discover-btn').addEventListener('click', () => this.discoverConfigs());
        
        // Configuration creation
        document.getElementById('create-widget-btn').addEventListener('click', () => this.createConfigWidget());
        
        // Form actions
        document.getElementById('save-config-btn').addEventListener('click', () => this.saveConfiguration());
        document.getElementById('cancel-config-btn').addEventListener('click', () => this.cancelConfiguration());
        document.getElementById('export-config-btn').addEventListener('click', () => this.exportConfiguration());
        
        // Pipeline wizard
        document.getElementById('create-pipeline-btn').addEventListener('click', () => this.createPipelineWizard());
        
        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });
    }

    initializeTabs() {
        // Initialize with JSON tab active
        this.switchTab('json');
    }

    async discoverConfigs() {
        try {
            const workspaceDirs = document.getElementById('workspace-dirs').value
                .split(',')
                .map(dir => dir.trim())
                .filter(dir => dir.length > 0);
            
            const cacheKey = `discover-${JSON.stringify(workspaceDirs)}`;
            const data = await this.makeRequest(`${this.apiBase}/discover`, {
                method: 'POST',
                body: JSON.stringify({
                    workspace_dirs: workspaceDirs.length > 0 ? workspaceDirs : null
                })
            }, cacheKey);
            
            if (data) {
                this.availableConfigs = data.configs;
                this.renderConfigList();
                this.populateConfigTypeSelect();
                this.showStatus(`Discovered ${Object.keys(this.availableConfigs).length} configuration types`, 'success');
            }
            
        } catch (error) {
            this.handleApiError(error, 'Configuration discovery');
        }
    }

    async renderConfigList() {
        const container = document.getElementById('config-list');
        container.innerHTML = '';
        
        if (Object.keys(this.availableConfigs).length === 0) {
            container.innerHTML = '<p class="text-center">No configurations discovered. Click "Discover Configurations" to scan for available types.</p>';
            return;
        }
        
        // Show loading message at the bottom while fetching field data
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'text-center loading-message';
        loadingDiv.style.cssText = 'position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); background: white; padding: 10px 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); z-index: 1000;';
        loadingDiv.innerHTML = '<p>Loading configuration forms...</p>';
        document.body.appendChild(loadingDiv);
        
        // Create configuration sections with form fields
        for (const [name, info] of Object.entries(this.availableConfigs)) {
            await this.renderConfigWithFields(container, name, info);
        }
        
        // Remove loading message when done
        if (loadingDiv.parentNode) {
            loadingDiv.parentNode.removeChild(loadingDiv);
        }
    }

    async renderConfigWithFields(container, configName, configInfo) {
        try {
            // Fetch field data for this configuration
            const fieldData = await this.makeRequest(`${this.apiBase}/create-widget`, {
                method: 'POST',
                body: JSON.stringify({
                    config_class_name: configName,
                    base_config: null
                })
            }, `fields-${configName}`);
            
            // Create configuration section
            const configSection = document.createElement('div');
            configSection.className = 'config-section';
            configSection.innerHTML = `
                <div class="config-header">
                    <h3>${configName}</h3>
                    <p>${configInfo.description || 'Configuration class for pipeline components'}</p>
                    <div class="config-meta">
                        <span>Module: ${configInfo.module || 'Unknown'}</span>
                        <span>Fields: ${fieldData.fields?.length || 0}</span>
                    </div>
                </div>
                <div class="config-form-container" id="form-${configName}">
                    <!-- Form fields will be inserted here -->
                </div>
                <div class="config-actions">
                    <button class="btn btn-success" onclick="window.cursusUI.saveConfigurationByName('${configName}')">
                        Save ${configName}
                    </button>
                    <button class="btn btn-info" onclick="window.cursusUI.exportConfigurationByName('${configName}')">
                        Export JSON
                    </button>
                </div>
            `;
            
            container.appendChild(configSection);
            
            // Render form fields if available
            if (fieldData.fields && fieldData.fields.length > 0) {
                this.renderFormFields(configName, fieldData);
            } else if (fieldData.specialized_component) {
                this.renderSpecializedComponentInline(configName, fieldData);
            }
            
        } catch (error) {
            console.error(`Error loading fields for ${configName}:`, error);
            
            // Create fallback section
            const configSection = document.createElement('div');
            configSection.className = 'config-section error';
            configSection.innerHTML = `
                <div class="config-header">
                    <h3>${configName}</h3>
                    <p class="error-text">Error loading configuration fields: ${error.message}</p>
                    <div class="config-meta">
                        <span>Module: ${configInfo.module || 'Unknown'}</span>
                    </div>
                </div>
            `;
            container.appendChild(configSection);
        }
    }

    renderFormFields(configName, fieldData) {
        const formContainer = document.getElementById(`form-${configName}`);
        if (!formContainer) return;
        
        // Initialize form data for this config
        if (!this.currentFormData[configName]) {
            this.currentFormData[configName] = { ...fieldData.values };
        }
        
        // Create form with better organization
        const form = document.createElement('div');
        form.className = 'dynamic-form';
        
        // Group fields logically based on field names and types
        const fieldGroups = this.organizeFieldsIntoGroups(fieldData.fields);
        
        fieldGroups.forEach(group => {
            if (group.title) {
                const sectionDiv = document.createElement('div');
                sectionDiv.className = 'field-group-section';
                
                const sectionTitle = document.createElement('h4');
                sectionTitle.textContent = group.title;
                sectionDiv.appendChild(sectionTitle);
                
                // Create form rows for this section
                this.createFormRowsForFields(sectionDiv, configName, group.fields, fieldData.values);
                
                form.appendChild(sectionDiv);
            } else {
                // Create form rows for ungrouped fields
                this.createFormRowsForFields(form, configName, group.fields, fieldData.values);
            }
        });
        
        formContainer.appendChild(form);
    }

    organizeFieldsIntoGroups(fields) {
        const groups = [];
        const requiredFields = [];
        const optionalFields = [];
        const processingFields = [];
        const modelFields = [];
        const otherFields = [];
        
        fields.forEach(field => {
            if (field.name.includes('processing_')) {
                processingFields.push(field);
            } else if (field.name.includes('model_')) {
                modelFields.push(field);
            } else if (field.required) {
                requiredFields.push(field);
            } else {
                optionalFields.push(field);
            }
        });
        
        // Add required fields first
        if (requiredFields.length > 0) {
            groups.push({
                title: 'Required Configuration',
                fields: requiredFields
            });
        }
        
        // Add processing fields
        if (processingFields.length > 0) {
            groups.push({
                title: 'Processing Configuration',
                fields: processingFields
            });
        }
        
        // Add model fields
        if (modelFields.length > 0) {
            groups.push({
                title: 'Model Configuration',
                fields: modelFields
            });
        }
        
        // Add optional fields
        if (optionalFields.length > 0) {
            groups.push({
                title: 'Optional Configuration',
                fields: optionalFields
            });
        }
        
        // If no logical grouping, return all fields
        if (groups.length === 0) {
            groups.push({
                title: null,
                fields: fields
            });
        }
        
        return groups;
    }

    createFormRowsForFields(container, configName, fields, values) {
        // Create form rows (2 fields per row)
        for (let i = 0; i < fields.length; i += 2) {
            const row = document.createElement('div');
            row.className = 'form-row';
            
            // Add first field
            const field1 = fields[i];
            const fieldGroup1 = this.createFormFieldForConfig(configName, field1, values[field1.name]);
            row.appendChild(fieldGroup1);
            
            // Add second field if it exists
            if (i + 1 < fields.length) {
                const field2 = fields[i + 1];
                const fieldGroup2 = this.createFormFieldForConfig(configName, field2, values[field2.name]);
                row.appendChild(fieldGroup2);
            } else {
                // Add empty div to maintain grid layout
                const emptyDiv = document.createElement('div');
                row.appendChild(emptyDiv);
            }
            
            container.appendChild(row);
        }
    }

    renderSpecializedComponentInline(configName, fieldData) {
        const formContainer = document.getElementById(`form-${configName}`);
        if (!formContainer) return;
        
        formContainer.innerHTML = `
            <div class="specialized-widget-inline">
                <p><strong>🎛️ Specialized Interface Required</strong></p>
                <p>This configuration type uses a specialized interface.</p>
                <button class="btn btn-primary" onclick="window.open('/cradle-ui', '_blank')">
                    Open Specialized Interface
                </button>
            </div>
        `;
    }

    createFormFieldForConfig(configName, field, currentValue) {
        const fieldGroup = document.createElement('div');
        fieldGroup.className = `field-group ${field.required ? 'required' : ''}`;
        
        // Label
        const label = document.createElement('label');
        label.textContent = `${field.name}${field.required ? ' *' : ''}:`;
        label.className = 'form-label';
        fieldGroup.appendChild(label);
        
        // Input element
        let input;
        const value = currentValue !== undefined ? currentValue : '';
        
        switch (field.type) {
            case 'checkbox':
                input = document.createElement('input');
                input.type = 'checkbox';
                input.checked = Boolean(value);
                input.className = 'form-check-input';
                break;
                
            case 'number':
                input = document.createElement('input');
                input.type = 'number';
                input.step = 'any';
                input.value = value;
                input.className = 'form-control';
                break;
                
            case 'list':
                input = document.createElement('textarea');
                input.value = Array.isArray(value) ? JSON.stringify(value, null, 2) : value;
                input.placeholder = 'Enter JSON array, e.g., ["item1", "item2"]';
                input.className = 'form-control';
                input.rows = 3;
                break;
                
            case 'keyvalue':
                input = document.createElement('textarea');
                input.value = typeof value === 'object' ? JSON.stringify(value, null, 2) : value;
                input.placeholder = 'Enter JSON object, e.g., {"key": "value"}';
                input.className = 'form-control';
                input.rows = 4;
                break;
                
            default:
                input = document.createElement('input');
                input.type = 'text';
                input.value = value;
                input.className = 'form-control';
        }
        
        input.id = `field-${configName}-${field.name}`;
        input.addEventListener('change', () => this.updateFormDataForConfig(configName, field.name, input, field.type));
        input.addEventListener('input', () => this.updateFormDataForConfig(configName, field.name, input, field.type));
        
        fieldGroup.appendChild(input);
        
        // Description
        if (field.description) {
            const desc = document.createElement('div');
            desc.className = 'field-description';
            desc.textContent = field.description;
            fieldGroup.appendChild(desc);
        }
        
        // Error container
        const errorDiv = document.createElement('div');
        errorDiv.className = 'field-error';
        errorDiv.id = `error-${configName}-${field.name}`;
        fieldGroup.appendChild(errorDiv);
        
        return fieldGroup;
    }

    updateFormDataForConfig(configName, fieldName, input, fieldType) {
        if (!this.currentFormData[configName]) {
            this.currentFormData[configName] = {};
        }
        
        let value = input.value;
        
        try {
            switch (fieldType) {
                case 'checkbox':
                    value = input.checked;
                    break;
                case 'number':
                    value = value === '' ? null : parseFloat(value);
                    break;
                case 'list':
                    value = value.trim() ? JSON.parse(value) : [];
                    break;
                case 'keyvalue':
                    value = value.trim() ? JSON.parse(value) : {};
                    break;
            }
            
            this.currentFormData[configName][fieldName] = value;
            this.clearFieldErrorForConfig(configName, fieldName);
            this.markFormDirty();
            
        } catch (error) {
            this.showFieldErrorForConfig(configName, fieldName, `Invalid ${fieldType}: ${error.message}`);
        }
    }

    showFieldErrorForConfig(configName, fieldName, message) {
        const errorDiv = document.getElementById(`error-${configName}-${fieldName}`);
        if (errorDiv) {
            errorDiv.textContent = message;
        }
    }

    clearFieldErrorForConfig(configName, fieldName) {
        const errorDiv = document.getElementById(`error-${configName}-${fieldName}`);
        if (errorDiv) {
            errorDiv.textContent = '';
        }
    }

    async saveConfigurationByName(configName) {
        if (!this.currentFormData[configName]) {
            this.showStatus(`No data to save for ${configName}`, 'warning');
            return;
        }
        
        this.showLoading(true);
        
        try {
            const response = await fetch(`${this.apiBase}/save-config`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    config_class_name: configName,
                    form_data: this.currentFormData[configName]
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                
                // Handle Pydantic validation errors specifically
                if (response.status === 422 && errorData.detail?.error_type === 'validation_error') {
                    this.handlePydanticValidationErrorsForConfig(configName, errorData.detail.validation_errors);
                    this.showStatus(`Please fix the validation errors for ${configName}`, 'error');
                    return;
                } else {
                    throw new Error(errorData.detail?.message || `HTTP ${response.status}: ${response.statusText}`);
                }
            }
            
            const result = await response.json();
            
            // Update results display
            this.displayResults(result);
            
            this.showStatus(`${configName} saved successfully!`, 'success');
            
        } catch (error) {
            console.error(`Save error for ${configName}:`, error);
            this.showStatus(`Save failed for ${configName}: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    exportConfigurationByName(configName) {
        if (!this.currentFormData[configName] || Object.keys(this.currentFormData[configName]).length === 0) {
            this.showStatus(`No data to export for ${configName}`, 'warning');
            return;
        }
        
        const dataStr = JSON.stringify(this.currentFormData[configName], null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `${configName}.json`;
        link.click();
        
        this.showStatus(`${configName} configuration exported`, 'success');
    }

    handlePydanticValidationErrorsForConfig(configName, validationErrors) {
        console.log(`Handling Pydantic validation errors for ${configName}:`, validationErrors);
        
        // Display each validation error on the corresponding field
        validationErrors.forEach(error => {
            const fieldName = error.field;
            const message = error.message;
            const errorType = error.type;
            
            // Format user-friendly error message
            let userMessage = message;
            if (errorType === 'missing') {
                userMessage = `${fieldName} is required`;
            } else if (errorType === 'value_error') {
                userMessage = `Invalid value for ${fieldName}: ${message}`;
            } else if (errorType === 'type_error') {
                userMessage = `Wrong type for ${fieldName}: ${message}`;
            }
            
            // Show error on the specific field
            this.showFieldErrorForConfig(configName, fieldName, userMessage);
            
            // Highlight the field with error
            const fieldInput = document.getElementById(`field-${configName}-${fieldName}`);
            if (fieldInput) {
                fieldInput.classList.add('error');
                fieldInput.addEventListener('input', () => {
                    fieldInput.classList.remove('error');
                    this.clearFieldErrorForConfig(configName, fieldName);
                }, { once: true });
            }
        });
        
        // Scroll to first error field
        if (validationErrors.length > 0) {
            const firstErrorField = document.getElementById(`field-${configName}-${validationErrors[0].field}`);
            if (firstErrorField) {
                firstErrorField.scrollIntoView({ behavior: 'smooth', block: 'center' });
                firstErrorField.focus();
            }
        }
    }

    populateConfigTypeSelect() {
        const select = document.getElementById('config-type');
        select.innerHTML = '<option value="">Select a configuration type...</option>';
        
        Object.keys(this.availableConfigs).forEach(name => {
            const option = document.createElement('option');
            option.value = name;
            option.textContent = name;
            select.appendChild(option);
        });
    }

    async createConfigWidget() {
        const configType = document.getElementById('config-type').value;
        const baseConfigText = document.getElementById('base-config').value.trim();
        
        if (!configType) {
            this.showStatus('Please select a configuration type', 'warning');
            return;
        }
        
        this.showLoading(true);
        
        try {
            let baseConfig = null;
            if (baseConfigText) {
                try {
                    baseConfig = JSON.parse(baseConfigText);
                } catch (e) {
                    throw new Error('Invalid JSON in base configuration');
                }
            }
            
            const response = await fetch(`${this.apiBase}/create-widget`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    config_class_name: configType,
                    base_config: baseConfig
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            this.currentConfig = data;
            
            this.renderConfigForm(data);
            this.showConfigFormSection(true);
            
            this.showStatus(`Created ${configType} configuration form`, 'success');
            
        } catch (error) {
            console.error('Widget creation error:', error);
            this.showStatus(`Widget creation failed: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    renderConfigForm(configData) {
        const container = document.getElementById('config-form-container');
        container.innerHTML = '';
        
        // Check if this is a specialized component
        if (configData.specialized_component) {
            this.renderSpecializedComponent(container, configData);
            return;
        }
        
        // Create dynamic form
        const form = document.createElement('div');
        form.className = 'dynamic-form';
        
        configData.fields.forEach(field => {
            const fieldGroup = this.createFormField(field, configData.values[field.name]);
            form.appendChild(fieldGroup);
        });
        
        container.appendChild(form);
        this.currentFormData = { ...configData.values };
    }

    renderSpecializedComponent(container, configData) {
        const widget = document.createElement('div');
        widget.className = 'specialized-widget';
        widget.innerHTML = `
            <h3>🎛️ Specialized ${configData.config_class_name} Interface</h3>
            <p>This configuration type uses a specialized interface.</p>
            <p>For ${configData.config_class_name}, please use the dedicated Jupyter widget or specialized UI component.</p>
            <div class="mt-3">
                <button class="btn btn-primary" onclick="window.open('/cradle-ui', '_blank')">
                    Open Specialized Interface
                </button>
            </div>
        `;
        container.appendChild(widget);
    }

    createFormField(field, currentValue) {
        const fieldGroup = document.createElement('div');
        fieldGroup.className = `field-group ${field.required ? 'required' : ''}`;
        
        // Label
        const label = document.createElement('label');
        label.textContent = `${field.name}${field.required ? ' *' : ''}:`;
        label.className = 'form-label';
        fieldGroup.appendChild(label);
        
        // Input element
        let input;
        const value = currentValue !== undefined ? currentValue : '';
        
        switch (field.type) {
            case 'checkbox':
                input = document.createElement('input');
                input.type = 'checkbox';
                input.checked = Boolean(value);
                input.className = 'form-check-input';
                break;
                
            case 'number':
                input = document.createElement('input');
                input.type = 'number';
                input.step = 'any';
                input.value = value;
                input.className = 'form-control';
                break;
                
            case 'list':
                input = document.createElement('textarea');
                input.value = Array.isArray(value) ? JSON.stringify(value, null, 2) : value;
                input.placeholder = 'Enter JSON array, e.g., ["item1", "item2"]';
                input.className = 'form-control';
                input.rows = 3;
                break;
                
            case 'keyvalue':
                input = document.createElement('textarea');
                input.value = typeof value === 'object' ? JSON.stringify(value, null, 2) : value;
                input.placeholder = 'Enter JSON object, e.g., {"key": "value"}';
                input.className = 'form-control';
                input.rows = 4;
                break;
                
            default:
                input = document.createElement('input');
                input.type = 'text';
                input.value = value;
                input.className = 'form-control';
        }
        
        input.id = `field-${field.name}`;
        input.addEventListener('change', () => this.updateFormData(field.name, input, field.type));
        input.addEventListener('input', () => this.updateFormData(field.name, input, field.type));
        
        fieldGroup.appendChild(input);
        
        // Description
        if (field.description) {
            const desc = document.createElement('div');
            desc.className = 'field-description';
            desc.textContent = field.description;
            fieldGroup.appendChild(desc);
        }
        
        // Error container
        const errorDiv = document.createElement('div');
        errorDiv.className = 'field-error';
        errorDiv.id = `error-${field.name}`;
        fieldGroup.appendChild(errorDiv);
        
        return fieldGroup;
    }

    updateFormData(fieldName, input, fieldType) {
        let value = input.value;
        
        try {
            switch (fieldType) {
                case 'checkbox':
                    value = input.checked;
                    break;
                case 'number':
                    value = value === '' ? null : parseFloat(value);
                    break;
                case 'list':
                    value = value.trim() ? JSON.parse(value) : [];
                    break;
                case 'keyvalue':
                    value = value.trim() ? JSON.parse(value) : {};
                    break;
            }
            
            this.currentFormData[fieldName] = value;
            this.clearFieldError(fieldName);
            
        } catch (error) {
            this.showFieldError(fieldName, `Invalid ${fieldType}: ${error.message}`);
        }
    }

    showFieldError(fieldName, message) {
        const errorDiv = document.getElementById(`error-${fieldName}`);
        if (errorDiv) {
            errorDiv.textContent = message;
        }
    }

    clearFieldError(fieldName) {
        const errorDiv = document.getElementById(`error-${fieldName}`);
        if (errorDiv) {
            errorDiv.textContent = '';
        }
    }

    async saveConfiguration() {
        if (!this.currentConfig) {
            this.showStatus('No configuration to save', 'warning');
            return;
        }
        
        // Clear previous validation errors
        this.clearFormErrors();
        
        this.showLoading(true);
        
        try {
            const response = await fetch(`${this.apiBase}/save-config`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    config_class_name: this.currentConfig.config_class_name,
                    form_data: this.currentFormData
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                
                // Handle Pydantic validation errors specifically
                if (response.status === 422 && errorData.detail?.error_type === 'validation_error') {
                    this.handlePydanticValidationErrors(errorData.detail.validation_errors);
                    this.showStatus('Please fix the validation errors below', 'error');
                    return;
                } else {
                    throw new Error(errorData.detail?.message || `HTTP ${response.status}: ${response.statusText}`);
                }
            }
            
            const result = await response.json();
            
            // Update results display
            this.displayResults(result);
            
            this.showStatus('Configuration saved successfully!', 'success');
            this.markFormClean();
            
        } catch (error) {
            console.error('Save error:', error);
            this.showStatus(`Save failed: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    cancelConfiguration() {
        this.showConfigFormSection(false);
        this.currentConfig = null;
        this.currentFormData = {};
        this.showStatus('Configuration cancelled', 'info');
    }

    exportConfiguration() {
        if (!this.currentFormData || Object.keys(this.currentFormData).length === 0) {
            this.showStatus('No configuration data to export', 'warning');
            return;
        }
        
        const dataStr = JSON.stringify(this.currentFormData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `${this.currentConfig?.config_class_name || 'config'}.json`;
        link.click();
        
        this.showStatus('Configuration exported', 'success');
    }

    async createPipelineWizard() {
        const dagText = document.getElementById('dag-definition').value.trim();
        
        if (!dagText) {
            this.showStatus('Please provide a DAG definition', 'warning');
            return;
        }
        
        this.showLoading(true);
        
        try {
            const dag = JSON.parse(dagText);
            
            // Step 1: Analyze DAG to discover required configurations
            const analysisResult = await this.analyzePipelineDAG(dag);
            
            // Step 2: Display analysis results to user
            this.displayDAGAnalysis(analysisResult);
            
            // Step 3: Initialize workflow steps
            this.workflowSteps = analysisResult.workflow_steps;
            this.currentWorkflowStep = 0;
            
            // Step 4: Start configuration workflow
            this.startConfigurationWorkflow();
            
            this.showStatus('Pipeline analysis complete - starting workflow', 'success');
            
        } catch (error) {
            console.error('Pipeline wizard error:', error);
            this.showStatus(`Pipeline wizard creation failed: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }
    
    async analyzePipelineDAG(pipelineDAG) {
        // Analyze PipelineDAG to discover required configurations
        
        const response = await fetch(`${this.apiBase}/analyze-dag`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                pipeline_dag: pipelineDAG,
                base_config: this.getBaseConfigFromForm()
            })
        });
        
        if (!response.ok) {
            throw new Error(`DAG analysis failed: HTTP ${response.status}`);
        }
        
        return await response.json();
    }
    
    displayDAGAnalysis(analysisResult) {
        // Display DAG analysis results to user
        
        const container = document.getElementById('pipeline-wizard-container');
        
        container.innerHTML = `
            <div class="dag-analysis-results">
                <h3>📊 Pipeline Analysis Results</h3>
                
                <div class="discovered-steps">
                    <h4>🔍 Discovered Pipeline Steps:</h4>
                    <ul class="step-list">
                        ${analysisResult.discovered_steps.map(step => 
                            `<li><strong>${step.step_name}</strong> (${step.step_type})</li>`
                        ).join('')}
                    </ul>
                </div>
                
                <div class="required-configs">
                    <h4>⚙️ Required Configurations (Only These Will Be Shown):</h4>
                    <ul class="config-list">
                        ${analysisResult.required_configs.map(config => 
                            `<li>✅ <strong>${config.config_class_name}</strong></li>`
                        ).join('')}
                    </ul>
                    <p class="hidden-configs">❌ Hidden: ${analysisResult.hidden_configs_count} other config types not needed</p>
                </div>
                
                <div class="workflow-summary">
                    <h4>📋 Configuration Workflow:</h4>
                    <p>Base Config → Processing Config → ${analysisResult.required_configs.length} Specific Configs</p>
                    <div class="workflow-progress">
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: 0%"></div>
                        </div>
                        <div class="progress-text">Ready to start (0/${analysisResult.total_steps} steps)</div>
                    </div>
                </div>
                
                <div class="workflow-actions">
                    <button class="btn btn-primary" onclick="window.cursusUI.startConfigurationWorkflow()">
                        Start Configuration Workflow
                    </button>
                    <button class="btn btn-secondary" onclick="window.cursusUI.resetPipelineWizard()">
                        Reset
                    </button>
                </div>
            </div>
        `;
    }
    
    startConfigurationWorkflow() {
        // Start the step-by-step configuration workflow
        
        if (!this.workflowSteps || this.workflowSteps.length === 0) {
            this.showStatus('No workflow steps available', 'error');
            return;
        }
        
        this.currentWorkflowStep = 0;
        this.workflowData = {};
        this.renderCurrentWorkflowStep();
    }
    
    async renderCurrentWorkflowStep() {
        // Render the current configuration step
        
        if (this.currentWorkflowStep >= this.workflowSteps.length) {
            this.renderWorkflowCompletion();
            return;
        }
        
        const step = this.workflowSteps[this.currentWorkflowStep];
        const container = document.getElementById('pipeline-wizard-container');
        
        // Render step header
        container.innerHTML = `
            <div class="workflow-step-container">
                <div class="workflow-header">
                    <h2>🏗️ Configuration Workflow - Step ${step.step_number} of ${this.workflowSteps.length}</h2>
                    <h3>📋 ${step.title}</h3>
                    <div class="workflow-progress">
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${(this.currentWorkflowStep / this.workflowSteps.length) * 100}%"></div>
                        </div>
                        <div class="progress-text">Step ${this.currentWorkflowStep + 1} of ${this.workflowSteps.length}</div>
                    </div>
                </div>
                <div id="workflow-step-content" class="workflow-step-content"></div>
                <div class="workflow-navigation">
                    <button class="btn btn-secondary" onclick="window.cursusUI.previousWorkflowStep()" ${this.currentWorkflowStep === 0 ? 'disabled' : ''}>
                        ← Previous Step
                    </button>
                    <button class="btn btn-primary" onclick="window.cursusUI.nextWorkflowStep()">
                        Continue to Next Step →
                    </button>
                </div>
            </div>
        `;
        
        // Render step-specific content
        await this.renderWorkflowStepContent(step);
    }
    
    async renderWorkflowStepContent(step) {
        // Render content for a specific workflow step
        
        const stepContent = document.getElementById('workflow-step-content');
        
        if (step.type === 'base') {
            await this.renderBaseConfigStep(stepContent);
        } else if (step.type === 'processing') {
            await this.renderProcessingConfigStep(stepContent);
        } else if (step.type === 'specific') {
            await this.renderSpecificConfigStep(stepContent, step);
        }
    }
    
    async renderSpecificConfigStep(container, step) {
        // Render a specific configuration step
        
        if (step.is_specialized) {
            // Handle specialized configurations (e.g., CradleDataLoadConfig)
            container.innerHTML = `
                <div class="specialized-config-step">
                    <h4>🎛️ Specialized Configuration</h4>
                    <p>This step uses a specialized wizard interface:</p>
                    <div class="specialized-features">
                        <ul>
                            <li>1️⃣ Data Sources Configuration with field discovery</li>
                            <li>2️⃣ Transform Specification with workflow context</li>
                            <li>3️⃣ Output Configuration with inheritance chain</li>
                            <li>4️⃣ Cradle Job Settings with DAG integration</li>
                        </ul>
                    </div>
                    <button class="btn btn-primary" onclick="window.cursusUI.openSpecializedWizard('${step.config_class_name}')">
                        Open ${step.config_class_name} Wizard
                    </button>
                    <p class="workflow-note"><small>(Base config will be pre-filled automatically)</small></p>
                </div>
            `;
        } else {
            // Handle standard configurations with 3-tier field categorization
            await this.renderStandardConfigStepWithTiers(container, step);
        }
    }
    
    async renderStandardConfigStepWithTiers(container, step) {
        // Render standard configuration step with 3-tier field categorization
        
        try {
            // Fetch configuration data with field categories
            const configData = await this.makeRequest(`${this.apiBase}/create-widget`, {
                method: 'POST',
                body: JSON.stringify({
                    config_class_name: step.config_class_name,
                    base_config: this.getWorkflowInheritedConfig(),
                    workflow_context: this.getWorkflowContext()
                })
            });
            
            // Render with 3-tier categorization
            this.renderConfigWithTiers(container, configData, step.config_class_name);
            
        } catch (error) {
            console.error(`Error loading step ${step.config_class_name}:`, error);
            container.innerHTML = `
                <div class="config-error">
                    <h4>❌ Configuration Error</h4>
                    <p>Error loading ${step.config_class_name}: ${error.message}</p>
                </div>
            `;
        }
    }
    
    renderConfigWithTiers(container, configData, configName) {
        // Render configuration form with 3-tier field categorization
        
        const fieldCategories = configData.field_categories || {
            essential: [],
            system: [],
            derived: []
        };
        
        let formHTML = '<div class="tiered-config-form">';
        
        // Tier 1: Essential Fields (Required, no defaults)
        if (fieldCategories.essential && fieldCategories.essential.length > 0) {
            formHTML += `
                <div class="field-tier essential-tier">
                    <h4>🔥 Essential Configuration (Tier 1)</h4>
                    <p class="tier-description">Required fields that you must fill - no defaults available</p>
                    <div class="tier-fields">
                        ${this.renderTierFields(configData.fields, fieldCategories.essential, configData.values, 'essential')}
                    </div>
                </div>
            `;
        }
        
        // Tier 2: System Fields (Optional with defaults)
        if (fieldCategories.system && fieldCategories.system.length > 0) {
            formHTML += `
                <div class="field-tier system-tier">
                    <h4>⚙️ System Configuration (Tier 2)</h4>
                    <p class="tier-description">Optional fields with defaults - modify if needed for customization</p>
                    <div class="tier-fields">
                        ${this.renderTierFields(configData.fields, fieldCategories.system, configData.values, 'system')}
                    </div>
                </div>
            `;
        }
        
        // Tier 3: Derived Fields (HIDDEN from UI completely)
        // These are never displayed to users as per design specs
        
        // Workflow Context Display
        if (this.workflowContext && Object.keys(this.workflowContext).length > 0) {
            formHTML += `
                <div class="workflow-context-display">
                    <h4>💾 Inherited from Workflow Context</h4>
                    <div class="inherited-values">
                        ${this.renderInheritedValues()}
                    </div>
                </div>
            `;
        }
        
        formHTML += '</div>';
        
        container.innerHTML = formHTML;
        
        // Initialize form data for this step
        if (!this.workflowData[configName]) {
            this.workflowData[configName] = { ...configData.values };
        }
        
        // Bind event listeners for form fields
        this.bindWorkflowFormEvents(configName);
    }
    
    renderTierFields(allFields, tierFieldNames, values, tierType) {
        // Render fields for a specific tier
        
        const tierFields = allFields.filter(field => tierFieldNames.includes(field.name));
        let fieldsHTML = '<div class="form-row">';
        
        tierFields.forEach((field, index) => {
            if (index > 0 && index % 2 === 0) {
                fieldsHTML += '</div><div class="form-row">';
            }
            
            const value = values[field.name] !== undefined ? values[field.name] : '';
            fieldsHTML += this.renderTierField(field, value, tierType);
        });
        
        fieldsHTML += '</div>';
        return fieldsHTML;
    }
    
    renderTierField(field, currentValue, tierType) {
        // Render a single field with tier-specific styling
        
        const tierClass = `tier-${tierType}`;
        const requiredIndicator = field.required ? ' *' : '';
        const value = currentValue !== undefined ? currentValue : '';
        
        let inputHTML = '';
        
        switch (field.type) {
            case 'checkbox':
                inputHTML = `<input type="checkbox" ${Boolean(value) ? 'checked' : ''} class="form-control ${tierClass}" id="field-${field.name}">`;
                break;
            case 'number':
                inputHTML = `<input type="number" step="any" value="${value}" class="form-control ${tierClass}" id="field-${field.name}">`;
                break;
            case 'list':
                const listValue = Array.isArray(value) ? JSON.stringify(value, null, 2) : value;
                inputHTML = `<textarea rows="3" class="form-control ${tierClass}" id="field-${field.name}" placeholder='["item1", "item2"]'>${listValue}</textarea>`;
                break;
            case 'keyvalue':
                const kvValue = typeof value === 'object' ? JSON.stringify(value, null, 2) : value;
                inputHTML = `<textarea rows="4" class="form-control ${tierClass}" id="field-${field.name}" placeholder='{"key": "value"}'>${kvValue}</textarea>`;
                break;
            default:
                inputHTML = `<input type="text" value="${value}" class="form-control ${tierClass}" id="field-${field.name}">`;
        }
        
        return `
            <div class="field-group ${field.required ? 'required' : ''} ${tierClass}">
                <label class="form-label" for="field-${field.name}">
                    ${field.name}${requiredIndicator}
                    <span class="tier-badge tier-${tierType}">${tierType.toUpperCase()}</span>
                </label>
                ${inputHTML}
                ${field.description ? `<div class="field-description">${field.description}</div>` : ''}
                <div class="field-error" id="error-${field.name}"></div>
            </div>
        `;
    }
    
    renderInheritedValues() {
        // Render inherited values from workflow context
        
        if (!this.workflowContext || !this.workflowContext.inherited_values) {
            return '<p>No inherited values available</p>';
        }
        
        const inherited = this.workflowContext.inherited_values;
        let inheritedHTML = '<ul class="inherited-list">';
        
        Object.entries(inherited).forEach(([key, value]) => {
            inheritedHTML += `<li><strong>${key}:</strong> ${value}</li>`;
        });
        
        inheritedHTML += '</ul>';
        return inheritedHTML;
    }
    
    bindWorkflowFormEvents(configName) {
        // Bind event listeners for workflow form fields
        
        const formFields = document.querySelectorAll('#workflow-step-content .form-control');
        formFields.forEach(field => {
            const fieldName = field.id.replace('field-', '');
            
            field.addEventListener('change', () => this.updateWorkflowFormData(configName, fieldName, field));
            field.addEventListener('input', () => this.updateWorkflowFormData(configName, fieldName, field));
        });
    }
    
    updateWorkflowFormData(configName, fieldName, input) {
        // Update workflow form data for a specific field
        
        if (!this.workflowData[configName]) {
            this.workflowData[configName] = {};
        }
        
        let value = input.value;
        
        try {
            // Handle different field types
            if (input.type === 'checkbox') {
                value = input.checked;
            } else if (input.type === 'number') {
                value = value === '' ? null : parseFloat(value);
            } else if (input.tagName === 'TEXTAREA') {
                // Try to parse as JSON for list/keyvalue fields
                if (value.trim().startsWith('[') || value.trim().startsWith('{')) {
                    value = value.trim() ? JSON.parse(value) : (value.trim().startsWith('[') ? [] : {});
                }
            }
            
            this.workflowData[configName][fieldName] = value;
            this.clearWorkflowFieldError(fieldName);
            this.markFormDirty();
            
        } catch (error) {
            this.showWorkflowFieldError(fieldName, `Invalid JSON: ${error.message}`);
        }
    }
    
    showWorkflowFieldError(fieldName, message) {
        // Show error for a workflow field
        const errorDiv = document.getElementById(`error-${fieldName}`);
        if (errorDiv) {
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
        }
    }
    
    clearWorkflowFieldError(fieldName) {
        // Clear error for a workflow field
        const errorDiv = document.getElementById(`error-${fieldName}`);
        if (errorDiv) {
            errorDiv.textContent = '';
            errorDiv.style.display = 'none';
        }
    }
    
    nextWorkflowStep() {
        // Move to the next workflow step
        
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
        // Move to the previous workflow step
        
        if (this.currentWorkflowStep > 0) {
            this.currentWorkflowStep--;
            this.renderCurrentWorkflowStep();
        }
    }
    
    validateCurrentWorkflowStep() {
        // Validate the current workflow step
        
        // Basic validation - check for required fields
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
        // Save the current workflow step data
        
        const currentStep = this.workflowSteps[this.currentWorkflowStep];
        if (currentStep && this.workflowData[currentStep.config_class_name]) {
            // Mark step as completed
            currentStep.completed = true;
            console.log(`Step ${currentStep.title} completed with data:`, this.workflowData[currentStep.config_class_name]);
        }
    }
    
    renderWorkflowCompletion() {
        // Render the workflow completion summary
        
        const container = document.getElementById('pipeline-wizard-container');
        
        const completedSteps = this.workflowSteps.filter(step => step.completed);
        
        container.innerHTML = `
            <div class="workflow-completion">
                <h2>✅ Configuration Complete - All Steps Configured</h2>
                
                <div class="completion-summary">
                    <h3>📋 Configuration Summary:</h3>
                    <ul class="completed-configs">
                        ${this.workflowSteps.map(step => 
                            `<li class="${step.completed ? 'completed' : 'incomplete'}">
                                ${step.completed ? '✅' : '⏳'} ${step.title}
                            </li>`
                        ).join('')}
                    </ul>
                </div>
                
                <div class="workflow-results">
                    <h3>🎯 Ready for Pipeline Execution:</h3>
                    <div class="config-code-preview">
                        <pre><code>config_list = [
    ${Object.keys(this.workflowData).map(configName => `    ${configName.toLowerCase()}_config`).join(',\n')}
]</code></pre>
                    </div>
                </div>
                
                <div class="export-options">
                    <h3>💾 Export Options:</h3>
                    <div class="export-buttons">
                        <button class="btn btn-success btn-large" onclick="window.cursusUI.saveAllMerged()">
                            💾 Save All Merged
                            <small>Creates unified hierarchical JSON like demo_config.ipynb (Recommended)</small>
                        </button>
                        <button class="btn btn-info" onclick="window.cursusUI.exportIndividualConfigs()">
                            📤 Export Individual
                            <small>Individual JSON files for each configuration</small>
                        </button>
                    </div>
                </div>
                
                <div class="completion-actions">
                    <button class="btn btn-primary" onclick="window.cursusUI.executeWorkflowPipeline()">
                        🚀 Execute Pipeline
                    </button>
                    <button class="btn btn-secondary" onclick="window.cursusUI.saveWorkflowAsTemplate()">
                        📋 Save as Template
                    </button>
                    <button class="btn btn-info" onclick="window.cursusUI.modifyWorkflowConfiguration()">
                        🔧 Modify Configuration
                    </button>
                </div>
            </div>
        `;
    }

    getBaseConfigFromForm() {
        const baseConfigText = document.getElementById('base-config').value.trim();
        if (baseConfigText) {
            try {
                return JSON.parse(baseConfigText);
            } catch (e) {
                return null;
            }
        }
        return null;
    }

    renderPipelineWizard(wizardData) {
        const container = document.getElementById('pipeline-wizard-container');
        container.innerHTML = `
            <div class="pipeline-wizard-content">
                <h3>Pipeline Configuration Wizard</h3>
                <p>Multi-step wizard with ${wizardData.steps?.length || 0} configuration steps.</p>
                <div class="mt-3">
                    <p><strong>Note:</strong> Pipeline wizards are best experienced in Jupyter notebooks.</p>
                    <button class="btn btn-primary" onclick="this.openJupyterExample()">
                        View Jupyter Example
                    </button>
                </div>
            </div>
        `;
    }

    displayResults(result) {
        // JSON tab
        document.getElementById('json-output').textContent = JSON.stringify(result.config, null, 2);
        
        // Python tab
        if (result.python_code) {
            document.getElementById('python-output').textContent = result.python_code;
        }
        
        // Summary tab
        const summaryDiv = document.getElementById('summary-output');
        summaryDiv.innerHTML = `
            <div class="config-summary">
                <h4>Configuration Summary</h4>
                <ul>
                    <li><strong>Type:</strong> ${result.config_type}</li>
                    <li><strong>Fields:</strong> ${Object.keys(result.config).length}</li>
                    <li><strong>Created:</strong> ${new Date().toLocaleString()}</li>
                </ul>
            </div>
        `;
    }

    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabName);
        });
        
        // Update tab panes
        document.querySelectorAll('.tab-pane').forEach(pane => {
            pane.classList.toggle('active', pane.id === `${tabName}-tab`);
        });
    }

    showConfigFormSection(show) {
        const section = document.getElementById('config-form-section');
        section.style.display = show ? 'block' : 'none';
        
        if (show) {
            section.scrollIntoView({ behavior: 'smooth' });
        }
    }

    showLoading(show) {
        const overlay = document.getElementById('loading-overlay');
        overlay.style.display = show ? 'flex' : 'none';
    }

    showStatus(message, type = 'info') {
        const container = document.getElementById('status-messages');
        
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

    openJupyterExample() {
        // This would open a Jupyter notebook example
        window.open('/jupyter-example', '_blank');
    }

    // Enhanced error handling
    async handleApiError(error, context) {
        console.error(`${context} error:`, error);
        
        let userMessage = `${context} failed`;
        if (error.message) {
            userMessage += `: ${error.message}`;
        }
        
        this.showStatus(userMessage, 'error');
    }

    // Enhanced field validation with debouncing
    createEnhancedFormField(field, currentValue) {
        const fieldGroup = this.createFormField(field, currentValue);
        
        // Add real-time validation with debouncing
        const input = fieldGroup.querySelector('input, textarea, select');
        if (input) {
            const debouncedValidation = this.debounce((fieldName, value, fieldConfig) => {
                this.validateFieldValue(fieldName, value, fieldConfig);
            }, 300);
            
            input.addEventListener('input', (e) => {
                this.isDirty = true;
                debouncedValidation(field.name, e.target.value, field);
            });
            
            input.addEventListener('blur', (e) => {
                // Immediate validation on blur
                this.validateFieldValue(field.name, e.target.value, field);
            });
        }
        
        return fieldGroup;
    }

    // Field validation logic
    validateFieldValue(fieldName, value, fieldConfig) {
        const errors = [];
        
        // Required field validation
        if (fieldConfig.required && (!value || value.toString().trim() === '')) {
            errors.push(`${fieldName} is required`);
        }
        
        // Type-specific validation
        if (value && value.toString().trim() !== '') {
            switch (fieldConfig.type) {
                case 'number':
                    if (isNaN(parseFloat(value))) {
                        errors.push(`${fieldName} must be a valid number`);
                    }
                    break;
                    
                case 'list':
                    try {
                        const parsed = JSON.parse(value);
                        if (!Array.isArray(parsed)) {
                            errors.push(`${fieldName} must be a valid JSON array`);
                        }
                    } catch (e) {
                        errors.push(`${fieldName} must be valid JSON`);
                    }
                    break;
                    
                case 'keyvalue':
                    try {
                        const parsed = JSON.parse(value);
                        if (typeof parsed !== 'object' || Array.isArray(parsed)) {
                            errors.push(`${fieldName} must be a valid JSON object`);
                        }
                    } catch (e) {
                        errors.push(`${fieldName} must be valid JSON`);
                    }
                    break;
            }
        }
        
        // Update validation state
        if (errors.length > 0) {
            this.validationErrors[fieldName] = errors;
            this.showFieldError(fieldName, errors[0]);
        } else {
            delete this.validationErrors[fieldName];
            this.clearFieldError(fieldName);
        }
        
        return errors.length === 0;
    }

    // Clear all form errors
    clearFormErrors() {
        this.validationErrors = {};
        document.querySelectorAll('.field-error').forEach(errorDiv => {
            errorDiv.textContent = '';
        });
    }

    // Check if form has validation errors
    hasValidationErrors() {
        return Object.keys(this.validationErrors).length > 0;
    }

    // Enhanced form state management
    markFormDirty() {
        this.isDirty = true;
    }

    markFormClean() {
        this.isDirty = false;
    }

    // Check unsaved changes before navigation
    checkUnsavedChanges() {
        if (this.isDirty) {
            return confirm('You have unsaved changes. Are you sure you want to continue?');
        }
        return true;
    }

    // Enhanced status message system
    showEnhancedStatus(message, type = 'info', duration = 5000, dismissible = true) {
        const container = document.getElementById('status-messages');
        const statusId = `status-${Date.now()}`;
        
        const statusDiv = document.createElement('div');
        statusDiv.className = `status-message ${type}`;
        statusDiv.id = statusId;
        statusDiv.innerHTML = `
            <span class="status-text">${message}</span>
            ${dismissible ? '<button class="status-close" onclick="this.parentElement.remove()">×</button>' : ''}
        `;
        
        container.appendChild(statusDiv);
        
        // Auto-remove after duration
        if (duration > 0) {
            setTimeout(() => {
                const element = document.getElementById(statusId);
                if (element && element.parentNode) {
                    element.parentNode.removeChild(element);
                }
            }, duration);
        }
        
        return statusId;
    }

    // Remove specific status message
    removeStatus(statusId) {
        const element = document.getElementById(statusId);
        if (element && element.parentNode) {
            element.parentNode.removeChild(element);
        }
    }

    // Handle Pydantic validation errors from backend
    handlePydanticValidationErrors(validationErrors) {
        console.log('Handling Pydantic validation errors:', validationErrors);
        
        // Clear existing errors first
        this.clearFormErrors();
        
        // Display each validation error on the corresponding field
        validationErrors.forEach(error => {
            const fieldName = error.field;
            const message = error.message;
            const errorType = error.type;
            
            // Format user-friendly error message
            let userMessage = message;
            if (errorType === 'missing') {
                userMessage = `${fieldName} is required`;
            } else if (errorType === 'value_error') {
                userMessage = `Invalid value for ${fieldName}: ${message}`;
            } else if (errorType === 'type_error') {
                userMessage = `Wrong type for ${fieldName}: ${message}`;
            }
            
            // Show error on the specific field
            this.showFieldError(fieldName, userMessage);
            
            // Add to validation errors state
            this.validationErrors[fieldName] = [userMessage];
            
            // Highlight the field with error
            const fieldInput = document.getElementById(`field-${fieldName}`);
            if (fieldInput) {
                fieldInput.classList.add('error');
                fieldInput.addEventListener('input', () => {
                    fieldInput.classList.remove('error');
                    this.clearFieldError(fieldName);
                }, { once: true });
            }
        });
        
        // Scroll to first error field
        if (validationErrors.length > 0) {
            const firstErrorField = document.getElementById(`field-${validationErrors[0].field}`);
            if (firstErrorField) {
                firstErrorField.scrollIntoView({ behavior: 'smooth', block: 'center' });
                firstErrorField.focus();
            }
        }
    }

    // Save All Merged - Main functionality replicating demo_config.ipynb experience
    async saveAllMerged() {
        try {
            // Collect all configurations from current session
            const sessionConfigs = this.getAllSessionConfigs();
            
            if (Object.keys(sessionConfigs).length === 0) {
                this.showStatus('No configurations to merge. Please configure at least one component.', 'warning');
                return;
            }
            
            this.showLoading(true);
            this.showStatus('Creating unified configuration file...', 'info');
            
            // Call the merge endpoint
            const response = await fetch(`${this.apiBase}/merge-and-save-configs`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_configs: sessionConfigs,
                    filename: null, // Let backend generate filename
                    workspace_dirs: null
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            
            // Display the merge results
            this.displayMergeResults(result);
            
            this.showStatus(`Successfully merged ${Object.keys(sessionConfigs).length} configurations into ${result.filename}`, 'success');
            
        } catch (error) {
            console.error('Save All Merged failed:', error);
            this.showStatus(`Merge failed: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }
    
    getAllSessionConfigs() {
        // Collect all configurations from the current session
        const sessionConfigs = {};
        
        // From workflow data (if using pipeline wizard)
        if (this.workflowData && Object.keys(this.workflowData).length > 0) {
            Object.assign(sessionConfigs, this.workflowData);
        }
        
        // From individual form data (if using discovery mode)
        if (this.currentFormData && Object.keys(this.currentFormData).length > 0) {
            Object.assign(sessionConfigs, this.currentFormData);
        }
        
        return sessionConfigs;
    }
    
    displayMergeResults(result) {
        // Display the merge results with file save dialog
        const container = document.getElementById('pipeline-wizard-container') || document.getElementById('config-list');
        
        // Generate smart default filename based on configuration data
        const defaultFilename = this.generateSmartFilename(result.merged_config);
        
        container.innerHTML = `
            <div class="merge-results">
                <h2>💾 Save All Merged - Configuration Export Complete</h2>
                
                <div class="merge-success">
                    <h3>🎯 Unified Configuration Created</h3>
                    <div class="merge-details">
                        <div class="merge-info">
                            <h4>📁 Generated File:</h4>
                            <div class="file-info">
                                <span class="filename">📄 ${result.filename}</span>
                                <p class="file-description">Hierarchical JSON structure with shared vs specific fields</p>
                            </div>
                        </div>
                        
                        <div class="merge-structure">
                            <h4>📊 Structure Preview:</h4>
                            <div class="structure-preview">
                                <pre><code>{
  "shared": { /* Common fields across all configs */ },
  "processing_shared": { /* Processing step fields */ },
  "specific": {
    ${Object.keys(result.merged_config.specific || {}).map(key => 
                        `    "${key}": { /* Step-specific fields */ }`
                    ).join(',\n')}
  },
  "inverted_index": { /* Field → Steps mapping */ },
  "step_list": [ /* All pipeline steps */ ]
}</code></pre>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="file-save-dialog">
                    <h3>💾 Save Configuration File</h3>
                    <div class="save-options">
                        <div class="save-field-group">
                            <label for="save-filename" class="form-label">📄 Filename:</label>
                            <input type="text" id="save-filename" class="form-control" value="${defaultFilename}" placeholder="config_service_region.json">
                            <div class="field-description">Default format: config_{service_name}_{region}.json</div>
                        </div>
                        
                        <div class="save-field-group">
                            <label for="save-location" class="form-label">📁 Save Location:</label>
                            <select id="save-location" class="form-control">
                                <option value="current">📂 Current Directory (where Jupyter notebook runs)</option>
                                <option value="downloads">⬇️ Downloads Folder</option>
                                <option value="custom">📁 Custom Location (browser default)</option>
                            </select>
                            <div class="field-description">Current directory is recommended for Jupyter notebook workflows</div>
                        </div>
                        
                        <div class="save-preview">
                            <h4>💡 Save Preview:</h4>
                            <div class="preview-info">
                                <span id="save-preview-text">Will save as: <strong>${defaultFilename}</strong> in current directory</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="download-actions">
                    <h3>✅ Ready for Pipeline Execution!</h3>
                    <div class="action-buttons">
                        <button class="btn btn-success btn-large" onclick="window.cursusUI.downloadMergedConfigWithOptions('${result.download_url}')">
                            💾 Save Configuration File
                        </button>
                        <button class="btn btn-info" onclick="window.cursusUI.previewMergedJSON(${JSON.stringify(result.merged_config).replace(/"/g, '&quot;')})">
                            👁️ Preview JSON
                        </button>
                        <button class="btn btn-secondary" onclick="window.cursusUI.copyMergedToClipboard(${JSON.stringify(result.merged_config).replace(/"/g, '&quot;')})">
                            📋 Copy to Clipboard
                        </button>
                    </div>
                </div>
                
                <div class="next-steps">
                    <h4>🚀 Next Steps:</h4>
                    <p>Use this unified configuration file in your pipeline execution:</p>
                    <div class="code-example">
                        <pre><code># Load the merged configuration
from cursus.core.config_fields import load_configs
config_list = load_configs("${defaultFilename}")

# Execute your pipeline
pipeline.run(config_list)</code></pre>
                    </div>
                </div>
            </div>
        `;
        
        // Bind event listeners for the save dialog
        this.bindSaveDialogEvents();
    }
    
    async downloadMergedConfig(downloadUrl, filename) {
        try {
            // Create a temporary link to trigger download
            const link = document.createElement('a');
            link.href = downloadUrl;
            link.download = filename;
            link.style.display = 'none';
            
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            this.showStatus(`Downloaded ${filename}`, 'success');
            
        } catch (error) {
            console.error('Download failed:', error);
            this.showStatus(`Download failed: ${error.message}`, 'error');
        }
    }
    
    previewMergedJSON(mergedConfig) {
        // Show JSON preview in a modal
        const modal = document.createElement('div');
        modal.className = 'json-preview-modal';
        modal.innerHTML = `
            <div class="modal-overlay" onclick="this.parentElement.remove()">
                <div class="modal-content" onclick="event.stopPropagation()">
                    <div class="modal-header">
                        <h3>👁️ Merged Configuration Preview</h3>
                        <button class="modal-close" onclick="this.closest('.json-preview-modal').remove()">×</button>
                    </div>
                    <div class="modal-body">
                        <pre><code>${JSON.stringify(mergedConfig, null, 2)}</code></pre>
                    </div>
                    <div class="modal-footer">
                        <button class="btn btn-secondary" onclick="this.closest('.json-preview-modal').remove()">Close</button>
                        <button class="btn btn-primary" onclick="window.cursusUI.copyMergedToClipboard(${JSON.stringify(mergedConfig).replace(/"/g, '&quot;')})">
                            📋 Copy to Clipboard
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
    }
    
    copyMergedToClipboard(mergedConfig) {
        const jsonString = JSON.stringify(mergedConfig, null, 2);
        
        navigator.clipboard.writeText(jsonString).then(() => {
            this.showStatus('Merged configuration copied to clipboard', 'success');
        }).catch(err => {
            console.error('Copy failed:', err);
            this.showStatus('Copy failed', 'error');
        });
    }
    
    generateSmartFilename(mergedConfig) {
        // Generate smart default filename based on configuration data
        // Format: config_{service_name}_{region}.json
        
        let serviceName = 'pipeline';
        let region = 'default';
        
        // Extract service_name and region from shared config
        if (mergedConfig && mergedConfig.shared) {
            serviceName = mergedConfig.shared.service_name || serviceName;
            region = mergedConfig.shared.region || region;
        }
        
        // Clean up service name and region for filename
        serviceName = serviceName.replace(/[^a-zA-Z0-9_-]/g, '_');
        region = region.replace(/[^a-zA-Z0-9_-]/g, '_');
        
        return `config_${serviceName}_${region}.json`;
    }
    
    bindSaveDialogEvents() {
        // Bind event listeners for the save dialog
        
        const filenameInput = document.getElementById('save-filename');
        const locationSelect = document.getElementById('save-location');
        const previewText = document.getElementById('save-preview-text');
        
        if (filenameInput && locationSelect && previewText) {
            const updatePreview = () => {
                const filename = filenameInput.value || 'config.json';
                const location = locationSelect.value;
                
                let locationText = 'current directory';
                if (location === 'downloads') {
                    locationText = 'Downloads folder';
                } else if (location === 'custom') {
                    locationText = 'browser default location';
                }
                
                previewText.innerHTML = `Will save as: <strong>${filename}</strong> in ${locationText}`;
            };
            
            filenameInput.addEventListener('input', updatePreview);
            locationSelect.addEventListener('change', updatePreview);
            
            // Initial preview update
            updatePreview();
        }
    }
    
    async downloadMergedConfigWithOptions(downloadUrl) {
        // Download merged config with user-specified options
        
        const filenameInput = document.getElementById('save-filename');
        const locationSelect = document.getElementById('save-location');
        
        if (!filenameInput || !locationSelect) {
            // Fallback to simple download
            this.downloadMergedConfig(downloadUrl, 'config.json');
            return;
        }
        
        const filename = filenameInput.value || 'config.json';
        const location = locationSelect.value;
        
        try {
            // Fetch the configuration data
            const response = await fetch(downloadUrl);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const configData = await response.json();
            const jsonString = JSON.stringify(configData, null, 2);
            
            if (location === 'current') {
                // For current directory, we need to use the File System Access API if available
                // or fall back to regular download
                if ('showSaveFilePicker' in window) {
                    try {
                        const fileHandle = await window.showSaveFilePicker({
                            suggestedName: filename,
                            types: [{
                                description: 'JSON files',
                                accept: { 'application/json': ['.json'] }
                            }]
                        });
                        
                        const writable = await fileHandle.createWritable();
                        await writable.write(jsonString);
                        await writable.close();
                        
                        this.showStatus(`Configuration saved as ${filename}`, 'success');
                        return;
                    } catch (err) {
                        if (err.name !== 'AbortError') {
                            console.error('File System Access API failed:', err);
                        }
                        // Fall through to regular download
                    }
                }
            }
            
            // Regular download (works for all browsers)
            const blob = new Blob([jsonString], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            
            const link = document.createElement('a');
            link.href = url;
            link.download = filename;
            link.style.display = 'none';
            
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            URL.revokeObjectURL(url);
            
            let locationText = 'Downloads folder';
            if (location === 'current') {
                locationText = 'default download location (current directory not supported by browser)';
            } else if (location === 'custom') {
                locationText = 'selected location';
            }
            
            this.showStatus(`Configuration downloaded as ${filename} to ${locationText}`, 'success');
            
        } catch (error) {
            console.error('Download with options failed:', error);
            this.showStatus(`Download failed: ${error.message}`, 'error');
        }
    }
    
    exportIndividualConfigs() {
        // Export individual configuration files (existing functionality)
        const sessionConfigs = this.getAllSessionConfigs();
        
        if (Object.keys(sessionConfigs).length === 0) {
            this.showStatus('No configurations to export', 'warning');
            return;
        }
        
        // Export each configuration as a separate file
        Object.entries(sessionConfigs).forEach(([configName, configData]) => {
            const dataStr = JSON.stringify(configData, null, 2);
            const dataBlob = new Blob([dataStr], { type: 'application/json' });
            
            const link = document.createElement('a');
            link.href = URL.createObjectURL(dataBlob);
            link.download = `${configName}.json`;
            link.click();
        });
        
        this.showStatus(`Exported ${Object.keys(sessionConfigs).length} individual configuration files`, 'success');
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.cursusUI = new CursusConfigUI();
});

// Utility functions
function formatJSON(obj) {
    return JSON.stringify(obj, null, 2);
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        window.cursusUI.showStatus('Copied to clipboard', 'success');
    }).catch(err => {
        console.error('Copy failed:', err);
        window.cursusUI.showStatus('Copy failed', 'error');
    });
}

// Export for global access
window.CursusConfigUI = CursusConfigUI;
