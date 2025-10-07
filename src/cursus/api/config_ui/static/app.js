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
            
            const response = await fetch(`${this.apiBase}/create-pipeline-wizard`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    dag: dag,
                    base_config: this.getBaseConfigFromForm()
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            this.renderPipelineWizard(result);
            
            this.showStatus('Pipeline wizard created successfully', 'success');
            
        } catch (error) {
            console.error('Pipeline wizard error:', error);
            this.showStatus(`Pipeline wizard creation failed: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
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
