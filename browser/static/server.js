"use strict";

// CONFIGURATION - Easy to add new controls here
const config = {
    // Define global text boxes
    textBoxes: [
        { id: 'classId', label: 'Class ID', placeholder: 'Enter class ID' },
        { id: 'exampleId', label: 'Example ID', placeholder: 'Enter example ID' },
    ],
    
    // Define sliders
    sliders: [
        { id: 'nu', label: 'Nu', min: 0, max: 100, value: 50 },
    ],
    
    // Define output logs with flexible controls
    outputLogs: [
        { 
            id: 'log1', 
            label: 'Output Log 1',
            controls: [
                { type: 'text', id: 'experimentName', label: 'Experiment Name', placeholder: 'Enter experiment name' },
                // { type: 'toggle', id: 'enableFeature', label: 'Enable Feature', value: false }
                { type: 'slider', id: 'numFeatures', label: 'Number of Features', min: 0, max: 100, value: 50 },
                { type: 'slider', id: 'numDims', label: 'Number of Features', min: 0, max: 100, value: 50 }
            ],
            imagePaths: []
        },
        { 
            id: 'log2', 
            label: 'Output Log 2',
            controls: [
                { type: 'text', id: 'experimentName', label: 'Experiment Name', placeholder: 'Enter experiment name' },
            ],
            imagePaths: []
        }
    ]
};

// Enhanced State management with caching
const appState = {
    abortControllers: new Map(),
    updateTimeouts: new Map(),
    sliderDomains: new Map(),
    imageCache: new Map(), // Cache for storing image data
    cacheHits: 0,
    cacheMisses: 0
};

// Cache configuration
const CACHE_CONFIG = {
    maxSize: 100, // Maximum number of parameter sets to cache
    preloadNeighbors: true, // Preload similar parameter sets
    preloadRadius: 2 // How many neighboring values to preload
};

// Initialize the page
document.addEventListener('DOMContentLoaded', async function() {
    // Generate controls from configuration
    generateGlobalControls();
    generateOutputLogs();
    
    // Set up domains for ALL sliders after everything is generated
    await setupAllSlidersWithDomains();
    
    // Add cache stats display
    addCacheStatsDisplay();
    
    // Initialize all output logs
    updateAllOutputLogs();
});

// Add cache statistics display
function addCacheStatsDisplay() {
    const statsHTML = `
        <div class="cache-stats" style="position: fixed; bottom: 10px; right: 10px; background: #f0f0f0; padding: 10px; border-radius: 5px; font-size: 12px;">
            <div>Cache Hits: <span id="cacheHits">0</span></div>
            <div>Cache Misses: <span id="cacheMisses">0</span></div>
            <div>Hit Rate: <span id="hitRate">0%</span></div>
            <div>Cache Size: <span id="cacheSize">0</span></div>
        </div>
    `;
    document.body.insertAdjacentHTML('beforeend', statsHTML);
}

// Update cache statistics display
function updateCacheStats() {
    const total = appState.cacheHits + appState.cacheMisses;
    const hitRate = total > 0 ? ((appState.cacheHits / total) * 100).toFixed(1) : 0;
    
    document.getElementById('cacheHits').textContent = appState.cacheHits;
    document.getElementById('cacheMisses').textContent = appState.cacheMisses;
    document.getElementById('hitRate').textContent = `${hitRate}%`;
    document.getElementById('cacheSize').textContent = appState.imageCache.size;
}

// Generate cache key from parameters
function generateCacheKey(parameters) {
    // Create a stable string representation of the parameters
    const sortedParams = {};
    Object.keys(parameters).sort().forEach(key => {
        sortedParams[key] = parameters[key];
    })
    return JSON.stringify(sortedParams);
}

// Get cached images for parameters
function getCachedImages(parameters) {
    const cacheKey = generateCacheKey(parameters);
    return appState.imageCache.get(cacheKey);
}

// Store images in cache
function cacheImages(parameters, imageUrls) {
    const cacheKey = generateCacheKey(parameters);
    
    // Implement cache eviction if we exceed max size
    if (appState.imageCache.size >= CACHE_CONFIG.maxSize) {
        const firstKey = appState.imageCache.keys().next().value;
        appState.imageCache.delete(firstKey);
    }
    
    appState.imageCache.set(cacheKey, {
        urls: imageUrls,
        timestamp: Date.now(),
        accessCount: 0
    });
    
    updateCacheStats();
}

// Preload images for neighboring parameter values
async function preloadNeighborImages(currentParams, sliderId, currentValue) {
    if (!CACHE_CONFIG.preloadNeighbors) return;
    
    const domain = appState.sliderDomains.get(sliderId);
    if (!domain) return;
    
    const currentIndex = domain.indexOf(parseFloat(currentValue));
    if (currentIndex === -1) return;
    
    const start = Math.max(0, currentIndex - CACHE_CONFIG.preloadRadius);
    const end = Math.min(domain.length - 1, currentIndex + CACHE_CONFIG.preloadRadius);
    
    const preloadPromises = [];
    
    for (let i = start; i <= end; i++) {
        if (i === currentIndex) continue; // Skip current value
        
        const neighborValue = domain[i];
        const neighborParams = { ...currentParams, [sliderId]: neighborValue };
        const cacheKey = generateCacheKey(neighborParams);
        
        // Only preload if not already cached
        if (!appState.imageCache.has(cacheKey)) {
            preloadPromises.push(
                fetchExperimentDataFromJson(neighborParams)
                    .then(urls => {
                        cacheImages(neighborParams, urls);
                    })
                    .catch(error => {
                        console.debug(`Preload failed for ${sliderId}=${neighborValue}:`, error);
                    })
            );
        }
    }
    
    // Preload in background, don't await
    if (preloadPromises.length > 0) {
        Promise.allSettled(preloadPromises).then(() => {
            console.debug(`Preloaded ${preloadPromises.length} neighbor parameter sets`);
        });
    }
}


// Add event listeners to output log controls
function addOutputLogControlListeners(outputLogId, controls) {
    controls.forEach(control => {
        const controlId = `${outputLogId}_${control.id}`;
        const element = document.getElementById(controlId);
        
        if (!element) return;
        
        // Add slider value display update for sliders
        if (control.type === 'slider') {
            const valueDisplay = document.getElementById(`${controlId}Value`);
            element.addEventListener('input', function() {
                if (valueDisplay) {
                    valueDisplay.textContent = this.value;
                }
                debouncedUpdateAll();
            });
        } else {
            // For other control types
            element.addEventListener('input', debouncedUpdateAll);
            element.addEventListener('change', debouncedUpdateAll);
        }
    });
}


// Generic control generation functions that can be used by both global and output log controls
function generateControlHTML(controlConfig, namespace = '') {
    const controlId = namespace ? `${namespace}_${controlConfig.id}` : controlConfig.id;
    
    switch (controlConfig.type) {
        case 'text':
            return `
                <div class="input-group">
                    <label for="${controlId}">${controlConfig.label}:</label>
                    <input type="text" 
                           id="${controlId}" 
                           placeholder="${controlConfig.placeholder || ''}"
                           value="${controlConfig.value || ''}">
                </div>
            `;
            
        case 'slider':
            // Use the existing slider generation logic
            const min = controlConfig.min || 0;
            const max = controlConfig.max || 100;
            const value = controlConfig.value || min;
            
            return `
                <div class="slider-container">
                    <div class="slider-label">
                        <span>${controlConfig.label}</span>
                        <span class="slider-value" id="${controlId}Value">${value}</span>
                    </div>
                    <input type="range"
                           id="${controlId}"
                           min="${min}"
                           max="${max}"
                           value="${value}"
                           step="${controlConfig.step || 'any'}">
                </div>
            `;
            
        case 'toggle':
            return `
                <div class="input-group toggle-control">
                    <label for="${controlId}">${controlConfig.label}:</label>
                    <input type="checkbox" 
                           id="${controlId}" 
                           ${controlConfig.value ? 'checked' : ''}>
                </div>
            `;
            
        case 'select':
            if (!controlConfig.options) return '';
            return `
                <div class="input-group">
                    <label for="${controlId}">${controlConfig.label}:</label>
                    <select id="${controlId}">
                        ${controlConfig.options.map(option => 
                            `<option value="${option.value}" ${option.selected ? 'selected' : ''}>
                                ${option.label}
                            </option>`
                        ).join('')}
                    </select>
                </div>
            `;
            
        default:
            return '';
    }
}

function addControlEventListener(controlConfig, controlId, debouncedUpdateAll) {
    const element = document.getElementById(controlId);
    
    if (!element) return;
    
    if (controlConfig.type === 'slider') {
        const valueDisplay = document.getElementById(`${controlId}Value`);
        element.addEventListener('input', function() {
            if (valueDisplay) {
                valueDisplay.textContent = this.value;
            }
            debouncedUpdateAll();
        });
    } else {
        element.addEventListener('input', debouncedUpdateAll);
        element.addEventListener('change', debouncedUpdateAll);
    }
}

// Refactored generateOutputLogs using the generic functions
function generateOutputLogs() {
    const container = document.getElementById('outputLogsContainer');
    
    config.outputLogs.forEach(outputLogConfig => {
        const controlsHTML = outputLogConfig.controls.map(control => 
            generateControlHTML(control, outputLogConfig.id)
        ).join('');
        
        const outputLogHTML = `
            <div class="output-log" id="${outputLogConfig.id}">
                <h3>${outputLogConfig.label}</h3>
                <div class="output-log-controls" id="${outputLogConfig.id}Controls">
                    ${controlsHTML}
                </div>

                <div class="images-grid">
                    <h4>Examples:</h4>
                    <div class="images-container" id="${outputLogConfig.id}Examples">
                        ${generateImagesGrid(outputLogConfig.examplePaths)}
                    </div>
                </div>
                
                <div class="images-grid">
                    <h4>Images:</h4>
                    <div class="images-container" id="${outputLogConfig.id}Images">
                        ${generateImagesGrid(outputLogConfig.imagePaths)}
                    </div>
                </div>
                
                <div class="json-display" id="${outputLogConfig.id}Json">Configure controls to see JSON</div>
            </div>
        `;
        
        container.insertAdjacentHTML('beforeend', outputLogHTML);
        
        // Add event listeners using generic function
        outputLogConfig.controls.forEach(control => {
            const controlId = `${outputLogConfig.id}_${control.id}`;
            addControlEventListener(control, controlId, debouncedUpdateAll);
        });
    });
}


function generateGlobalControls() {
    generateGlobalTextBoxes();
    generateGlobalSliders();
}

function generateGlobalTextBoxes() {
    const container = document.getElementById('globalTextBoxes');
    
    config.textBoxes.forEach(textBoxConfig => {
        // Convert textBoxConfig to the new format and use generic function
        const controlConfig = {
            type: 'text',
            id: textBoxConfig.id,
            label: textBoxConfig.label,
            placeholder: textBoxConfig.placeholder
        };
        
        container.insertAdjacentHTML('beforeend', generateControlHTML(controlConfig));
        
        const textBox = document.getElementById(textBoxConfig.id);
        textBox.addEventListener('input', debouncedUpdateAll);
    });
}


function generateGlobalSliders() {
    const container = document.getElementById('slidersContainer');
    
    config.sliders.forEach(sliderConfig => {
        const controlConfig = {
            type: 'slider',
            id: sliderConfig.id,
            label: sliderConfig.label,
            min: sliderConfig.min,  // Initial values from config
            max: sliderConfig.max,
            value: sliderConfig.value
        };
        
        container.insertAdjacentHTML('beforeend', generateControlHTML(controlConfig));
    });
    
}

async function setupAllSlidersWithDomains() {

    // Now set up domain snapping for ALL sliders in the DOM
    const allSliders = document.querySelectorAll('.slider-container input[type="range"]');
    
    // Extract all base slider IDs
    const baseSliderIds = Array.from(allSliders).map(slider => {
	let id = slider.id;           // or slider.sliderId — whatever your objects use

	if (id.includes('_')) {
	    const parts = id.split('_');
	    id = parts[parts.length - 1];
	}

	return id;
    });

    // // Fetch domains for all sliders
    const domainPromises = baseSliderIds.map(id => fetchSliderDomain(id));
    const domains = await Promise.all(domainPromises);
    // Store domains in appState
    baseSliderIds.forEach((sliderId, i) => {
        const sliderDomain = domains[i];

        if (sliderDomain && sliderDomain.length > 0) {
            appState.sliderDomains.set(sliderId, sliderDomain);
        }
    });


    // Setup domain for ALL sliders in dom
    allSliders.forEach((slider, i) => {
        const sliderId = slider.id;
        
        // Extract the base slider ID for domain lookup
        const baseSliderId = baseSliderIds[i]
        
	// // Store domains in appState
        const domain = appState.sliderDomains.get(baseSliderId);

        const valueDisplay = document.getElementById(`${sliderId}Value`);

        // Remove any existing event listeners to avoid duplicates
        const newSlider = slider.cloneNode(true);
        slider.parentNode.replaceChild(newSlider, slider);

        if (domain && domain.length > 0) {
            // Update min/max based on domain for visual consistency
            newSlider.min = Math.min(...domain);
            newSlider.max = Math.max(...domain);
            
            // Set initial value to closest domain value if needed
            const currentValue = parseFloat(newSlider.value);
            if (!domain.includes(currentValue)) {
                const closest = domain.reduce((a, b) => 
                    Math.abs(b - currentValue) < Math.abs(a - currentValue) ? b : a
                );
                newSlider.value = closest;
                if (valueDisplay) {
                    valueDisplay.textContent = closest;
                }
            }

            // Add domain-snapping event listener
            newSlider.addEventListener('input', function() {
                const closest = domain.reduce((a, b) => 
                    Math.abs(b - this.value) < Math.abs(a - this.value) ? b : a
                );
                this.value = closest;
                if (valueDisplay) {
                    valueDisplay.textContent = closest;
                }
                debouncedUpdateAll();
            });
        } else {
            // For sliders without domains, just add the update listener
            newSlider.addEventListener('input', function() {
                if (valueDisplay) {
                    valueDisplay.textContent = this.value;
                }
                debouncedUpdateAll();
            });
        }
    });
}

// Helper function to generate the images grid HTML
function generateImagesGrid(imagePaths) {
    if (!imagePaths || imagePaths.length === 0) {
        return '<div class="no-images">No images available</div>';
    }
    
    return imagePaths.map(imagePath => `
        <div class="image-item">
            <img src="${imagePath}" alt="Image" loading="lazy">
        </div>
    `).join('');
}

// Get all global values as JSON
function getTextBoxValuesJson() {
    const values = {};
    
    config.textBoxes.forEach(textBoxConfig => {
        const textBox = document.getElementById(textBoxConfig.id);
        if (textBox) {
            values[textBoxConfig.id] = textBox.value;
        }
    });
    
    return values;
}

// Get slider values as JSON 
function getSliderValuesJson() {
    const values = {};
    
    config.sliders.forEach(sliderConfig => {
        const slider = document.getElementById(sliderConfig.id);
        if (slider) {
            values[sliderConfig.id] = slider.value;
        }
    });

    return values;
}

// Get JSON values for a specific output log's controls
function getOutputLogValuesJson(outputLogId) {
    const outputLogConfig = config.outputLogs.find(log => log.id === outputLogId);
    if (!outputLogConfig) return {};
    
    const values = {};
    
    outputLogConfig.controls.forEach(control => {
        const controlId = `${outputLogId}_${control.id}`;
        const element = document.getElementById(controlId);
        
        if (!element) return;
        
        switch (control.type) {
            case 'text':
                values[control.id] = element.value; // This was missing!
                break;
                
            case 'slider':
                values[control.id] = parseFloat(element.value);
                break;
                
            case 'toggle':
                values[control.id] = element.checked;
                break;
                
            case 'select':
                values[control.id] = element.value;
                break;
        }
    });
    
    return values;
}


// Get global values as JSON -
function getGlobalValuesJson() {
    const textBoxValues = getTextBoxValuesJson();
    const sliderValues = getSliderValuesJson();

    // Convert slider values from strings to numbers
    const parsedSliderValues = {};
    Object.keys(sliderValues).forEach(key => {
	parsedSliderValues[key] = parseFloat(sliderValues[key]);
    });

    return {
	...textBoxValues,
	...parsedSliderValues,
    };
}

// Update a specific output log 
async function updateOutputLog(outputLogId) {
    const outputLogJson = document.getElementById(`${outputLogId}Json`);
    
    if (outputLogJson) {
	const globalValues = getGlobalValuesJson();
        const outputLogValues = getOutputLogValuesJson(outputLogId);
        
        const combinedJson = {
            ...globalValues,
            ...outputLogValues
        };
        
        outputLogJson.textContent = JSON.stringify(combinedJson, null, 2);
    }
}

// Update Images Grid 
async function updateImagesGrid(outputLogId) {
    // Cancel any ongoing request for this output log
    if (appState.abortControllers.has(outputLogId)) {
        appState.abortControllers.get(outputLogId).abort();
    }
    
    const outputLogImages = document.getElementById(`${outputLogId}Images`);
    
    if (!outputLogImages) return;
    
    const globalValues = getGlobalValuesJson();
    const outputLogValues = getOutputLogValuesJson(outputLogId);

    const combinedJson = {
        ...globalValues,
        ...outputLogValues
    };

    // Check cache first
    const cachedResult = getCachedImages(combinedJson);
    if (cachedResult) {
        appState.cacheHits++;
        updateCacheStats();
        outputLogImages.innerHTML = generateImagesGrid(cachedResult.urls);
        
        // Update cache metadata
        cachedResult.accessCount++;
        cachedResult.timestamp = Date.now();
        
        // Preload neighbors in background - FIXED: Use globalValues instead of sliderValues
        config.sliders.forEach(sliderConfig => {
            const sliderId = sliderConfig.id;
            if (globalValues[sliderId] !== undefined) {
                preloadNeighborImages(combinedJson, sliderId, globalValues[sliderId]);
            }
        });
        
        return;
    }

    appState.cacheMisses++;
    updateCacheStats();
    
    // Show loading state
    outputLogImages.innerHTML = '<div class="loading">Loading images...</div>';
    
    try {
        const controller = new AbortController();
        appState.abortControllers.set(outputLogId, controller);
        
        const urls = await fetchExperimentDataFromJson(combinedJson, controller.signal);
        
        if (!controller.signal.aborted) {
            // Cache the result
            cacheImages(combinedJson, urls);
            outputLogImages.innerHTML = generateImagesGrid(urls);
            
            // Preload neighbors in background - FIXED: Use globalValues instead of sliderValues
            config.sliders.forEach(sliderConfig => {
                const sliderId = sliderConfig.id;
                if (globalValues[sliderId] !== undefined) {
                    preloadNeighborImages(combinedJson, sliderId, globalValues[sliderId]);
                }
            });
        }
        
    } catch (error) {
        if (error.name !== 'AbortError') {
            console.error('Error fetching images:', error);
            outputLogImages.innerHTML = '<div class="error">Error loading images</div>';
        }
    } finally {
        appState.abortControllers.delete(outputLogId);
    }
}


// Enhanced preload function with better error handling
async function preloadNeighborImages(currentParams, sliderId, currentValue) {
    if (!CACHE_CONFIG.preloadNeighbors) return;
    
    const domain = appState.sliderDomains.get(sliderId);
    if (!domain || domain.length === 0) return;
    
    // Ensure currentValue is a number
    const currentNumValue = parseFloat(currentValue);
    if (isNaN(currentNumValue)) return;
    
    const currentIndex = domain.indexOf(currentNumValue);
    if (currentIndex === -1) return;
    
    const start = Math.max(0, currentIndex - CACHE_CONFIG.preloadRadius);
    const end = Math.min(domain.length - 1, currentIndex + CACHE_CONFIG.preloadRadius);
    
    const preloadPromises = [];
    
    for (let i = start; i <= end; i++) {
        if (i === currentIndex) continue; // Skip current value
        
        const neighborValue = domain[i];
        const neighborParams = { ...currentParams, [sliderId]: neighborValue };
        const cacheKey = generateCacheKey(neighborParams);
        
        // Only preload if not already cached
        if (!appState.imageCache.has(cacheKey)) {
            preloadPromises.push(
                fetchExperimentDataFromJson(neighborParams)
                    .then(urls => {
                        cacheImages(neighborParams, urls);
                    })
                    .catch(error => {
                        if (error.name !== 'AbortError') {
                            console.debug(`Preload failed for ${sliderId}=${neighborValue}:`, error);
                        }
                    })
            );
        }
    }
    
    // Preload in background, don't await
    if (preloadPromises.length > 0) {
        Promise.allSettled(preloadPromises).then((results) => {
            const successful = results.filter(r => r.status === 'fulfilled').length;
            console.debug(`Preloaded ${successful}/${preloadPromises.length} neighbor parameter sets`);
        });
    }
}

// Update all output logs
function updateAllOutputLogs() {
    config.outputLogs.forEach(outputLogConfig => {
        updateOutputLog(outputLogConfig.id);
    });
}

// Update all Images Grid
async function updateAllImagesGrid() {
    const updatePromises = config.outputLogs.map(outputLogConfig => 
        updateImagesGrid(outputLogConfig.id)
    );
    
    Promise.allSettled(updatePromises).then(results => {
        results.forEach((result, index) => {
            if (result.status === 'rejected' && result.reason.name !== 'AbortError') {
                console.error(`Failed to update images for ${config.outputLogs[index].id}:`, result.reason);
            }
        });
    });
}

// Debounced update function
function debouncedUpdateAll() {
    if (appState.updateTimeouts.has('global')) {
        clearTimeout(appState.updateTimeouts.get('global'));
    }
    
    updateAllOutputLogs();
    
    const timeoutId = setTimeout(() => {
        updateAllImagesGrid();
        appState.updateTimeouts.delete('global');
    }, 300);
    
    appState.updateTimeouts.set('global', timeoutId);
}

// Fetch functions (unchanged)
async function fetchExperimentDataFromJson(json, signal) {
    const res = await fetch("/get_experiment_urls", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(json),
        signal: signal
    });

    if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
    }
    
    const data = await res.json();   
    return data;
}

async function fetchSliderDomain(sliderId) {
    const res = await fetch("/get_slider_domain", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(sliderId)
    });

    if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
    }
    
    const data = await res.json();   
    return data;
}

