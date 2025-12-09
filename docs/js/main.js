document.addEventListener('DOMContentLoaded', () => {
    let currentLang = 'zh'; // 'zh' or 'en'
    let currentTopCategory = 'datasetScripts'; // 'datasetScripts', 'toolScripts', 'trainingScripts'
    let currentSubCategoryIndex = 0; // 用于数据集脚本的子分类索引
    
    const langToggleBtn = document.getElementById('lang-toggle');
    const categoryNav = document.getElementById('category-nav');
    const scriptsList = document.getElementById('scripts-list');

    // Initialize
    render();

    // Event Listeners
    langToggleBtn.addEventListener('click', () => {
        currentLang = currentLang === 'zh' ? 'en' : 'zh';
        updateLangButton();
        render();
    });

    function updateLangButton() {
        langToggleBtn.textContent = currentLang === 'zh' ? 'Switch to English' : '切换到中文';
        document.documentElement.lang = currentLang === 'zh' ? 'zh-CN' : 'en';
    }

    function render() {
        // Clear existing content
        categoryNav.innerHTML = '';
        scriptsList.innerHTML = '';

        // ========== 渲染顶层分类导航 (三大类) ==========
        const topNavContainer = document.createElement('div');
        topNavContainer.className = 'top-nav-container';
        
        const topNavUl = document.createElement('ul');
        topNavUl.className = 'top-nav';
        
        const topCategories = [
            { key: 'datasetScripts', name_zh: docsData.datasetScripts.name_zh, name_en: docsData.datasetScripts.name_en },
            { key: 'toolScripts', name_zh: docsData.toolScripts.name_zh, name_en: docsData.toolScripts.name_en },
            { key: 'trainingScripts', name_zh: docsData.trainingScripts.name_zh, name_en: docsData.trainingScripts.name_en }
        ];
        
        topCategories.forEach(cat => {
            const li = document.createElement('li');
            const a = document.createElement('a');
            a.href = '#';
            a.textContent = currentLang === 'zh' ? cat.name_zh : cat.name_en;
            a.className = 'top-nav-link';
            
            if (cat.key === currentTopCategory) {
                a.classList.add('active');
            }
            
            a.addEventListener('click', (e) => {
                e.preventDefault();
                currentTopCategory = cat.key;
                currentSubCategoryIndex = 0; // 重置子分类索引
                render();
            });
            
            li.appendChild(a);
            topNavUl.appendChild(li);
        });
        
        topNavContainer.appendChild(topNavUl);
        categoryNav.appendChild(topNavContainer);

        // ========== 渲染子分类导航 (仅数据集脚本有子分类) ==========
        if (currentTopCategory === 'datasetScripts') {
            const subNavContainer = document.createElement('div');
            subNavContainer.className = 'sub-nav-container';
            
            const subNavUl = document.createElement('ul');
            subNavUl.className = 'sub-nav';
            
            const categories = docsData.datasetScripts.categories;
            
            categories.forEach((category, index) => {
                const li = document.createElement('li');
                const a = document.createElement('a');
                a.href = '#';
                a.textContent = currentLang === 'zh' ? category.category_zh : category.category_en;
                a.className = 'sub-nav-link';
                
                if (index === currentSubCategoryIndex) {
                    a.classList.add('active');
                }
                
                a.addEventListener('click', (e) => {
                    e.preventDefault();
                    currentSubCategoryIndex = index;
                    renderScriptsContent();
                    updateSubNav();
                });
                
                li.appendChild(a);
                subNavUl.appendChild(li);
            });
            
            subNavContainer.appendChild(subNavUl);
            categoryNav.appendChild(subNavContainer);
        }

        // 渲染脚本内容
        renderScriptsContent();

        function renderScriptsContent() {
            scriptsList.innerHTML = '';
            
            let scripts = [];
            let sectionTitle = '';
            
            if (currentTopCategory === 'datasetScripts') {
                const category = docsData.datasetScripts.categories[currentSubCategoryIndex];
                if (category) {
                    scripts = category.scripts;
                    sectionTitle = currentLang === 'zh' ? category.category_zh : category.category_en;
                }
            } else if (currentTopCategory === 'toolScripts') {
                scripts = docsData.toolScripts.scripts;
                sectionTitle = currentLang === 'zh' ? docsData.toolScripts.name_zh : docsData.toolScripts.name_en;
            } else if (currentTopCategory === 'trainingScripts') {
                scripts = docsData.trainingScripts.scripts;
                sectionTitle = currentLang === 'zh' ? docsData.trainingScripts.name_zh : docsData.trainingScripts.name_en;
            }
            
            if (!scripts || scripts.length === 0) {
                const emptyMsg = document.createElement('p');
                emptyMsg.className = 'empty-message';
                emptyMsg.textContent = currentLang === 'zh' ? '暂无脚本' : 'No scripts available';
                scriptsList.appendChild(emptyMsg);
                return;
            }

            const section = document.createElement('section');
            section.className = 'category-section';

            const title = document.createElement('h2');
            title.className = 'category-title';
            title.textContent = sectionTitle;
            section.appendChild(title);

            scripts.forEach(script => {
                const card = document.createElement('div');
                card.className = 'script-card';

                const header = document.createElement('div');
                header.className = 'script-header';
                
                const path = document.createElement('div');
                path.className = 'script-path';
                path.textContent = script.path;
                
                const expandBtn = document.createElement('button');
                expandBtn.className = 'expand-btn';
                expandBtn.textContent = currentLang === 'zh' ? '展开详情' : 'View Details';
                
                header.appendChild(path);
                header.appendChild(expandBtn);

                const desc = document.createElement('div');
                desc.className = 'script-desc';
                desc.textContent = currentLang === 'zh' ? script.description_zh : script.description_en;

                const details = document.createElement('div');
                details.className = 'script-details';
                // Use marked.js to render markdown content
                const rawDetails = currentLang === 'zh' ? script.details_zh : script.details_en;
                if (rawDetails) {
                    details.innerHTML = marked.parse(rawDetails);
                }

                // Expand/Collapse Logic
                expandBtn.addEventListener('click', () => {
                    const isActive = details.classList.contains('active');
                    if (isActive) {
                        details.classList.remove('active');
                        expandBtn.textContent = currentLang === 'zh' ? '展开详情' : 'View Details';
                    } else {
                        details.classList.add('active');
                        expandBtn.textContent = currentLang === 'zh' ? '收起详情' : 'Hide Details';
                    }
                });

                card.appendChild(header);
                card.appendChild(desc);
                card.appendChild(details);
                section.appendChild(card);
            });

            scriptsList.appendChild(section);
        }

        function updateSubNav() {
            const subNavLinks = categoryNav.querySelectorAll('.sub-nav-link');
            subNavLinks.forEach((link, index) => {
                if (index === currentSubCategoryIndex) {
                    link.classList.add('active');
                } else {
                    link.classList.remove('active');
                }
            });
        }
    }
});