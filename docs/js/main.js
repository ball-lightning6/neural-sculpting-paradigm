document.addEventListener('DOMContentLoaded', () => {
    let currentLang = 'zh'; // 'zh' or 'en'
    let currentCategoryIndex = 0; // 当前选中的分类索引
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

        // 过滤掉没有脚本的空分类
        const filteredDocsData = docsData.filter(category => category.scripts && category.scripts.length > 0);

        // Render Categories Navigation
        const navUl = document.createElement('ul');
        filteredDocsData.forEach((category, index) => {
            const li = document.createElement('li');
            const a = document.createElement('a');
            a.href = '#';
            a.textContent = currentLang === 'zh' ? category.category_zh : category.category_en;
            
            // 添加点击事件来切换分类
            a.addEventListener('click', (e) => {
                e.preventDefault();
                currentCategoryIndex = index;
                renderScriptsContent(); // 只重新渲染脚本内容
                updateActiveNav(); // 更新导航栏高亮
            });
            
            // 如果是当前选中的分类，添加active类
            if (index === currentCategoryIndex) {
                a.classList.add('active');
            }
            
            li.appendChild(a);
            navUl.appendChild(li);
        });
        categoryNav.appendChild(navUl);

        // 渲染脚本内容
        renderScriptsContent();

        function renderScriptsContent() {
            // 清除脚本内容
            scriptsList.innerHTML = '';

            // 只渲染当前选中的分类
            const category = filteredDocsData[currentCategoryIndex];
            if (!category) return;

            const section = document.createElement('section');
            section.id = `cat-${currentCategoryIndex}`;
            section.className = 'category-section';

            const title = document.createElement('h2');
            title.className = 'category-title';
            title.textContent = currentLang === 'zh' ? category.category_zh : category.category_en;
            section.appendChild(title);

            category.scripts.forEach(script => {
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
                details.innerHTML = marked.parse(rawDetails);

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

        function updateActiveNav() {
            // 更新导航栏的高亮状态
            const navLinks = categoryNav.querySelectorAll('a');
            navLinks.forEach((link, index) => {
                if (index === currentCategoryIndex) {
                    link.classList.add('active');
                } else {
                    link.classList.remove('active');
                }
            });
        }
    }
});